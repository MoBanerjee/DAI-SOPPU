import numpy as np
import torch
from typing import List, Tuple, Dict, Optional
from safetensors import safe_open
from safetensors.torch import save_file
import logging
import os
import re
import time  # Import time for measuring execution time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_layer_name(key: str) -> Optional[Tuple[int, str, str]]:
    """Parse layer index and type from a key name."""
    
    match = re.search(r'base_model.model.model.layers\.(\d+)\.self_attn\.(\w+)_proj\.lora_([ABab]).weight', key)
    if match:
        layer_idx = int(match.group(1))
        proj_type = match.group(2)  # q or v
        lora_type = match.group(3).upper() # A or B
        return layer_idx, proj_type, lora_type
    return None

def load_safetensors(file_paths: List[str]) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """Load multiple adapter files and group them by adapter file."""
    all_adapters = {}
    for file_path in file_paths:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            continue
        try:
            adapters = {}
            base_file = os.path.splitext(os.path.basename(file_path))[0]
            logger.info(f"Loading file: {base_file}")
            with safe_open(file_path, framework="numpy") as f:
                keys = list(f.keys())

                layer_groups = {}
                for key in keys:
              
                    parsed = parse_layer_name(key)
                    if parsed:
                        layer_idx, proj_type, lora_type = parsed
                        group_key = (layer_idx, proj_type)
                        if group_key not in layer_groups:
                            layer_groups[group_key] = {'A': None, 'B': None}
                        layer_groups[group_key][lora_type] = f.get_tensor(key)
                        
                for (layer_idx, proj_type), weights in layer_groups.items():
                    if weights['A'] is not None and weights['B'] is not None:
                        adapter_name = f"layer_{layer_idx}_{proj_type}_proj"
                        adapters[adapter_name] = (weights['A'], weights['B'])
            if adapters:
                all_adapters[base_file] = adapters
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            continue
    return all_adapters

def joint_diagonalization(As: List[np.ndarray], Bs: List[np.ndarray], r: int, max_iter: int = 100) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Perform joint diagonalization on multiple adapter pairs."""
    m, k = Bs[0].shape
    n = As[0].shape[1]
    r = min(r, k, m, n)
    U = np.random.randn(m, r)
    V = np.random.randn(n, r)
    U, _ = np.linalg.qr(U)
    V, _ = np.linalg.qr(V)

    for _ in range(max_iter):
        M = sum(B @ A @ V @ V.T @ A.T @ B.T for A, B in zip(As, Bs))
        U, _ = np.linalg.qr(M @ U)
        N = sum(A.T @ B.T @ U @ U.T @ B @ A for A, B in zip(As, Bs))
        V, _ = np.linalg.qr(N @ V)

    Sigmas = [U.T @ B @ A @ V for A, B in zip(As, Bs)]
    return U, V, Sigmas

def compress_adapters(adapters: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]], rank: int) -> Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, Dict[str, Dict[str, np.ndarray]]]]:
    """Compress adapters grouped by shape."""
    start_time = time.time()  # Start timing
    shape_groups = {}
    for file, layers in adapters.items():
        for layer, (A, B) in layers.items():
            shape_key = (A.shape[1], B.shape[0])
            shape_groups.setdefault(shape_key, []).append((file, layer, A, B))

    compressed = {}
    for shape, group in shape_groups.items():
        As, Bs, file_layer_map = [], [], {}
        for file, layer, A, B in group:
            
            As.append(A)
            Bs.append(B)
            file_layer_map.setdefault(file, {})[layer] = len(As) - 1

        U, V, Sigmas = joint_diagonalization(As, Bs, rank)
        compressed[shape] = (U, V, {file: {layer: Sigmas[idx] for layer, idx in layers.items()} for file, layers in file_layer_map.items()})
    end_time = time.time()
    print(f"Compression completed in {end_time - start_time:.2f} seconds.")
    return compressed

def save_compressed_adapters(compressed: Dict, file_path: str):
    """Save compressed adapters efficiently."""
    tensors = {}
    for shape, (U, V, sigma_dict) in compressed.items():
        shape_key = f"{shape[0]}x{shape[1]}"
        tensors[f"U_{shape_key}"] = torch.from_numpy(U)
        tensors[f"V_{shape_key}"] = torch.from_numpy(V)
        for file, layers in sigma_dict.items():
            for layer, sigma in layers.items():
                tensors[f"{file}.{layer}.Sigma_{shape_key}"] = torch.from_numpy(sigma)
    save_file(tensors, file_path)

def evaluate_compression(original_adapters, compressed_file, early_layer_threshold=6):
    """
    Evaluate compression efficiency and reconstruction errors.
    Outputs:
      - Total parameters (original vs compressed)
      - Compression ratio
      - Parameter reduction percentage
      - Mean and std reconstruction errors (q_proj vs v_proj, early vs later layers)
      - Detailed per-layer reconstruction errors
    """
    total_original_params = sum(sum(A.size + B.size for A, B in layers.values()) for layers in original_adapters.values())
    total_compressed_params = 0
    q_errors, v_errors, early_errors, later_errors = [], [], [], []
    layerwise_errors = {}

    with safe_open(compressed_file, framework="pt") as f:
        U_params, V_params, sigma_params = 0, 0, 0
        
        # Count U, V, and Sigma parameters
        for key in f.keys():
            if key.startswith("U_"):
                U_params += f.get_tensor(key).numel()
            elif key.startswith("V_"):
                V_params += f.get_tensor(key).numel()
        
        # Calculate reconstruction errors
        for file, layers in original_adapters.items():
            layerwise_errors[file] = {}
            for layer, (A, B) in layers.items():
                shape = f"{A.shape[1]}x{B.shape[0]}"
                U = f.get_tensor(f"U_{shape}").numpy()
                V = f.get_tensor(f"V_{shape}").numpy()
                Sigma_key = f"{file}.{layer}.Sigma_{shape}"
                Sigma = f.get_tensor(Sigma_key).numpy()
                sigma_params += Sigma.size

                # Reconstruct and calculate error
                BA_reconstructed = U @ Sigma @ V.T
                error = np.linalg.norm(B @ A - BA_reconstructed) / np.linalg.norm(B @ A)

                # Store errors for per-layer analysis
                layerwise_errors[file][layer] = error

                # Classify errors for statistics
                proj_type = "q" if "q_proj" in layer else "v"
                (q_errors if proj_type == "q" else v_errors).append(error)
                layer_idx = int(layer.split('_')[1])
                (early_errors if layer_idx < early_layer_threshold else later_errors).append(error)

    total_compressed_params = U_params + V_params + sigma_params

    # Print cumulative results
    print(f"\n--- Compression Metrics ---")
    print(f"Total Original Parameters: {total_original_params}")
    print(f"Total Compressed Parameters: {total_compressed_params}")
    print(f"Compression Ratio: {total_original_params / total_compressed_params:.2f}")
    print(f"Parameter Reduction: {100 * (1 - total_compressed_params / total_original_params):.2f}%\n")

    # Print reconstruction error statistics
    print(f"--- Reconstruction Error Statistics ---")
    print(f"q_proj: Mean={np.mean(q_errors):.6f}, Std={np.std(q_errors):.6f}")
    print(f"v_proj: Mean={np.mean(v_errors):.6f}, Std={np.std(v_errors):.6f}")
    print(f"Early Layers: Mean={np.mean(early_errors):.6f}, Std={np.std(early_errors):.6f}")
    print(f"Later Layers: Mean={np.mean(later_errors):.6f}, Std={np.std(later_errors):.6f}\n")

    # Print detailed per-layer errors
    print(f"--- Per-Layer Reconstruction Errors ---")
    for file, layers in layerwise_errors.items():
        print(f"Adapter File: {file}")
        for layer, error in layers.items():
            print(f"  {layer}: Reconstruction Error = {error:.6f}")
        avg_file_error = np.mean(list(layers.values()))
        print(f"  Average Error for {file}: {avg_file_error:.6f}\n")


def reconstruct_original_adapter(compressed_file: str, target_file: str, output_file: str):
   
    start_time = time.time()
    U_matrices, V_matrices, Sigmas = {}, {}, {}

 
    with safe_open(compressed_file, framework="pt") as f:
        for key in f.keys():
            if key.startswith("U_"):
                U_matrices[key.split("_", 1)[1]] = f.get_tensor(key).numpy()
            elif key.startswith("V_"):
                V_matrices[key.split("_", 1)[1]] = f.get_tensor(key).numpy()
            elif key.startswith(target_file):
                _, layer, shape_key = key.split(".")
                shape_key = shape_key.split("_")[1]
                Sigmas[layer] = (f.get_tensor(key).numpy(), shape_key)


    tensors = {}
    for layer, (Sigma, shape) in Sigmas.items():
        U, V = U_matrices[shape], V_matrices[shape]

        # Parse projection type (q_proj or v_proj) from layer name
        match = re.match(r'layer_(\d+)_(\w+)_proj', layer)
        if not match:
            raise ValueError(f"Invalid layer format: {layer}")
        layer_idx, proj_type = match.groups()  # Extract the layer index and proj_type

        # Reconstruct BA
        BA = U @ Sigma @ V.T
        r = Sigma.shape[0]

        # Map to correct projection type
        tensors[f"model.layers.{layer_idx}.self_attn.{proj_type}_proj.lora_a"] = torch.from_numpy(np.eye(r, V.shape[0]))
        tensors[f"model.layers.{layer_idx}.self_attn.{proj_type}_proj.lora_b"] = torch.from_numpy(BA[:, :r])

    # Save the reconstructed file
    save_file(tensors, output_file)
    print(f"Reconstruction completed in {time.time() - start_time:.2f} seconds.")


def main():
    input_files = ["adapters.safetensors", "adapters2.safetensors"]
    adapters = load_safetensors(input_files)
    rank = 32
    compressed = compress_adapters(adapters, rank)
    compressed_file = "compressed_adapters.safetensors"
    save_compressed_adapters(compressed, compressed_file)
    evaluate_compression(adapters, compressed_file)
    reconstruct_original_adapter(compressed_file, "adapters", "reconstructed_adapters.safetensors")

if __name__ == "__main__":
    main()
