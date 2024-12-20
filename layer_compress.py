import numpy as np
import torch
from typing import List, Tuple, Dict, Optional
from safetensors import safe_open
from safetensors.torch import save_file
import logging
import os
import re
import time  

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_layer_name(key: str) -> Optional[Tuple[str]]:
    """Parse layer index and projection type from a key name."""
    match = re.search(r'base_model\.model\.model\.layers\.(\d+)\.self_attn\.(\w+)_proj\.lora_([ABab])\.weight', key)
    if match:
        layer_idx = match.group(1)
        proj_type = match.group(2)  # q or v
        lora_type = match.group(3).upper()  # A or B
        return f"layer_{layer_idx}_{proj_type}_proj", lora_type
    return None

def load_safetensors(file_paths: List[str]) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """Load multiple adapter files and group them by layer name."""
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
                        layer_name, lora_type = parsed
                        if layer_name not in layer_groups:
                            layer_groups[layer_name] = {'A': None, 'B': None}
                        layer_groups[layer_name][lora_type] = f.get_tensor(key)

                for layer_name, weights in layer_groups.items():
                    if weights['A'] is not None and weights['B'] is not None:
                        adapters[layer_name] = (weights['A'], weights['B'])
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

def compress_adapters(adapters: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]], rank: int) -> Dict[str, Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]]:
    """Compress adapters grouped by layer name."""
    start_time = time.time()
    layer_groups = {}
    for file, layers in adapters.items():
        for layer_name, (A, B) in layers.items():
            if layer_name not in layer_groups:
                layer_groups[layer_name] = []
            layer_groups[layer_name].append((file, A, B))

    compressed = {}
    for layer_name, group in layer_groups.items():
        As, Bs, file_map = [], [], {}
        for idx, (file, A, B) in enumerate(group):
            As.append(A)
            Bs.append(B)
            file_map[file] = idx

        U, V, Sigmas = joint_diagonalization(As, Bs, rank)
        compressed[layer_name] = (U, V, {file: Sigmas[file_map[file]] for file in file_map})

    print(f"Compression completed in {time.time() - start_time:.2f} seconds.")
    return compressed

def save_compressed_adapters(compressed: Dict, file_path: str):
    """Save compressed adapters efficiently."""
    tensors = {}
    for layer_name, (U, V, sigmas) in compressed.items():
        tensors[f"{layer_name}.U"] = torch.from_numpy(U)
        tensors[f"{layer_name}.V"] = torch.from_numpy(V)
        for file, sigma in sigmas.items():
            tensors[f"{file}.{layer_name}.Sigma"] = torch.from_numpy(sigma)
    save_file(tensors, file_path)

def evaluate_compression(original_adapters: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]], compressed_file: str):
    """Evaluate compression efficiency and reconstruction errors."""
    total_original_params = sum(sum(A.size + B.size for A, B in layers.values()) for layers in original_adapters.values())
    total_compressed_params = 0
    q_errors, v_errors = [], []
    early_errors, middle_errors, later_errors = [], [], []
    layerwise_errors = {}

    with safe_open(compressed_file, framework="pt") as f:
        for key in f.keys():
            if key.endswith(".U") or key.endswith(".V"):
                total_compressed_params += f.get_tensor(key).numel()

        for file, layers in original_adapters.items():
            layerwise_errors[file] = {}
            for layer, (A, B) in layers.items():
                U = f.get_tensor(f"{layer}.U").numpy()
                V = f.get_tensor(f"{layer}.V").numpy()
                Sigma = f.get_tensor(f"{file}.{layer}.Sigma").numpy()
                BA_reconstructed = U @ Sigma @ V.T
                error = np.linalg.norm(B @ A - BA_reconstructed) / np.linalg.norm(B @ A)

                # Store errors for statistics
                layerwise_errors[file][layer] = error
                layer_idx, proj_type = map(lambda x: int(x) if x.isdigit() else x, re.match(r"layer_(\d+)_(\w+)_proj", layer).groups())
                (q_errors if proj_type == "q" else v_errors).append(error)
                if layer_idx < 8:
                    early_errors.append(error)
                elif layer_idx < 16:
                    middle_errors.append(error)
                else:
                    later_errors.append(error)

    # Print compression metrics
    print("\n--- Compression Metrics ---")
    print(f"Total Original Parameters: {total_original_params}")
    print(f"Total Compressed Parameters: {total_compressed_params}")
    print(f"Compression Ratio: {total_original_params / total_compressed_params:.2f}")
    print(f"Parameter Reduction: {100 * (1 - total_compressed_params / total_original_params):.2f}%")

    # Print reconstruction error statistics
    print(f"\n--- Reconstruction Error Statistics ---")
    print(f"q_proj: Mean={np.mean(q_errors):.6f}, Std={np.std(q_errors):.6f}")
    print(f"v_proj: Mean={np.mean(v_errors):.6f}, Std={np.std(v_errors):.6f}")
    print(f"Early Layers: Mean={np.mean(early_errors):.6f}, Std={np.std(early_errors):.6f}")
    print(f"Middle Layers: Mean={np.mean(middle_errors):.6f}, Std={np.std(middle_errors):.6f}")
    print(f"Later Layers: Mean={np.mean(later_errors):.6f}, Std={np.std(later_errors):.6f}")

    # Print detailed layer-wise errors
    print("\n--- Per-Layer Reconstruction Errors ---")
    for file, layers in layerwise_errors.items():
        print(f"Adapter File: {file}")
        for layer, error in layers.items():
            print(f"  {layer}: Reconstruction Error = {error:.6f}")
        avg_file_error = np.mean(list(layers.values()))
        print(f"  Average Error for {file}: {avg_file_error:.6f}\n")
def main():
    input_files = ["adapters.safetensors", "adapters2.safetensors","adapters3.safetensors","adapters4.safetensors","adapters5.safetensors"]
    adapters = load_safetensors(input_files)
    rank = 8
    compressed = compress_adapters(adapters, rank)
    compressed_file = "compressed_adapters.safetensors"
    save_compressed_adapters(compressed, compressed_file)
    evaluate_compression(adapters, compressed_file)

if __name__ == "__main__":
    main()
