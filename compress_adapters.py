import numpy as np
import torch
from typing import List, Tuple, Dict
from safetensors import safe_open
from safetensors.torch import save_file

def load_safetensors(file_path: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    adapters = {}
    with safe_open(file_path, framework="numpy") as f:
        keys = list(f.keys())
        print(f"Available keys: {keys}")  # Debug print to see actual keys
        
        for key in keys:
            if 'lora_A.weight' in key:
                name = key.replace('lora_A.weight', '')
                A = f.get_tensor(key)
                B_key = f"{name}lora_B.weight"
                if B_key in keys:
                    B = f.get_tensor(B_key)
                    adapters[name] = (A, B)
                    print(f"Found adapter pair for {name}")  # Debug print

    return adapters

def group_adapters_by_shape(adapters: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict[Tuple[Tuple[int, int], Tuple[int, int]], Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    grouped = {}
    for name, (A, B) in adapters.items():
        shape_key = (A.shape, B.shape)
        if shape_key not in grouped:
            grouped[shape_key] = {}
        grouped[shape_key][name] = (A, B)
    return grouped

def joint_diagonalization_full(As: List[np.ndarray], Bs: List[np.ndarray], r: int, max_iter: int = 100) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    m, k = Bs[0].shape  # B is (m, k)
    n = As[0].shape[1]  # A is (k, n)

    # Print shapes for debugging
    print(f"Matrix shapes: B: {Bs[0].shape}, A: {As[0].shape}")
    
    r = min(r, k, m, n)
    print(f"Using rank {r}")

    # Initialize random orthogonal matrices
    U = np.random.randn(m, r)
    V = np.random.randn(n, r)
    U, _ = np.linalg.qr(U)
    V, _ = np.linalg.qr(V)

    for _ in range(max_iter):
        M = sum(B @ A @ V @ V.T @ A.T @ B.T for A, B in zip(As, Bs))
        U, _ = np.linalg.qr(M @ U)

        N = sum(A.T @ B.T @ U @ U.T @ B @ A for A, B in zip(As, Bs))
        V, _ = np.linalg.qr(N @ V)

    # Compute compressed representations - shape should be (r, r)
    Sigmas = [U.T @ B @ A @ V for A, B in zip(As, Bs)]
    print(f"Sigma shape before reshape: {Sigmas[0].shape}")

    # No reshape needed - Sigmas should already be (r, r)
    return U, V, Sigmas

def compress_lora_adapters(adapters: Dict[str, Tuple[np.ndarray, np.ndarray]], method: str, r: int) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    grouped_adapters = group_adapters_by_shape(adapters)
    compressed = {}

    for shape, group in grouped_adapters.items():
        As, Bs = zip(*group.values())
        try:
            if shape[1][1] != shape[0][0]:
                continue
            U, V, Sigmas = joint_diagonalization_full(list(As), list(Bs), r)
            for i, name in enumerate(group.keys()):
                compressed[name] = (U, Sigmas[i], V.T)
        except Exception as e:
            print(f"Error compressing shape group {shape}: {str(e)}")

    return compressed


# def save_compressed_adapters(compressed: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]], file_path: str):
#     tensors = {}
#     for name, (U, S, V) in compressed.items():
#         # Convert to torch tensors and ensure they are contiguous
#         tensors[f"{name}_U"] = torch.from_numpy(U).contiguous()
#         tensors[f"{name}_S"] = torch.from_numpy(S).contiguous()
#         tensors[f"{name}_V"] = torch.from_numpy(V).contiguous()

#     # Use safetensors to save the file
#     save_file(tensors, file_path)


def load_compressed_adapters(file_path: str) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    with safe_open(file_path, framework="pt") as f:
        tensors = {key: f.get_tensor(key) for key in f.keys()}

    compressed = {}
    for key, tensor in tensors.items():
        name = key.rsplit('_', 1)[0]
        if name not in compressed:
            compressed[name] = (None, None, None)
        if key.endswith("_U"):
            compressed[name] = (tensor, compressed[name][1], compressed[name][2])
        elif key.endswith("_S"):
            compressed[name] = (compressed[name][0], tensor, compressed[name][2])
        elif key.endswith("_V"):
            compressed[name] = (compressed[name][0], compressed[name][1], tensor)
    return compressed

def save_compressed_adapters(compressed: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]], file_path: str):
    tensors = {}
    print("\nPreparing tensors for saving:")
    for name, (U, S, V) in compressed.items():


        # Convert to torch tensors, ensure they are contiguous, create a copy, and cast to float32
        tensors[f"{name}_U"] = torch.from_numpy(U).contiguous().clone().to(torch.float32)
        tensors[f"{name}_S"] = torch.from_numpy(S).contiguous().clone().to(torch.float32)
        tensors[f"{name}_V"] = torch.from_numpy(V).contiguous().clone().to(torch.float32)

        print(f"  {name}:")
        print(f"    U shape: {tensors[f'{name}_U'].shape}, dtype: {tensors[f'{name}_U'].dtype}")
        print(f"    S shape: {tensors[f'{name}_S'].shape}, dtype: {tensors[f'{name}_S'].dtype}")
        print(f"    V shape: {tensors[f'{name}_V'].shape}, dtype: {tensors[f'{name}_V'].dtype}")

    print(f"\nAttempting to save {len(tensors)} tensors to {file_path}")

    try:
        # Use safetensors to save the file
        save_file(tensors, file_path)
        print("File saved successfully")
    except Exception as e:
        print(f"Error in save_file: {str(e)}")
        print("Tensor details:")
        for key, tensor in tensors.items():
            print(f"  {key}: shape {tensor.shape}, dtype {tensor.dtype}, device {tensor.device}")
        raise  # Re-raise the exception for the main function to catch

def main():
    input_file = "adapter_model.safetensors"  
    adapters = load_safetensors(input_file)

    if not adapters:
        print("No LoRA adapters found in the input file.")
        return

    print(f"Loaded {len(adapters)} LoRA adapters:")
    for name, (A, B) in adapters.items():
        print(f"  {name}: A shape {A.shape}, B shape {B.shape}")

    r_jd = 32
    compressed_combined = {}

    try:
        compressed_q = compress_lora_adapters(adapters, "q_proj", r_jd)
        if compressed_q:
            compressed_combined.update(compressed_q)
            print("\nJoint Diagonalization Compression for q_proj successful.")
            print(f"Compressed {len(compressed_q)} q_proj adapters.")
        else:
            print("\nNo q_proj adapters were compressed.")
    except Exception as e:
        print(f"\nError during Joint Diagonalization compression for q_proj: {str(e)}")

    try:
        compressed_v = compress_lora_adapters(adapters, "v_proj", r_jd)
        if compressed_v:
            compressed_combined.update(compressed_v)
            print("\nJoint Diagonalization Compression for v_proj successful.")
            print(f"Compressed {len(compressed_v)} v_proj adapters.")
        else:
            print("\nNo v_proj adapters were compressed.")
    except Exception as e:
        print(f"\nError during Joint Diagonalization compression for v_proj: {str(e)}")

    if compressed_combined:
        output_file = "compressed_combined_adapters.safetensors"
        try:
            save_compressed_adapters(compressed_combined, output_file)
            print(f"\nSaved combined compressed adapters to {output_file}")
        except Exception as e:
            print(f"\nError saving combined compressed adapters: {str(e)}")
    else:
        print("\nNo adapters were compressed. No output file was created.")

if __name__ == "__main__":
    main()