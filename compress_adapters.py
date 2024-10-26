## Usage
# python compress_adapters.py --input adapters.safetensors --output compressed.safetensors --rank 32


import numpy as np
import torch
from typing import List, Tuple, Dict
from safetensors import safe_open
from safetensors.torch import save_file

def load_safetensors(file_path: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load LoRA adapters from a safetensors file.
    
    Args:
        file_path: Path to the safetensors file
        
    Returns:
        Dictionary mapping adapter names to (A, B) matrix pairs
    """
    adapters = {}
    with safe_open(file_path, framework="numpy") as f:
        keys = f.keys()
        for key in keys:
            if key.endswith("a"):  # LoRA A matrix
                name = key[:-1]
                A = f.get_tensor(key)
                B_key = f"{name}b"
                if B_key in keys:
                    B = f.get_tensor(B_key)
                    adapters[name] = (A, B)
    return adapters

def compress_lora_adapters(adapters: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                          target_rank: int = 32, 
                          max_iter: int = 100) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Compress LoRA adapters using joint diagonalization.
    
    Args:
        adapters: Dictionary mapping adapter names to (A, B) matrix pairs
        target_rank: Target rank for compression
        max_iter: Maximum number of iterations for optimization
        
    Returns:
        Dictionary mapping adapter names to compressed (U, Sigma, V) triplets
    """
    # Group adapters by shape dimensions
    grouped = {}
    for name, (A, B) in adapters.items():
        shape_key = (A.shape, B.shape)
        if shape_key not in grouped:
            grouped[shape_key] = {}
        grouped[shape_key][name] = (A, B)

    compressed = {}
    for group in grouped.values():
        As, Bs = zip(*group.values())
        try:
            # Get dimensions
            m, k = Bs[0].shape
            n = As[0].shape[1]
            r = min(target_rank, k, m, n)

            # Initialize random orthogonal matrices
            U = np.random.randn(m, r)
            V = np.random.randn(n, r)
            U, _ = np.linalg.qr(U)
            V, _ = np.linalg.qr(V)

            # Alternating optimization
            for _ in range(max_iter):
                # Update U
                M = sum(B @ A @ V @ V.T @ A.T @ B.T for A, B in zip(As, Bs))
                U, _ = np.linalg.qr(M @ U)

                # Update V
                N = sum(A.T @ B.T @ U @ U.T @ B @ A for A, B in zip(As, Bs))
                V, _ = np.linalg.qr(N @ V)

            # Compute adapter-specific coefficients
            for i, name in enumerate(group.keys()):
                Sigma = U.T @ Bs[i] @ As[i] @ V
                compressed[name] = (U, Sigma, V.T)
                
        except Exception as e:
            print(f"Error compressing adapter group: {str(e)}")
            continue

    return compressed

def save_compressed_adapters(compressed: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]], 
                           file_path: str):
    """Save compressed adapters to a safetensors file.
    
    Args:
        compressed: Dictionary mapping adapter names to (U, Sigma, V) triplets
        file_path: Output file path
    """
    tensors = {}
    print(f"\nPreparing {len(compressed)} adapters for saving:")
    
    for name, (U, S, V) in compressed.items():
        # Convert to torch tensors and ensure proper format
        tensors[f"{name}_U"] = torch.from_numpy(U).contiguous().clone().to(torch.float32)
        tensors[f"{name}_S"] = torch.from_numpy(S).contiguous().clone().to(torch.float32)
        tensors[f"{name}_V"] = torch.from_numpy(V).contiguous().clone().to(torch.float32)
        
        print(f"  {name}:")
        print(f"    U shape: {tensors[f'{name}_U'].shape}")
        print(f"    S shape: {tensors[f'{name}_S'].shape}")
        print(f"    V shape: {tensors[f'{name}_V'].shape}")

    try:
        save_file(tensors, file_path)
        print(f"Successfully saved compressed adapters to {file_path}")
    except Exception as e:
        print(f"Error saving compressed adapters: {str(e)}")
        raise

def main():
    """Example usage of adapter compression."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compress LoRA adapters')
    parser.add_argument('--input', required=True, help='Input safetensors file containing LoRA adapters')
    parser.add_argument('--output', required=True, help='Output file path for compressed adapters')
    parser.add_argument('--rank', type=int, default=32, help='Target rank for compression')
    parser.add_argument('--iterations', type=int, default=100, help='Number of optimization iterations')
    args = parser.parse_args()

    # Load adapters
    print(f"Loading adapters from {args.input}")
    adapters = load_safetensors(args.input)
    print(f"Loaded {len(adapters)} adapters")

    # Compress adapters
    print(f"\nCompressing adapters to rank {args.rank}")
    compressed = compress_lora_adapters(
        adapters, 
        target_rank=args.rank,
        max_iter=args.iterations
    )
    print(f"Compressed {len(compressed)} adapters")

    # Save compressed adapters
    save_compressed_adapters(compressed, args.output)

if __name__ == "__main__":
    main()