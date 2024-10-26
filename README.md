
# SOPPU: Scalable One PEFT per User

SOPPU is a framework for decentralized, personalized AI that enables efficient management of individual user adaptations. It combines:
- Federated training from [FedBiOT](https://github.com/HarliWu/FedBiOT)
- Adapter compression (our contribution)
- Dynamic serving from [LoRAX](https://github.com/predibase/lorax)

## Quick Start

### Installation
```bash
pip install torch numpy safetensors
```

### Basic Usage

```python
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

def compress_lora_adapters(adapters, target_rank=32, max_iter=100):
    """Compress LoRA adapters using joint diagonalization.
    
    Args:
        adapters: Dict of adapter name to (A, B) matrices
        target_rank: Target rank for compression (default: 32)
        max_iter: Number of iterations (default: 100)
        
    Returns:
        Dict of adapter name to (U, Sigma, V) compressed representation
    """
    # Group adapters by shape
    grouped = {}
    for name, (A, B) in adapters.items():
        shape_key = (A.shape, B.shape)
        if shape_key not in grouped:
            grouped[shape_key] = {}
        grouped[shape_key][name] = (A, B)

    compressed = {}
    for group in grouped.values():
        As, Bs = zip(*group.values())
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

    return compressed

# Example usage
if __name__ == "__main__":
    # Load your LoRA adapters
    adapters = {}  # Load your adapters here
    
    # Compress adapters
    compressed = compress_lora_adapters(adapters)
    
    # Save compressed adapters
    tensors = {}
    for name, (U, S, V) in compressed.items():
        tensors[f"{name}_U"] = torch.from_numpy(U).contiguous()
        tensors[f"{name}_S"] = torch.from_numpy(S).contiguous()
        tensors[f"{name}_V"] = torch.from_numpy(V).contiguous()
    
    save_file(tensors, "compressed_adapters.safetensors")
```

## Architecture

SOPPU consists of three main components:

1. **FedBiOT Integration** [GitHub](https://github.com/HarliWu/FedBiOT)
   - Federated learning without full model access
   - Privacy-preserving personalization
   - Client-side LoRA training

2. **Compression (Our Contribution)**
   - Joint diagonalization of LoRA adapters
   - Shared basis optimization
   - Memory-efficient storage

3. **LoRAX Integration** [GitHub](https://github.com/predibase/lorax)
   - Dynamic adapter loading
   - Efficient request batching
   - Memory management

## How It Works

1. **Training**: Users train personal LoRA adapters using FedBiOT
2. **Compression**: Adapters are compressed using joint diagonalization
3. **Serving**: Compressed adapters are served efficiently using LoRAX

## References

1. FedBiOT: [FedBiOT: LLM Local Fine-tuning in Federated Learning without Full Model](https://github.com/HarliWu/FedBiOT)
2. LoRAX: [LoRAX: Serving Thousands of Concurrent LoRA Adapters](https://github.com/predibase/lorax)

<!-- ## Citation
```bibtex
[Will be added after publication]
```
``` -->
