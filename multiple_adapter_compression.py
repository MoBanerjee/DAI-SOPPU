import numpy as np
import torch
from typing import List, Tuple, Dict, Optional, Set
from safetensors import safe_open
from safetensors.torch import save_file
import logging
import os
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_layer_name(key: str) -> Optional[Tuple[int, str, str]]:
    """Parse layer index and type from a key name."""
    match = re.search(r'layers\.(\d+)\.self_attn\.(\w+)_proj\.lora_([AB])\.weight', key)
    if match:
        layer_idx = int(match.group(1))
        proj_type = match.group(2)  # q or v
        lora_type = match.group(3)  # A or B
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
            logger.info(f"Loading file: {file_path}")
            
            with safe_open(file_path, framework="numpy") as f:
                keys = list(f.keys())
                logger.info(f"Found {len(keys)} keys")
                
                # Group keys by layer and projection type
                layer_groups = {}
                for key in keys:
                    parsed = parse_layer_name(key)
                    if parsed:
                        layer_idx, proj_type, lora_type = parsed
                        group_key = (layer_idx, proj_type)
                        if group_key not in layer_groups:
                            layer_groups[group_key] = {'A': None, 'B': None}
                        layer_groups[group_key][lora_type] = f.get_tensor(key)
                
                # Create adapter pairs
                for (layer_idx, proj_type), weights in layer_groups.items():
                    if weights['A'] is not None and weights['B'] is not None:
                        adapter_name = f"layer_{layer_idx}_{proj_type}_proj"
                        adapters[adapter_name] = (weights['A'], weights['B'])
                        logger.info(f"Loaded adapter pair for {adapter_name}")
                        logger.info(f"Shape A: {weights['A'].shape}, Shape B: {weights['B'].shape}")
            
            if adapters:
                all_adapters[file_path] = adapters
                logger.info(f"Successfully loaded {len(adapters)} adapter pairs from {file_path}")
            else:
                logger.warning(f"No valid adapter pairs found in {file_path}")
                
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            continue
    
    return all_adapters

def get_unique_layer_names(adapters_dict: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]) -> Set[str]:
    """Extract unique layer names across all adapter files."""
    layer_names = set()
    for adapters in adapters_dict.values():
        layer_names.update(adapters.keys())
    logger.info(f"Found unique layer names: {layer_names}")
    return layer_names

def group_adapters_by_layer(adapters_dict: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]) -> Dict[str, List[Tuple[np.ndarray, np.ndarray]]]:
    """Group adapters by layer name across all files."""
    layer_names = get_unique_layer_names(adapters_dict)
    grouped = {name: [] for name in layer_names}
    
    for file_path, adapters in adapters_dict.items():
        for name, (A, B) in adapters.items():
            grouped[name].append((A, B))
            logger.info(f"Added adapter pair for {name} from {file_path}")
    
    return grouped

def joint_diagonalization_full(As: List[np.ndarray], Bs: List[np.ndarray], r: int, max_iter: int = 100) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Perform joint diagonalization on multiple adapter pairs."""
    if not As or not Bs:
        raise ValueError("Empty input lists")
    
    logger.info(f"Starting joint diagonalization with {len(As)} adapter pairs")
    logger.info(f"Shapes - A: {As[0].shape}, B: {Bs[0].shape}, rank: {r}")
    
    m, k = Bs[0].shape
    n = As[0].shape[1]
    r = min(r, k, m, n)

    # Initialize random orthogonal matrices
    U = np.random.randn(m, r)
    V = np.random.randn(n, r)
    U, _ = np.linalg.qr(U)
    V, _ = np.linalg.qr(V)

    # Iterative optimization
    for iter_num in range(max_iter):
        # Update U
        M = sum(B @ A @ V @ V.T @ A.T @ B.T for A, B in zip(As, Bs))
        U_new, _ = np.linalg.qr(M @ U)
        
        # Update V
        N = sum(A.T @ B.T @ U @ U.T @ B @ A for A, B in zip(As, Bs))
        V_new, _ = np.linalg.qr(N @ V)
        
        # Check convergence
        if iter_num % 10 == 0:
            logger.debug(f"Iteration {iter_num}")
        
        U = U_new
        V = V_new

    # Compute compressed representations
    Sigmas = [U.T @ B @ A @ V for A, B in zip(As, Bs)]
    logger.info("Joint diagonalization completed successfully")
    
    return U, V, Sigmas

def compress_merged_adapters(adapters_dict: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]], r: int) -> Dict[str, Tuple[np.ndarray, np.ndarray, List[np.ndarray]]]:
    """Compress multiple adapters into shared basis vectors with individual Sigma matrices."""
    compressed = {}
    
    # Group adapters by layer
    grouped_adapters = group_adapters_by_layer(adapters_dict)
    logger.info(f"Grouped adapters into {len(grouped_adapters)} layers")
    
    # Compress each layer group
    for layer_name, adapter_pairs in grouped_adapters.items():
        if not adapter_pairs:
            logger.warning(f"No adapter pairs for layer {layer_name}")
            continue
            
        try:
            As, Bs = zip(*adapter_pairs)
            
            if Bs[0].shape[1] != As[0].shape[0]:
                logger.warning(f"Incompatible shapes for layer {layer_name}: {As[0].shape} and {Bs[0].shape}")
                continue
                
            logger.info(f"Compressing layer {layer_name} with {len(adapter_pairs)} adapter pairs")
            U, V, Sigmas = joint_diagonalization_full(list(As), list(Bs), r)
            compressed[layer_name] = (U, V.T, Sigmas)
            logger.info(f"Successfully compressed layer {layer_name}")
            
        except Exception as e:
            logger.error(f"Error compressing layer {layer_name}: {str(e)}")
            continue

    return compressed

def save_compressed_adapters(compressed: Dict[str, Tuple[np.ndarray, np.ndarray, List[np.ndarray]]], file_path: str):
    """Save compressed adapters with layer structure."""
    tensors = {}
    logger.info("Preparing tensors for saving")
    
    for name, (U, V, Sigmas) in compressed.items():
        # Extract layer index and projection type from name
        match = re.match(r'layer_(\d+)_(\w+)_proj', name)
        if match:
            layer_idx = match.group(1)
            proj_type = match.group(2)
            
            # Create keys that match the original structure
            base_key = f"base_model.model.model.layers.{layer_idx}.self_attn.{proj_type}_proj"
            
            # Save U, V as compressed weights
            tensors[f"{base_key}.compressed_U.weight"] = torch.from_numpy(U).contiguous().float()
            tensors[f"{base_key}.compressed_V.weight"] = torch.from_numpy(V).contiguous().float()
            
            # Save individual Sigma matrices
            for i, S in enumerate(Sigmas):
                tensors[f"{base_key}.compressed_S_{i}.weight"] = torch.from_numpy(S).contiguous().float()
                
            logger.info(f"Prepared tensors for {name}")
    
    try:
        save_file(tensors, file_path)
        logger.info(f"Successfully saved compressed adapters to {file_path}")
    except Exception as e:
        logger.error(f"Error saving compressed adapters: {str(e)}")
        raise

def load_compressed_adapters(file_path: str) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]]:
    """Load compressed adapters with layer structure."""
    compressed = {}
    
    with safe_open(file_path, framework="pt") as f:
        keys = list(f.keys())
        layer_groups = {}
        
        # Group keys by layer and projection type
        for key in keys:
            match = re.search(r'layers\.(\d+)\.self_attn\.(\w+)_proj\.compressed_(\w+)\.weight', key)
            if match:
                layer_idx = match.group(1)
                proj_type = match.group(2)
                comp_type = match.group(3)
                
                group_key = f"layer_{layer_idx}_{proj_type}_proj"
                if group_key not in layer_groups:
                    layer_groups[group_key] = {'U': None, 'V': None, 'S': []}
                
                tensor = f.get_tensor(key)
                if comp_type == 'U':
                    layer_groups[group_key]['U'] = tensor
                elif comp_type == 'V':
                    layer_groups[group_key]['V'] = tensor
                elif comp_type.startswith('S_'):
                    layer_groups[group_key]['S'].append(tensor)
        
        # Create compressed adapter entries
        for name, group in layer_groups.items():
            if group['U'] is not None and group['V'] is not None and group['S']:
                compressed[name] = (group['U'], group['V'], sorted(group['S'], key=lambda x: int(x.shape[0])))
    
    return compressed

def main():
    # List of input adapter files to compress together
    input_files = ["adapter_model.safetensors", "adapter_model2.safetensors"]
    
    try:
        logger.info("Starting adapter compression process")
        logger.info(f"Input files: {input_files}")
        
        # Load all adapter files
        all_adapters = load_safetensors(input_files)
        
        if not all_adapters:
            logger.error("No adapter files loaded successfully")
            return
            
        # Print loaded adapter information
        for file_path, adapters in all_adapters.items():
            logger.info(f"\nLoaded from {file_path}:")
            for name, (A, B) in adapters.items():
                logger.info(f"  {name}: A shape {A.shape}, B shape {B.shape}")

        # Compress adapters together
        r_compress = 32  # Compression rank
        compressed = compress_merged_adapters(all_adapters, r_compress)

        if compressed:
            output_file = "compressed_merged_adapters.safetensors"
            save_compressed_adapters(compressed, output_file)
            logger.info(f"Successfully compressed and saved merged adapters to {output_file}")
        else:
            logger.warning("No adapters were successfully compressed")
            
    except Exception as e:
        logger.error(f"Error during compression process: {str(e)}")
        raise

if __name__ == "__main__":
    main()