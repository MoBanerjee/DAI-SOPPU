import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import torch
from safetensors import safe_open
import pandas as pd
import re
import os

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
            with safe_open(file_path, framework="numpy") as f:
                keys = list(f.keys())
                
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
            
            if adapters:
                all_adapters[file_path] = adapters
                
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            continue
    
    return all_adapters

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

class CompressionEvaluator:
    def __init__(self, original_files: List[str], compressed_file: str):
        """Initialize evaluator with paths to original and compressed adapter files."""
        self.original_files = original_files
        self.compressed_file = compressed_file
        self.original_adapters = None
        self.compressed_adapters = None
        
    def load_data(self):
        """Load original and compressed adapter data."""
        logger.info("Loading original adapters...")
        self.original_adapters = load_safetensors(self.original_files)
        
        logger.info("Loading compressed adapters...")
        self.compressed_adapters = load_compressed_adapters(self.compressed_file)
        
    def calculate_compression_stats(self) -> Dict:
        """Calculate compression statistics."""
        stats = {
            'original_params': 0,
            'compressed_params': 0,
            'compression_ratios': [],
            'memory_savings': [],
            'layer_stats': []
        }
        
        # Calculate per-layer statistics
        for layer_name, (U, V, Sigmas) in self.compressed_adapters.items():
            original_params = 0
            for file_adapters in self.original_adapters.values():
                if layer_name in file_adapters:
                    A, B = file_adapters[layer_name]
                    original_params += A.size + B.size
            
            compressed_params = U.numel() + V.numel() + sum(S.numel() for S in Sigmas)
            compression_ratio = original_params / compressed_params
            memory_saving = 1 - (compressed_params / original_params)
            
            stats['original_params'] += original_params
            stats['compressed_params'] += compressed_params
            stats['compression_ratios'].append(compression_ratio)
            stats['memory_savings'].append(memory_saving)
            
            stats['layer_stats'].append({
                'layer': layer_name,
                'original_params': original_params,
                'compressed_params': compressed_params,
                'compression_ratio': compression_ratio,
                'memory_saving': memory_saving
            })
        
        return stats
    
    def calculate_reconstruction_error(self) -> Dict:
        """Calculate reconstruction error for compressed adapters."""
        errors = {}
        
        for layer_name, (U, V, Sigmas) in self.compressed_adapters.items():
            layer_errors = []
            
            # Calculate reconstruction for each original adapter
            for file_adapters in self.original_adapters.values():
                if layer_name in file_adapters:
                    A, B = file_adapters[layer_name]
                    
                    # Convert to torch tensors if necessary
                    if isinstance(A, np.ndarray):
                        A = torch.from_numpy(A)
                        B = torch.from_numpy(B)
                    if isinstance(U, np.ndarray):
                        U = torch.from_numpy(U)
                        V = torch.from_numpy(V)
                    
                    # Calculate reconstruction error
                    for S in Sigmas:
                        if isinstance(S, np.ndarray):
                            S = torch.from_numpy(S)
                        reconstructed = torch.mm(torch.mm(U, S), V)
                        error = torch.norm(torch.mm(B, A) - reconstructed).item()
                        relative_error = error / torch.norm(torch.mm(B, A)).item()
                        layer_errors.append(relative_error)
            
            errors[layer_name] = layer_errors
        
        return errors

    def plot_compression_stats(self, stats: Dict, output_dir: str = 'compression_analysis'):
        """Generate visualization plots for compression statistics."""
        Path(output_dir).mkdir(exist_ok=True)
        
        # 1. Compression Ratios by Layer
        plt.figure(figsize=(12, 6))
        layer_names = [s['layer'] for s in stats['layer_stats']]
        ratios = [s['compression_ratio'] for s in stats['layer_stats']]
        
        plt.bar(range(len(layer_names)), ratios)
        plt.xticks(range(len(layer_names)), layer_names, rotation=45)
        plt.title('Compression Ratio by Layer')
        plt.ylabel('Compression Ratio')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/compression_ratios.png')
        plt.close()
        
        # 2. Memory Savings Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(stats['memory_savings'], bins=20)
        plt.title('Distribution of Memory Savings')
        plt.xlabel('Memory Saving (%)')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/memory_savings_dist.png')
        plt.close()
        
        # 3. Parameters Comparison
        plt.figure(figsize=(8, 6))
        labels = ['Original', 'Compressed']
        sizes = [stats['original_params'], stats['compressed_params']]
        plt.pie(sizes, labels=labels, autopct='%1.1f%%')
        plt.title('Parameter Distribution')
        plt.savefig(f'{output_dir}/params_comparison.png')
        plt.close()

    def plot_reconstruction_errors(self, errors: Dict, output_dir: str = 'compression_analysis'):
        """Plot reconstruction errors."""
        # Prepare data for box plot
        data = []
        labels = []
        
        for layer_name, layer_errors in errors.items():
            data.extend(layer_errors)
            labels.extend([layer_name] * len(layer_errors))
        
        df = pd.DataFrame({'Layer': labels, 'Relative Error': data})
        
        # Create box plot
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Layer', y='Relative Error', data=df)
        plt.xticks(rotation=45)
        plt.title('Reconstruction Error by Layer')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/reconstruction_errors.png')
        plt.close()

    def generate_report(self):
        """Generate comprehensive evaluation report."""
        self.load_data()
        
        # Calculate statistics
        stats = self.calculate_compression_stats()
        errors = self.calculate_reconstruction_error()
        
        # Generate plots
        self.plot_compression_stats(stats)
        self.plot_reconstruction_errors(errors)
        
        # Print summary statistics
        logger.info("\nCompression Summary:")
        logger.info(f"Total original parameters: {stats['original_params']:,}")
        logger.info(f"Total compressed parameters: {stats['compressed_params']:,}")
        logger.info(f"Overall compression ratio: {stats['original_params']/stats['compressed_params']:.2f}x")
        logger.info(f"Average memory saving: {np.mean(stats['memory_savings'])*100:.1f}%")
        
        logger.info("\nReconstruction Error Summary:")
        for layer_name, layer_errors in errors.items():
            logger.info(f"{layer_name}:")
            logger.info(f"  Mean error: {np.mean(layer_errors):.6f}")
            logger.info(f"  Std error: {np.std(layer_errors):.6f}")

def main():
    # Example usage
    original_files = ["adapter_model.safetensors", "adapter_model2.safetensors"]
    compressed_file = "compressed_merged_adapters.safetensors"
    
    evaluator = CompressionEvaluator(original_files, compressed_file)
    evaluator.generate_report()

if __name__ == "__main__":
    main()