#!/usr/bin/env python3
"""
Context vs Target Subgraph Size Analysis

This script analyzes subgraph size distributions for all three partitioning methods
(Motif-based, METIS, Random Walk) to address reviewer concerns about the role of
subgraph size in JEPA performance.

The analysis:
1. Loads the full Aldeghi polymer dataset (42,966 graphs)
2. For each method, applies JEPA partitioning to extract context and target subgraphs
3. Measures actual subgraph sizes vs theoretical parameters (60% context, 10% target)
4. Generates visualizations and statistics for reviewer response
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.data import create_data
from src.config import cfg
from src.transform import GraphJEPAPartitionTransform

def analyze_context_target_separately():
    """
    Main analysis function that processes all three subgraphing methods.
    
    Steps:
    1. Setup: Load dataset and configuration
    2. Method Loop: For each partitioning method (Motif, METIS, Random Walk):
       - Create JEPA transform with method-specific parameters
       - Process all graphs in dataset to extract subgraph sizes
       - Calculate size statistics and adherence to theoretical parameters
    3. Visualization: Generate comparative plots
    4. Results: Print comprehensive statistics
    """
    
    # STEP 1: SETUP - Load dataset and configuration
    print("Setting up analysis...")
    cfg_local = cfg.clone()  # Clone config to avoid modifying global settings
    cfg_local.finetuneDataset = 'aldeghi'  # Use Aldeghi polymer dataset
    train_dataset, _, _ = create_data(cfg_local)  # Load 42,966 polymer graphs
    print(f"Loaded {len(train_dataset)} polymer graphs for analysis")
    
    # Define the three subgraphing methods to analyze
    # Each tuple: (method_id, human_readable_name)
    methods = [(0, 'Motif-based'), (1, 'METIS'), (2, 'Random Walk')]
    results = {}  # Store results for each method
    
    # STEP 2: METHOD LOOP - Analyze each partitioning method
    for subgraph_type, method_name in methods:
        print(f"\nAnalyzing {method_name} subgraphing method...")
        
        # Create JEPA transform for this specific method
        # Uses configuration values instead of hardcoded parameters
        transform = GraphJEPAPartitionTransform(
            subgraphing_type=subgraph_type,  # 0=Motif, 1=METIS, 2=Random Walk
            num_targets=cfg.jepa.num_targets,  # Number of target subgraphs (=1)
            n_patches=cfg.subgraphing.n_patches,  # Max patches (=32)
            context_size=cfg.subgraphing.context_size,  # Context size param (=0.6)
            target_size=cfg.subgraphing.target_size,  # Target size param (=0.1)
            dataset=cfg_local.finetuneDataset  # Dataset name for method-specific logic
        )
        
        # Initialize storage for this method's results
        context_sizes = []  # Actual context subgraph sizes (in nodes)
        target_sizes = []   # Actual target subgraph sizes (in nodes)
        full_graph_sizes = []  # Original graph sizes for percentage calculations
        context_percentages = []  # Context size as % of original graph
        target_percentages = []   # Target size as % of original graph
        
        # Process ALL graphs in the dataset (no sampling)
        sample_indices = torch.randperm(len(train_dataset))  # Randomize order
        
        # GRAPH PROCESSING LOOP - Extract subgraphs from each polymer graph
        for idx in sample_indices:
            try:
                # Get original polymer graph
                data = train_dataset[idx]
                full_size = data.num_nodes  # Original graph size
                
                # Apply JEPA partitioning transform
                # This creates context and target subgraphs using the specified method
                transformed_data = transform(data)
                
                # Extract partitioning results
                subgraphs_batch = transformed_data.subgraphs_batch  # Subgraph IDs for each node
                context_idx = transformed_data.context_subgraph_idx  # Which subgraph is context
                target_idxs = transformed_data.target_subgraph_idxs  # Which subgraphs are targets
                
                # SUBGRAPH SIZE CALCULATION - Measure each subgraph
                for subgraph_idx in subgraphs_batch.unique():
                    # Create mask for nodes belonging to this subgraph
                    mask = subgraphs_batch == subgraph_idx
                    size = mask.sum().item()  # Count nodes in this subgraph
                    percentage = size / full_size  # Calculate as % of original graph
                    
                    # Categorize subgraph and store measurements
                    if subgraph_idx == context_idx:
                        # This is the context subgraph (larger, for representation)
                        context_sizes.append(size)
                        context_percentages.append(percentage)
                        full_graph_sizes.append(full_size)
                    elif subgraph_idx in target_idxs:
                        # This is a target subgraph (smaller, for prediction)
                        target_sizes.append(size)
                        target_percentages.append(percentage)
            except Exception as e:
                # Skip graphs that cause errors (malformed data, etc.)
                continue
        
        # Store results for this method
        results[method_name] = {
            'context': context_sizes,  # List of context subgraph sizes
            'target': target_sizes,   # List of target subgraph sizes
            'full_sizes': full_graph_sizes,  # Original graph sizes
            'context_pct': context_percentages,  # Context sizes as percentages
            'target_pct': target_percentages     # Target sizes as percentages
        }
        print(f"  Processed {len(context_sizes)} context and {len(target_sizes)} target subgraphs")
    
    # STEP 3: VISUALIZATION - Create comparative plots
    print("\nGenerating visualization...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # 2 rows (context/target) x 3 cols (methods)
    fig.suptitle('Context vs Target Subgraph Size Distributions', fontsize=16)
    
    methods = ['Motif-based', 'METIS', 'Random Walk']
    colors = ['blue', 'orange', 'green']
    
    for i, (method, color) in enumerate(zip(methods, colors)):
        # Context subgraphs (top row)
        context_sizes = results[method]['context']
        axes[0, i].hist(context_sizes, bins=10, alpha=0.7, color=color, edgecolor='black')
        axes[0, i].set_title(f'{method}\\nContext Subgraphs')
        axes[0, i].set_xlabel('Size (nodes)')
        axes[0, i].set_ylabel('Frequency')
        axes[0, i].grid(True, alpha=0.3)
        
        mean_ctx = np.mean(context_sizes)
        std_ctx = np.std(context_sizes)
        axes[0, i].text(0.65, 0.8, f'μ={mean_ctx:.1f}\\nσ={std_ctx:.1f}', 
                       transform=axes[0, i].transAxes, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Target subgraphs (bottom row)
        target_sizes = results[method]['target']
        axes[1, i].hist(target_sizes, bins=10, alpha=0.7, color=color, edgecolor='black')
        axes[1, i].set_title(f'{method}\\nTarget Subgraphs')
        axes[1, i].set_xlabel('Size (nodes)')
        axes[1, i].set_ylabel('Frequency')
        axes[1, i].grid(True, alpha=0.3)
        
        mean_tgt = np.mean(target_sizes)
        std_tgt = np.std(target_sizes)
        axes[1, i].text(0.65, 0.8, f'μ={mean_tgt:.1f}\\nσ={std_tgt:.1f}', 
                       transform=axes[1, i].transAxes,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('context_target_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # STEP 4: RESULTS - Print comprehensive statistics
    print("\\nCONTEXT vs TARGET SUBGRAPH ANALYSIS")
    print("=" * 50)
    
    full_sizes = results['Motif-based']['full_sizes']
    print(f"Full graph sizes: {np.mean(full_sizes):.1f} ± {np.std(full_sizes):.1f} nodes")
    print(f"Context size parameter: {cfg.subgraphing.context_size} ({cfg.subgraphing.context_size*100:.0f}%), Target size parameter: {cfg.subgraphing.target_size} ({cfg.subgraphing.target_size*100:.0f}%)")
    
    for method in methods:
        ctx_sizes = results[method]['context']
        tgt_sizes = results[method]['target']
        ctx_pct = results[method]['context_pct']
        tgt_pct = results[method]['target_pct']
        
        print(f"\\n{method.upper()}:")
        print(f"  Context: {np.mean(ctx_sizes):.1f} ± {np.std(ctx_sizes):.1f} nodes ({np.mean(ctx_pct)*100:.1f}% of graph)")
        print(f"  Target:  {np.mean(tgt_sizes):.1f} ± {np.std(tgt_sizes):.1f} nodes ({np.mean(tgt_pct)*100:.1f}% of graph)")
        print(f"  Ratio:   {np.mean(ctx_sizes)/np.mean(tgt_sizes):.1f}:1")
        print(f"  Adherence: Context {np.mean(ctx_pct)/cfg.subgraphing.context_size*100:.0f}%, Target {np.mean(tgt_pct)/cfg.subgraphing.target_size*100:.0f}% of expected")

if __name__ == "__main__":
    analyze_context_target_separately()