#!/bin/sh
# Ablation with random walk to determine context, target size impact for single target
# Each setting with 3 repetitions and 5Fold-CV or MonomerCV
subgraph_type=2 #2 is random walk
nr_targets=1
aldeghiFTPercentage=0.01
layer_norm=0
early_stopping=0
# context size ablation with target size 0.15
target_size=0.15
for context_size in 0.2 0.4 0.6 0.8 0.95
do
    for seeds in 0 1 2
    do
        echo "Context size ablation"
        #sbatch ./run.sbatch --jepa.num_targets ${nr_targets} --subgraphing.type ${subgraph_type} --subgraphing.context_size ${context_size}  --subgraphing.target_size ${target_size} --seeds ${seeds} --pretrain.layer_norm ${layer_norm} --pretrain.early_stopping ${early_stopping} --finetune.aldeghiFTPercentage ${aldeghiFTPercentage}
    done
    # Run this with addtional modifications in config:, i.e., PL true&false, Pretraining True and False, etc.
done

# target size ablation with context size 0.6
context_size=0.8
for target_size in 0.05 0.1 0.2 0.25
do
    for seeds in 0 1 2
    do
        echo "Target size ablation"
        #sbatch ./run.sbatch --jepa.num_targets ${nr_targets} --subgraphing.type ${subgraph_type} --subgraphing.context_size ${context_size}  --subgraphing.target_size ${target_size} --seeds ${seeds} --pretrain.layer_norm ${layer_norm} --pretrain.early_stopping ${early_stopping} --finetune.aldeghiFTPercentage ${aldeghiFTPercentage}
    done
    # Run this with addtional modifications in config:, i.e., PL true&false, Pretraining True and False, etc.
done

# Single runs: 
target_size=0.1
context_size=0.8
seeds=0
sbatch ./run.sbatch --jepa.num_targets ${nr_targets} --subgraphing.type ${subgraph_type} --subgraphing.context_size ${context_size}  --subgraphing.target_size ${target_size} --seeds ${seeds} --pretrain.layer_norm ${layer_norm} --pretrain.early_stopping ${early_stopping} --finetune.aldeghiFTPercentage ${aldeghiFTPercentage}