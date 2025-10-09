# Environment Setup for Polymer-JEPA

This document provides multiple ways to recreate the exact environment used for the Polymer-JEPA project.

## Option 1: Using Conda Environment File (Recommended)

The most reliable way to recreate the exact environment:

```bash
# Create environment from the exported YAML file
conda env create -f environment_polymera-jepa-env.yml

# Activate the environment
conda activate polymera-jepa-env
```

## Option 2: Using pip requirements

If you prefer using pip or need to install in an existing environment:

```bash
# Create a new conda environment with Python 3.10
conda create -n polymer-jepa python=3.10
conda activate polymer-jepa

# Install PyTorch first (important for compatibility)
conda install pytorch=1.13.1 -c pytorch

# Install remaining packages
pip install -r requirements_g2senv.txt
```

## Option 3: Manual Installation (Original README approach)

```bash
conda create -n polymer-jepa python=3.10
conda activate polymer-jepa
pip install torch-geometric==2.3.0 torch-sparse torch-scatter torch-cluster -f https://pytorch-geometric.com/whl/torch-2.1.0+cu121.html
pip install torch==2.1 wandb networkx yacs metis kaleido tensorboard einops pillow tqdm pandas rdkit matplotlib plotly seaborn umap
```

## Key Dependencies

The environment includes these critical packages:
- **PyTorch**: 1.13.1 (deep learning framework)
- **PyTorch Geometric**: 2.3.0 (graph neural networks)
- **RDKit**: 2023.9.4 (molecular informatics)
- **NetworkX**: 3.2.1 (graph analysis)
- **METIS**: 0.2a5 (graph partitioning)
- **Weights & Biases**: 0.16.3 (experiment tracking)
- **YACS**: 0.1.8 (configuration management)

## Verification

After installation, verify the environment works:

```bash
python -c "import torch; import torch_geometric; import rdkit; print('Environment setup successful!')"
```

## Files Included

- `environment_g2senv.yml`: Complete conda environment export
- `requirements_g2senv.txt`: pip freeze output with exact versions
- This setup guide

## Notes

- The environment was exported from macOS with Apple Silicon
- Some packages may have different versions on other platforms
- If you encounter issues, try Option 1 (conda environment file) first
- For GPU support, you may need to adjust PyTorch installation based on your CUDA version