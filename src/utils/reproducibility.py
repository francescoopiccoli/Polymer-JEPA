import random
import numpy as np
import torch

def set_seed(seed: int):
    """
    Sets random seed for Python, NumPy, and PyTorch for full reproducibility.
    Call this before any model/data creation.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):
    """
    Ensures each DataLoader worker is seeded reproducibly.
    Pass this as worker_init_fn to DataLoader.
    """
    np.random.seed(torch.initial_seed() % 2**32)