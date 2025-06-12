import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)           # Sets seed for Python's built-in random module
    np.random.seed(seed)        # Sets seed for NumPy's random number generator
    torch.manual_seed(seed)     # Sets seed for PyTorch CPU operations
    torch.cuda.manual_seed_all(seed)  # Sets seed for all CUDA devices
    print(f"Seed set to {seed}. This ensures reproducibility of results across runs.")