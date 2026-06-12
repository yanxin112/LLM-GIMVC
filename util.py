import random

import numpy as np
import torch


def normalize(x):
    x_min = np.min(x)
    x_max = np.max(x)
    denom = x_max - x_min
    if denom == 0:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - x_min) / denom).astype(np.float32)


def normalize_row(x):
    x = x.astype(np.float32)
    x_min = np.min(x, axis=0, keepdims=True)
    x_max = np.max(x, axis=0, keepdims=True)
    denom = x_max - x_min
    denom[denom == 0] = 1
    return (x - x_min) / denom


def set_seed(seed_num):
    np.random.seed(seed_num)
    random.seed(seed_num + 1)
    torch.manual_seed(seed_num + 2)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_num + 3)
        torch.backends.cudnn.deterministic = True
