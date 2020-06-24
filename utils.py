import numpy as np
import psutil
import torch
from scipy.signal import convolve, gaussian


def is_enough_ram(min_available_gb=0.1):
    mem = psutil.virtual_memory()
    return mem.available >= min_available_gb * (1024 ** 3)


def smoothen(values):
    kernel = gaussian(100, std=100)
    kernel = kernel / np.sum(kernel)
    return convolve(values, kernel, 'valid')


def save_checkpoint(state, checkpoint_dir, is_best=False):
    if is_best:
        path = checkpoint_dir + '/best_checkpoint.pth'
    else:
        path = checkpoint_dir + '/checkpoint.pth'
    torch.save(state, path)


def load_checkpoint(checkpoint_dir, is_best=False):
    if is_best:
        path = checkpoint_dir + '/best_checkpoint.pth'
    else:
        path = checkpoint_dir + '/checkpoint.pth'
    return torch.load(path)
