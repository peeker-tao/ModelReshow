import torch
import numpy as np


# 统计mean和std
def compute_mean_std(loader):
    mean = 0.0
    std = 0.0
    total_samples = 0

    for batch in loader:
        images = batch[0]
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_samples += batch_samples

    mean /= total_samples
    std /= total_samples
    return mean, std
