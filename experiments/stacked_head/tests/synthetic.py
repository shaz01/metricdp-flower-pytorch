"""Synthetic stand-in for a 4-class image task, so the pipeline can be tested without downloads."""

import numpy as np


def make_images(labels, seed):
    rng = np.random.default_rng(seed)
    images = rng.normal(size=(len(labels), 1, 28, 28)).astype(np.float32) * 0.15
    for k in range(4):
        row, column = (k // 2) * 14, (k % 2) * 14
        images[labels == k, 0, row : row + 14, column : column + 14] += 0.6
    return images


def client_labels(client_id, seed):
    counts = np.full(4, 17)
    counts[client_id % 4] = 205
    labels = np.concatenate([np.full(c, k) for k, c in enumerate(counts)])
    return np.random.default_rng(seed + client_id).permutation(labels)


def labels_for(count_per_class):
    return np.repeat(np.arange(4), count_per_class)
