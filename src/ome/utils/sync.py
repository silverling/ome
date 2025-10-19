import numpy as np


def nearest_idx(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Given two 1D sorted numpy arrays a and b, find the index of the nearest element in a for each element in b."""
    n = a.size
    pos = np.searchsorted(a, b)  # O(m log n)
    pos = np.clip(pos, 1, n - 1)  # Make sure left is valid
    left = pos - 1

    # Distance comparison
    d_left = np.abs(a[left] - b)
    d_right = np.abs(a[pos] - b)

    # Choose left when distances are equal (can also change to choose right)
    return np.where(d_left <= d_right, left, pos)
