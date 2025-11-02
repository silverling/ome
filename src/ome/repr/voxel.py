import cupy as cp
import cupyx as cpx
import numpy as np

import ome.utils._nvidia_cdll_hook  # noqa: F401


# Code adapted from https://github.com/uzh-rpg/rpg_e2vid/blob/master/utils/inference_utils.py#L431
def events_to_voxel(x, y, p, t, width, height, n_bins=3):
    """Build a voxel grid with bilinear interpolation in the time domain from a set of events.

    Implements the event volume from Zhu et al. 2019, "Unsupervised event-based learning of optical
    flow, depth, and egomotion."

    Return shape is (n_bins, height, width)
    """
    assert len(x) == len(y) == len(p) == len(t), "All input arrays must have the same length"

    if len(x) == 0:
        return np.zeros((n_bins, height, width), np.float32)

    voxel_grid = np.zeros((n_bins, height, width), np.float32).ravel()

    x = x.astype(np.uint32)
    y = y.astype(np.uint32)
    p = p.astype(np.int8)
    p[p == 0] = -1  # convert polarity from 0/1 to -1/1

    # normalize the event timestamps so that they lie in [0, n_bins - 1]
    t = (n_bins - 1) * (t - t[0]) / (t[-1] - t[0])

    ti = t.astype(np.int64)  # ts in integer
    dt = t - ti  # delta t, i.e. the fractional part of the normalized timestamp
    contrib_left = p * (1.0 - dt)  # contribution to the left bin
    contrib_right = p * dt  # contribution to the right bin

    valid_indices = ti < n_bins
    np.add.at(
        voxel_grid,
        x[valid_indices] + y[valid_indices] * width + ti[valid_indices] * width * height,
        contrib_left[valid_indices],
    )

    valid_indices = (ti + 1) < n_bins
    np.add.at(
        voxel_grid,
        x[valid_indices] + y[valid_indices] * width + (ti[valid_indices] + 1) * width * height,
        contrib_right[valid_indices],
    )

    voxel_grid = np.reshape(voxel_grid, (n_bins, height, width))

    return voxel_grid


def events_to_voxel_cuda(x, y, p, t, width, height, n_bins=3):
    """Voxel grid with bilinear interpolation in time domain (CuPy GPU version)."""

    assert len(x) == len(y) == len(p) == len(t), "All input arrays must have the same length"

    if len(x) == 0:
        return np.zeros((n_bins, height, width), np.float32)

    # move everything to GPU
    x = cp.asarray(x, dtype=cp.uint32)
    y = cp.asarray(y, dtype=cp.uint32)
    p = cp.asarray(p, dtype=cp.int8)
    t = cp.asarray(t, dtype=cp.float32)

    p = cp.where(p == 0, -1, p)  # polarity 0/1 -> -1/1

    # normalize timestamps into [0, n_bins-1]
    t = (n_bins - 1) * (t - t[0]) / (t[-1] - t[0])

    ti = t.astype(cp.int64)
    dt = t - ti
    contrib_left = p * (1.0 - dt)
    contrib_right = p * dt

    voxel_grid = cp.zeros((n_bins * height * width,), dtype=cp.float32)

    # left bin contribution
    valid_left = ti < n_bins
    idx_left = x[valid_left] + y[valid_left] * width + ti[valid_left] * width * height
    cpx.scatter_add(voxel_grid, idx_left, contrib_left[valid_left])

    # right bin contribution
    valid_right = (ti + 1) < n_bins
    idx_right = x[valid_right] + y[valid_right] * width + (ti[valid_right] + 1) * width * height
    cpx.scatter_add(voxel_grid, idx_right, contrib_right[valid_right])

    voxel_grid = voxel_grid.reshape((n_bins, height, width))
    return cp.asnumpy(voxel_grid)
