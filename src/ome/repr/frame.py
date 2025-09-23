import cupy as cp
import numba
import numpy as np

import ome.utils._nvidia_cdll_hook  # noqa: F401

# ===
# Approach 1:   Accumulate a chunk of event counts per pixel, then tonemap using log1p. This is the naivest way.
#               Here, we provide both a CUDA and a Numba implementation.
#               The CUDA implementation is faster, but requires a compatible NVIDIA GPU and CuPy.
#               The Numba implementation is slower, but works on CPU.
# ===

_accumulate_kernel = r"""
extern "C" __global__
void accumulate_events(const unsigned int *coords, unsigned int *frame, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;
    atomicAdd(&frame[coords[idx]], 1);
}
"""

_accumulate_events_cuda = cp.RawKernel(_accumulate_kernel, "accumulate_events")


def events_to_grayscale_count_cuda(x, y, /, *, width, height, normalize=True):
    x_gpu = cp.asarray(x, dtype=cp.uint16)
    y_gpu = cp.asarray(y, dtype=cp.uint16)

    frame_gpu = cp.zeros((height, width), dtype=cp.uint32)
    coords = y_gpu.astype(cp.uint32) * width + x_gpu

    threads = 256
    blocks = (coords.size + threads - 1) // threads

    _accumulate_events_cuda((blocks,), (threads,), (coords, frame_gpu.ravel(), coords.size))

    if normalize and frame_gpu.max() > 0:
        frame_gpu = (cp.log1p(frame_gpu) * 255.0 / cp.log1p(frame_gpu.max())).astype(cp.uint8)

    return cp.asnumpy(frame_gpu)


@numba.njit(parallel=True)
def _accumulate_events_numba(x, y, width, height):
    frame = np.zeros((height, width), dtype=np.uint32)
    for i in numba.prange(x.size):  # ty: ignore[not-iterable]
        if 0 <= x[i] < width and 0 <= y[i] < height:
            frame[y[i], x[i]] += 1
    return frame


def events_to_grayscale_count_numba(x, y, /, *, width, height):
    frame = _accumulate_events_numba(x, y, width, height)
    if frame.max() > 0:
        frame = (np.log1p(frame) * 255.0 / np.log1p(frame.max())).astype(np.uint8)
    return frame
