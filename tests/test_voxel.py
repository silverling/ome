import numpy as np

from ome.repr.voxel import events_to_voxel, events_to_voxel_cuda
from ome.utils.timer import Timer

N = 1_000_000
x = np.random.randint(0, 1280, size=N, dtype=np.uint32)
y = np.random.randint(0, 720, size=N, dtype=np.uint32)
p = np.random.randint(0, 2, size=N, dtype=np.int8)
t = (np.random.uniform(0, 0.015, size=N) * 1e6).astype(np.int64)  # 15ms, in microseconds
t.sort()

ROUND = 10

timer = Timer()
timer.tick()
for i in range(ROUND):
    voxel_grid = events_to_voxel(x, y, p, t, 1280, 720, n_bins=5)
print(f"events_to_voxel          average time: {timer.elapsed() / ROUND} seconds")

assert voxel_grid.shape == (5, 720, 1280)

timer.tick()
for i in range(ROUND):
    voxel_grid = events_to_voxel_cuda(x, y, p, t, 1280, 720, n_bins=5)
print(f"events_to_voxel_cuda     average time: {timer.elapsed() / ROUND} seconds")
assert voxel_grid.shape == (5, 720, 1280)
