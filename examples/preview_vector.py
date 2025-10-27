import cv2
import numpy as np

from ome.io.reader.vector import VECTORReader
from ome.io.sample.uniform import UniformSampler
from ome.repr.frame import events_to_grayscale_count_cuda
from ome.repr.voxel import events_to_voxel_cuda
from ome.utils.timer import Timer
from ome.utils.transforms import contain_into
from tools.window import Window

reader = VECTORReader(
    "/var/mnt/data/datasets/vector/corridors-dolly/corridors_dolly1.synced.left_event.hdf5",
    image_folder="/var/mnt/data/datasets/vector/corridors-dolly/corridors_dolly1.synced.left_camera",
    timestamps_txt="/var/mnt/data/datasets/vector/corridors-dolly/corridors_dolly1.synced.left_camera/timestamp.txt",
)
sampler = UniformSampler(reader, sample_rate=1, duration_ms=33)

sample_idx_to_grayscale_idx = sampler.sync_with(reader.grayscale_wallclock[:], anchor="middle")
ms_per_frame = 33
window = Window("VECTor Preview", ms_per_frame)
timer = Timer()
frame_timer = Timer()
voxel_timer = Timer()

try:
    for idx, timestamp in enumerate(sampler.get_all_timestamps(anchor="start")):
        timer.tick()
        x, y, p, t = sampler[idx]

        # Create event frame
        frame_timer.tick()
        event_frame = events_to_grayscale_count_cuda(x, y, width=reader.width, height=reader.height)
        event_frame = reader.rectify(event_frame, camera="left_event")
        frame_time = frame_timer.elapsed()

        # Create event voxel
        voxel_timer.tick()
        event_voxel = events_to_voxel_cuda(x, y, p, t, width=reader.width, height=reader.height, n_bins=3)
        event_voxel = (event_voxel - event_voxel.min()) / (event_voxel.max() - event_voxel.min() + 1e-8)
        event_voxel = (event_voxel * 255).transpose(1, 2, 0).astype(np.uint8)
        event_voxel = reader.rectify(event_voxel, camera="left_event")
        voxel_time = voxel_timer.elapsed()

        # Get corresponding grayscale image
        grayscale_idx = sample_idx_to_grayscale_idx[idx]
        grayscale_file = reader.grayscale_files[grayscale_idx]
        grayscale_image = cv2.imread(str(grayscale_file), cv2.IMREAD_GRAYSCALE)
        grayscale_image = reader.rectify(grayscale_image, camera="left_grayscale")
        grayscale_image = contain_into(grayscale_image, (reader.height, reader.width))

        # Concatenate event frame and grayscale image horizontally
        combined_frame = np.hstack(
            [
                cv2.cvtColor(event_frame, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(event_voxel, cv2.COLOR_RGB2BGR),
            ]
        )

        elapsed = timer.elapsed()

        print(
            f"[{idx + 1}/{len(sampler)}] | Number of events: {len(x)}"
            f" | Processing time: {elapsed * 1e3:.2f} ms"
            f" | Event range: {timestamp} - {timestamp + ms_per_frame} ms"
            f" | Grayscale idx: {grayscale_idx}"
            f" | Frame time: {frame_time * 1e3:.2f} ms"
            f" | Voxel time: {voxel_time * 1e3:.2f} ms"
        )

        if window.show(combined_frame, elapsed * 1e3):
            break
except KeyboardInterrupt:
    pass
finally:
    cv2.destroyAllWindows()
