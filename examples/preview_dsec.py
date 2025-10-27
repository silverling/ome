import cv2
import numpy as np

from ome.io.reader.dsec import DSECReader
from ome.io.sample.uniform import UniformSampler
from ome.repr.frame import events_to_grayscale_count_cuda
from ome.repr.voxel import events_to_voxel_cuda
from ome.utils.timer import Timer
from tools.window import Window

reader = DSECReader(
    "/var/mnt/data/datasets/DSEC/unpack/train/interlaken_00_c/events/left/events.h5",
    event_rectify_map_file="/var/mnt/data/datasets/DSEC/unpack/train/interlaken_00_c/events/left/rectify_map.h5",
    image_folder="/var/mnt/data/datasets/DSEC/unpack/train/interlaken_00_c/images/left/rectified/",
    timestamps_txt="/var/mnt/data/datasets/DSEC/unpack/train/interlaken_00_c/images/timestamps.txt",
)
sampler = UniformSampler(reader, sample_rate=1, duration_ms=33)

sample_idx_to_rgb_idx = sampler.sync_with(reader.rgb_wallclock[:], anchor="middle")
ms_per_frame = 33
window = Window("DSEC Preview", ms_per_frame)
timer = Timer()
frame_timer = Timer()
voxel_timer = Timer()

map_x = reader.rectify_map[:, :, 0]
map_y = reader.rectify_map[:, :, 1]

try:
    for idx, timestamp in enumerate(sampler.get_all_timestamps(anchor="start")):
        timer.tick()
        x, y, p, t = sampler[idx]

        # Create event frame
        frame_timer.tick()
        event_frame = events_to_grayscale_count_cuda(x, y, width=reader.width, height=reader.height)
        frame_time = frame_timer.elapsed()

        # Create event voxel
        voxel_timer.tick()
        event_voxel = events_to_voxel_cuda(x, y, p, t, width=reader.width, height=reader.height, n_bins=3)
        event_voxel = (event_voxel - event_voxel.min()) / (event_voxel.max() - event_voxel.min() + 1e-8)
        event_voxel = (event_voxel * 255).transpose(1, 2, 0).astype(np.uint8)
        event_voxel = cv2.remap(event_voxel, map_x, map_y, interpolation=cv2.INTER_LINEAR)
        voxel_time = voxel_timer.elapsed()

        # Get corresponding rgb image
        rgb_idx = sample_idx_to_rgb_idx[idx]
        rgb_file = reader.rgb_files[rgb_idx]
        rgb_image = cv2.imread(str(rgb_file), cv2.IMREAD_COLOR)
        rgb_image = cv2.resize(rgb_image, (reader.width, reader.height))

        # Concatenate event frame and rgb image horizontally
        combined_frame = np.hstack(
            [
                cv2.cvtColor(event_frame, cv2.COLOR_GRAY2BGR),
                rgb_image,
                cv2.cvtColor(event_voxel, cv2.COLOR_RGB2BGR),
            ]
        )

        elapsed = timer.elapsed()

        print(
            f"[{idx + 1}/{len(sampler)}] | Number of events: {len(x)}"
            f" | Processing time: {elapsed * 1e3:.2f} ms"
            f" | Event range: {timestamp} - {timestamp + ms_per_frame} ms"
            f" | RGB idx: {rgb_idx}"
            f" | Frame time: {frame_time * 1e3:.2f} ms"
            f" | Voxel time: {voxel_time * 1e3:.2f} ms"
        )

        if window.show(combined_frame, elapsed * 1e3):
            break
except KeyboardInterrupt:
    pass
finally:
    cv2.destroyAllWindows()
