import cv2
import numpy as np

from ome.io.reader.mvsec import MVSECReader
from ome.io.sample.uniform import UniformSampler
from ome.repr.frame import events_to_grayscale_count_cuda
from ome.utils.timer import Timer
from tools.window import Window

reader = MVSECReader("/var/mnt/data/datasets/mvsec/outdoor_day/outdoor_day1_data.hdf5")
sampler = UniformSampler(reader, sample_rate=1, duration_ms=33)

sample_idx_to_grayscale_idx = sampler.sync_with(reader.grayscale_wallclock[:], anchor="end")
ms_per_frame = 33
window = Window("MVSEC Preview", ms_per_frame)
timer = Timer()

try:
    for idx, timestamp in enumerate(sampler.get_all_timestamps(anchor="start")):
        timer.tick()
        x, y, p, t = sampler[idx]

        # Create event frame
        event_frame = events_to_grayscale_count_cuda(x, y, width=reader.width, height=reader.height)

        # Get corresponding grayscale image
        grayscale_idx = sample_idx_to_grayscale_idx[idx]
        grayscale_image = reader.grayscale[grayscale_idx]

        # Concatenate event frame and grayscale image horizontally
        combined_frame = np.hstack([event_frame, grayscale_image])

        elapsed = timer.elapsed()

        print(
            f"[{idx + 1}/{len(sampler)}] | Number of events: {len(x)}"
            f" | Processing time: {elapsed * 1e3:.2f} ms"
            f" | Event time: {timestamp} - {timestamp + ms_per_frame} ms"
            f" | Grayscale idx: {grayscale_idx}"
        )

        if window.show(combined_frame, elapsed * 1e3):
            break
except KeyboardInterrupt:
    pass
finally:
    cv2.destroyAllWindows()
