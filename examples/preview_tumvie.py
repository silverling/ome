import cv2
import numpy as np

from ome.io.reader.tumvie import TUMVIEReader
from ome.io.sample.uniform import UniformSampler
from ome.repr.frame import events_to_grayscale_count_cuda
from ome.repr.voxel import events_to_voxel_cuda
from ome.utils.timer import Timer
from tools.window import Window


def contain_into(img, shape):
    """Contain image into given shape by padding with black pixels."""
    h, w = img.shape[:2]
    target_h, target_w = shape

    scale = min(target_h / h, target_w / w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized_img = cv2.resize(img, (new_w, new_h))

    padded_img = np.zeros((target_h, target_w, *img.shape[2:]), dtype=img.dtype)
    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2
    padded_img[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized_img

    return padded_img


def resize_and_center_crop(img, shape):
    """Resize and center crop image to given shape."""
    h, w = img.shape[:2]
    target_h, target_w = shape

    scale = max(target_h / h, target_w / w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized_img = cv2.resize(img, (new_w, new_h))

    start_y = (new_h - target_h) // 2
    start_x = (new_w - target_w) // 2
    cropped_img = resized_img[start_y : start_y + target_h, start_x : start_x + target_w]

    return cropped_img


reader = TUMVIEReader(
    "/var/mnt/data/datasets/tum-vie/bike-easy/bike-easy-events_left.h5",
    image_folder="/var/mnt/data/datasets/tum-vie/bike-easy/left_images",
    timestamps_txt="/var/mnt/data/datasets/tum-vie/bike-easy/left_images/image_timestamps_left.txt",
    calib_file="/var/mnt/data/datasets/tum-vie/camera-calibrationB.json",
)
sampler = UniformSampler(reader, sample_rate=1, duration_ms=33)

sample_idx_to_grayscale_idx = sampler.sync_with(reader.grayscale_t[:], use_t=True, anchor="end")
ms_per_frame = 33
window = Window("TUM-VIE Preview", ms_per_frame)
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
        grayscale_image = resize_and_center_crop(grayscale_image, (reader.height, reader.width))

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
