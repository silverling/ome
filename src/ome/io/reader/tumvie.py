import hdf5plugin  # noqa
from pathlib import Path
import json

import h5py
import numpy as np
import cv2

from ome.io.reader.base import BaseReader

camera_map = {
    "left_grayscale": 0,
    "right_grayscale": 1,
    "left_event": 2,
    "right_event": 3,
}


class TUMVIEReader(BaseReader):
    def __init__(
        self,
        file: str | Path,
        *,
        image_folder: str | Path | None = None,
        timestamps_txt: str | Path | None = None,
        calib_file: str | Path | None = None,
    ):
        self.h5 = h5py.File(file, "r")

        self.x: h5py.Dataset = self.h5["/events/x"]  # uint16, width: 1280
        self.y: h5py.Dataset = self.h5["/events/y"]  # uint16, height: 720
        self.p: h5py.Dataset = self.h5["/events/p"]  # int8, polarity, 0 or 1
        self.t: h5py.Dataset = self.h5["/events/t"]  # int64, microseconds, start from 0, relative
        self.ms_to_idx: np.ndarray = self.h5["/ms_to_idx"][:]  # uint64, (N,)

        self.width = 1280
        self.height = 720

        assert not ((image_folder is None) ^ (timestamps_txt is None)), (
            "Both image_folder and timestamps_txt should be provided or both should be None."
        )

        if image_folder is not None:
            image_folder = Path(image_folder)
            self.grayscale_files = sorted(image_folder.glob("*.jpg"))  # List[Path], 1024x1024 grayscale images
            self.grayscale_t = np.loadtxt(timestamps_txt, dtype=np.float64)  # (N,), float64, microseconds, relative, synced
            assert len(self.grayscale_files) == len(self.grayscale_t), "Number of images and timestamps must match."

        if calib_file is not None:
            with open(calib_file, "r") as f:
                self.calib = json.load(f)["value0"]
                self.intrinsics = self.calib["intrinsics"]
                self.resolutions = self.calib["resolution"]
                self.calibed_K = []
                self.maps = []

                # [left_grayscale, right_grayscale, left_event, right_event] respectively
                for idx, intrinsic in enumerate(self.intrinsics):
                    intrinsic = intrinsic["intrinsics"]
                    K = np.array(
                        [
                            [intrinsic["fx"], 0, intrinsic["cx"]],
                            [0, intrinsic["fy"], intrinsic["cy"]],
                            [0, 0, 1],
                        ]
                    )

                    distortion_coeffs = np.array([intrinsic["k1"], intrinsic["k2"], intrinsic["k3"], intrinsic["k4"]])
                    resolution = self.resolutions[idx]
                    new_K = K.copy()
                    map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
                        K,
                        distortion_coeffs,
                        np.eye(3),
                        new_K,
                        (resolution[0], resolution[1]),
                        cv2.CV_32FC1,
                    )

                    self.calibed_K.append(new_K)
                    self.maps.append((map_x, map_y))

        super().__post_init__()

    def rectify(self, img, camera: str):
        """Rectify image using precomputed maps.

        Args:
            img: Input image.
            camera: One of 'left_grayscale', 'right_grayscale', 'left_event', 'right_event'.

        Returns:
            Rectified image.
        """
        map_x, map_y = self.maps[camera_map[camera]]
        rectified_img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR)

        return rectified_img
