import hdf5plugin  # noqa
from pathlib import Path

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

# (width, height)
camera_resolutions = [
    (1224, 1024),  # left_grayscale
    (1224, 1024),  # right_grayscale
    (640, 480),  # left_event
    (640, 480),  # right_event
]

camera_matrices = [
    np.array([[886.19107, 0.0, 610.57891], [0.0, 886.59163, 514.59271], [0.0, 0.0, 1.0]]),
    np.array([[887.80428, 0.0, 616.17757], [0.0, 888.04815, 514.71295], [0.0, 0.0, 1.0]]),
    np.array([[327.32749, 0.0, 304.97749], [0.0, 327.46184, 235.37621], [0.0, 0.0, 1.0]]),
    np.array([[327.48497, 0.0, 318.53477], [0.0, 327.55395, 230.96356], [0.0, 0.0, 1.0]]),
]

distortion_coeffs_list = [
    np.array([-0.315760, 0.104955, 0.000320, -0.000156, 0.000000]),
    np.array([-0.311523, 0.096410, 0.000623, -0.000375, 0.000000]),
    np.array([-0.031982, 0.041966, -0.000507, -0.001031, 0.000000]),
    np.array([-0.026300, 0.037995, -0.000513, 0.000167, 0.000000]),
]


class VECTORReader(BaseReader):
    """Reader for the VECTOR dataset. See https://star-datasets.github.io/vector/"""

    def __init__(
        self,
        file: str | Path,
        *,
        image_folder: str | Path | None = None,
        timestamps_txt: str | Path | None = None,
    ):
        self.h5 = h5py.File(file, "r")

        self.x: h5py.Dataset = self.h5["/events/x"]  # uint16, width: 1280
        self.y: h5py.Dataset = self.h5["/events/y"]  # uint16, height: 720
        self.p: h5py.Dataset = self.h5["/events/p"]  # int8, polarity, 0 or 1
        self.t: h5py.Dataset = self.h5["/events/t"]  # int64, microseconds, start from 0
        self.ms_to_idx: np.ndarray = self.h5["/ms_to_idx"][:]  # uint64, (N,)

        self.t_offset: int = self.h5["/t_offset"][()]  # int64, microseconds, unix timestamp in world
        self.t_wallclock = (self.t_offset + self.t[:]).astype(np.float64) / 1e6  # float64, seconds, wallclock time

        self.width = 640
        self.height = 480

        assert not ((image_folder is None) ^ (timestamps_txt is None)), (
            "Both image_folder and timestamps_txt should be provided or both should be None."
        )

        if image_folder is not None:
            image_folder = Path(image_folder)
            self.grayscale_files = sorted(image_folder.glob("*.png"))  # List[Path], 1024x1024 grayscale images

            exposure_times = np.loadtxt(timestamps_txt, dtype=np.float64)  # (N,), float64, seconds, wallclock-time
            self.grayscale_wallclock = exposure_times.mean(axis=1)  # use mean exposure time as timestamp, seconds
            assert len(self.grayscale_files) == len(self.grayscale_wallclock), "Number of images and timestamps must match."

        self.calibed_K = []
        self.calib_maps = []
        self.calib_rois = []
        for intrinsic, distortion_coeffs, resolution in zip(camera_matrices, distortion_coeffs_list, camera_resolutions):
            new_K, roi = cv2.getOptimalNewCameraMatrix(intrinsic, distortion_coeffs, resolution, 1, resolution)
            map_x, map_y = cv2.initUndistortRectifyMap(
                intrinsic,
                distortion_coeffs,
                None,
                new_K,
                resolution,
                cv2.CV_32FC1,
            )

            self.calibed_K.append(new_K)
            self.calib_rois.append(roi)
            self.calib_maps.append((map_x, map_y))

        super().__post_init__()

    def rectify(self, img, camera: str):
        """Rectify image using precomputed undistortion maps.

        Args:
            img: Input image to be rectified.
            camera: One of ["left_grayscale", "right_grayscale", "left_event", "right_event"].

        Returns:
            Rectified image.
        """
        map_x, map_y = self.calib_maps[camera_map[camera]]
        rectified_img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR)

        return rectified_img
