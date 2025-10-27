from pathlib import Path

import cv2
import h5py
import hdf5plugin  # noqa
import numpy as np

from ome.io.reader.base import BaseReader


class DSECReader(BaseReader):
    """Reader for the DSEC dataset. See https://dsec.ifi.uzh.ch/"""

    def __init__(
        self,
        file: str | Path,
        /,
        *,
        event_rectify_map_file: str | Path | None = None,
        image_folder: str | Path | None = None,
        timestamps_txt: str | Path | None = None,
    ):
        self.h5 = h5py.File(file, "r")

        self.x: h5py.Dataset = self.h5["/events/x"]  # uint16, width: 1280
        self.y: h5py.Dataset = self.h5["/events/y"]  # uint16, height: 720
        self.p: h5py.Dataset = self.h5["/events/p"]  # int8, polarity, 0 or 1
        self.t: h5py.Dataset = self.h5["/events/t"]  # int64, microseconds, start from 0
        self.ms_to_idx: np.ndarray = self.h5["/ms_to_idx"][:]  # uint64, (N,)
        self.width = 640
        self.height = 480

        t_offset: int = self.h5["/t_offset"][()]  # int64, microseconds, unix timestamp in world
        self.t_wallclock = (t_offset + self.t[:]).astype(np.float64) / 1e6  # float64, seconds, wallclock time

        # Load rectify map if provided
        if event_rectify_map_file is not None:
            rectify_map = h5py.File(event_rectify_map_file, "r")["/rectify_map"][:]  # (H, W, 2), float32
            self.calib_map_x = rectify_map[:, :, 0]
            self.calib_map_y = rectify_map[:, :, 1]

        assert not ((image_folder is None) ^ (timestamps_txt is None)), (
            "Both image_folder and timestamps_txt should be provided or both should be None."
        )

        if image_folder is not None:
            image_folder = Path(image_folder)
            self.rgb_files = sorted(image_folder.glob("*.png"))  # List[Path], 1440x1080 rgb images, same aspect ratio as events
            self.rgb_wallclock = np.loadtxt(timestamps_txt, dtype=np.float64) / 1e6  # (N,), float64, seconds, wallclock time

        super().__post_init__()

    def rectify_voxel(self, voxel: np.ndarray) -> np.ndarray:
        """Rectify event voxel grid using the calibration rectify map.

        Args:
            voxel (np.ndarray): (N_bins, H, W) event voxel grid.

        Returns:
            np.ndarray: (N_bins, H, W) rectified event voxel grid.
        """
        voxel = cv2.remap(voxel, self.calib_map_x, self.calib_map_y, interpolation=cv2.INTER_LINEAR)

        return voxel
