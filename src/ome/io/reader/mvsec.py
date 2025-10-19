import hdf5plugin  # noqa
from pathlib import Path

import h5py
import numpy as np

from ome.io.reader.base import BaseReader


class MVSECReader(BaseReader):
    """Reader for the MVSEC dataset. See https://daniilidis-group.github.io/mvsec/"""

    def __init__(
        self,
        file: str | Path,
        left: bool = True,
    ):
        self.h5 = h5py.File(file, "r")

        prefix = f"/davis/{'left' if left else 'right'}"
        dataset: h5py.Dataset = self.h5[f"{prefix}/events"]

        # Events
        self.x: np.ndarray = dataset[:, 0].astype(np.uint16)  # uint16, width: 1280
        self.y: np.ndarray = dataset[:, 1].astype(np.uint16)  # uint16, height: 720

        self.p: np.ndarray = dataset[:, 3]
        self.p = np.where(self.p > 0, 1, 0).astype(np.int8)  # int8, polarity, 0 or 1

        self.t_wallclock = dataset[:, 2]  # float64, seconds, wallclock time
        self.t = ((self.t_wallclock - self.t_wallclock[0]) * 1e6).astype(np.int64)  # int64, microseconds, start from 0

        # Grayscale images from the DAVIS camera.
        self.grayscale = self.h5[f"{prefix}/image_raw"]  # [N, H, W], uint8
        self.grayscale_wallclock = self.h5[f"{prefix}/image_raw_ts"]  # [N], float64, seconds, wallclock time

        self.width = 346
        self.height = 260

        super().__post_init__()
