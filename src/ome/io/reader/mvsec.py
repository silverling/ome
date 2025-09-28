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

        dataset = f"/davis/{'left' if left else 'right'}/events"
        dataset: h5py.Dataset = self.h5[dataset]

        self.x: np.ndarray = dataset[:, 0]  # uint16, width: 1280
        self.y: np.ndarray = dataset[:, 1]  # uint16, height: 720

        self.p: np.ndarray = dataset[:, 3]
        self.p = np.where(self.p > 0, 1, 0).astype(np.int8)  # int8, polarity, 0 or 1

        self.t: np.ndarray = dataset[:, 2]
        self.t = ((self.t - self.t[0]) * 1e6).astype(np.int64)  # int64, microseconds, start from 0

        self.ms_to_idx = np.searchsorted(
            self.t,
            np.arange(0, self.t[-1], 1000),
            side="right",
        )

        self.width = 346
        self.height = 260

        super().__post_init__()
