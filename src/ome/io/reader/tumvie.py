import hdf5plugin  # noqa
from pathlib import Path

import h5py
import numpy as np

from ome.io.reader.base import BaseReader


class TUMVIEReader(BaseReader):
    def __init__(
        self,
        file: str | Path,
    ):
        self.h5 = h5py.File(file, "r")

        self.x: h5py.Dataset = self.h5["/events/x"]  # uint16, width: 1280
        self.y: h5py.Dataset = self.h5["/events/y"]  # uint16, height: 720
        self.p: h5py.Dataset = self.h5["/events/p"]  # int8, polarity, 0 or 1
        self.t: h5py.Dataset = self.h5["/events/t"]  # int64, microseconds, start from 0
        self.ms_to_idx: np.ndarray = self.h5["/ms_to_idx"][:]  # uint64, (N,)
        self.max_ms = len(self.ms_to_idx) - 1

        self.width = 1280
        self.height = 720
        self.sensor_size = (self.width, self.height)
