from pathlib import Path

import h5py
import numpy as np

from ome.io.reader.base import BaseReader


class M3EDReader(BaseReader):
    def __init__(
        self,
        file: str | Path,
        left: bool = True,
    ):
        self.h5 = h5py.File(file, "r")

        dataset = f"/prophesee/{'left' if left else 'right'}/"
        self.x: h5py.Dataset = self.h5[dataset + "x"]  # uint16, width: 1280
        self.y: h5py.Dataset = self.h5[dataset + "y"]  # uint16, height: 720
        self.p: h5py.Dataset = self.h5[dataset + "p"]  # int8, polarity, 0 or 1
        self.t: h5py.Dataset = self.h5[dataset + "t"]  # int64, microseconds, start from 0
        self.ms_to_idx: np.ndarray = self.h5[dataset + "ms_map_idx"][:]  # uint64, (N,)

        self.width = 1280
        self.height = 720
        self.sensor_size = (self.width, self.height)
