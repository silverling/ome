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
        super().__init__()

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

    def slice(self, start: int, end: int):
        return (
            self.x[start:end],
            self.y[start:end],
            self.p[start:end],
            self.t[start:end],
        )

    def duration(self, start_ms: int, length_ms: int):
        """Read a duration (ms) of events."""
        if start_ms > self.max_ms:
            raise ValueError(f"start_ms must be less than {self.max_ms}.")

        end_ms = min(start_ms + length_ms, self.max_ms)
        start = self.ms_to_idx[start_ms]
        end = self.ms_to_idx[end_ms]
        return self.slice(start, end)
