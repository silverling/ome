import numpy as np

from ome.io.reader.base import BaseReader


class UniformSampler:
    """Uniformly sample events from a reader.

    Given an event reader, this class samples events uniformly at a specified rate and duration.
    Here the sample is a window of events.
    """

    def __init__(self, reader: BaseReader, /, *, sample_rate: int = 7, duration_ms: int = 15, offset_ms: int = 0):
        self.reader = reader
        self.sample_rate = sample_rate
        self.duration_ms = duration_ms
        self.offset_ms = offset_ms

        self.num_samples = (self.reader.max_ms - offset_ms) // duration_ms // sample_rate

    def __getitem__(self, index: int):
        index = self.offset_ms + index * self.sample_rate * self.duration_ms
        return self.reader.duration(index, self.duration_ms)

    def __len__(self):
        return self.num_samples

    def get_timestamp(self, index: int) -> int:
        """Get the start timestamp (in milliseconds) of the sample at given index."""
        index = self.offset_ms + index * self.sample_rate * self.duration_ms
        return index

    def get_all_timestamps(self) -> np.ndarray:
        """Get all start timestamps (in milliseconds) of the samples."""
        return np.arange(
            self.offset_ms,
            self.offset_ms + self.num_samples * self.sample_rate * self.duration_ms,
            self.sample_rate * self.duration_ms,
        )
