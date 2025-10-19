import numpy as np

from ome.io.reader.base import BaseReader
from ome.utils.sync import nearest_idx


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

    def get_timestamp(self, index: int, anchor: str = "start") -> int:
        """Get the start/middle/end timestamp (in milliseconds) of the sample at given index."""
        timestamp = self.offset_ms + index * self.sample_rate * self.duration_ms

        if anchor == "start":
            return timestamp
        elif anchor == "middle":
            return timestamp + self.duration_ms // 2
        elif anchor == "end":
            return timestamp + self.duration_ms
        else:
            raise ValueError(f"Invalid anchor: {anchor}. Must be 'start', 'middle', or 'end'.")

    def get_all_timestamps(self, anchor: str = "start") -> np.ndarray:
        """Get all start/middle/end timestamps (in milliseconds) of the samples."""
        all_timestamps = np.arange(
            self.offset_ms,
            self.offset_ms + self.num_samples * self.sample_rate * self.duration_ms,
            self.sample_rate * self.duration_ms,
        )
        if anchor == "start":
            return all_timestamps
        elif anchor == "middle":
            return all_timestamps + self.duration_ms // 2
        elif anchor == "end":
            return all_timestamps + self.duration_ms
        else:
            raise ValueError(f"Invalid anchor: {anchor}. Must be 'start', 'middle', or 'end'.")

    def sync_with(self, wallclock_times: np.ndarray, anchor: str = "end") -> np.ndarray:
        """Get the nearest indices in the given wallclock times of the samples.

        Args:
            wallclock_times (np.ndarray): An array of wallclock times (in seconds) to sync with.
            anchor (str): The anchor point of the sample to consider for synchronization.
                          Can be 'start', 'middle', or 'end'. Default is 'end'.

        Returns:
            np.ndarray: A mapping from sample indices to the nearest indices in the given wallclock times.
        """
        sample_timestamps = self.get_all_timestamps(anchor=anchor)  # in milliseconds
        sample_indices = self.reader.ms_to_idx[sample_timestamps]  # map to event indices
        sample_wallclock_times = self.reader.t_wallclock[sample_indices]  # in seconds

        return nearest_idx(wallclock_times, sample_wallclock_times)
