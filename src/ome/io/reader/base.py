from abc import ABC


class BaseReader(ABC):
    def __post_init__(self):
        self.max_ms = len(self.ms_to_idx) - 1
        self.sensor_size = (self.width, self.height)

    def slice(self, start: int, end: int):
        return (
            self.x[start:end],
            self.y[start:end],
            self.p[start:end],
            self.t[start:end],
        )

    def duration(self, start_ms: int, duration_ms: int):
        """Read a duration (ms) of events."""
        if start_ms > self.max_ms:
            raise ValueError(f"start_ms must be less than {self.max_ms}.")

        end_ms = min(start_ms + duration_ms, self.max_ms)
        start = self.ms_to_idx[start_ms]
        end = self.ms_to_idx[end_ms]
        return self.slice(start, end)
