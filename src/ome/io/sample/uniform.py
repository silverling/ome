from ome.io.reader.base import BaseReader


class UniformSampler:
    def __init__(self, reader: BaseReader, /, *, sample_rate: int = 7, offset: int = 0, duration_ms: int = 15):
        self.reader = reader
        self.sample_rate = sample_rate
        self.duration_ms = duration_ms
        self.offset = offset

        self.num_samples = (self.reader.max_ms - offset) // duration_ms // sample_rate

    def __getitem__(self, index: int):
        index = (index + self.offset) * self.sample_rate * self.duration_ms
        return self.reader.duration(index, self.duration_ms)

    def __len__(self):
        return self.num_samples
