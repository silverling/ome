import abc


class BaseReader(abc.ABC):
    @abc.abstractmethod
    def slice(self, start: int, end: int):
        pass

    @abc.abstractmethod
    def duration(self, start_ms: int, length_ms: int):
        pass
