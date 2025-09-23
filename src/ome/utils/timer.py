import time


class Timer:
    def __init__(self):
        pass

    def tick(self):
        self._tic = time.perf_counter()

    def elapsed(self):
        """Return the elapsed time (in second) since the last tick() or elapsed() call, and reset the timer."""
        elapsed = time.perf_counter() - self._tic
        self._tic = time.perf_counter()
        return elapsed
