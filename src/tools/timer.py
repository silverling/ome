import time


class Timer:
    def __init__(self):
        pass

    def tick(self):
        self.tic = time.perf_counter()

    def elapsed(self):
        elapsed = time.perf_counter() - self.tic
        self.tic = time.perf_counter()
        return elapsed
