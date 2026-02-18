import time

class Benchmark:
    def __init__(self):
        self.times = []

    def measure(self, func, *args):
        start = time.perf_counter()
        func(*args)
        dt = time.perf_counter() - start
        self.times.append(dt)
        return dt

    def fps(self):
        if not self.times:
            return 0
        avg = sum(self.times[-30:]) / min(len(self.times),30)
        return 1/avg
