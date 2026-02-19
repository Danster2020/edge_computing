import time
import csv
import numpy as np

class Benchmark:
    def __init__(self):
        self.times = []
        self.start_time = None

    def measure(self, func, *args):
        start = time.perf_counter()
        func(*args)
        dt = time.perf_counter() - start
        self.times.append(dt)
        return dt

    def fps(self):
        if not self.times:
            return 0
        avg = sum(self.times[-30:]) / min(len(self.times), 30)
        return 1 / avg

    def average_latency_ms(self):
        if not self.times:
            return 0
        return np.mean(self.times) * 1000

    def percentile_latency_ms(self, p=95):
        if not self.times:
            return 0
        return np.percentile(self.times, p) * 1000

    def save_csv(self, filename="benchmark_log.csv"):
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Frame", "Latency (ms)"])
            for i, t in enumerate(self.times):
                writer.writerow([i+1, t * 1000])

        print(f"Saved benchmark to {filename}")
