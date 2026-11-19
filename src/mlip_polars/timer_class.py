import time
from dataclasses import dataclass
from functools import wraps


@dataclass
class Measurement:
    name: str
    elapsed_s: float


class Timer:
    def __init__(self):
        self.records: list[Measurement] = []

    def measure(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()

            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start

                self.records.append(
                    Measurement(
                        name=func.__name__,
                        elapsed_s=elapsed,
                    )
                )

        return wrapper
