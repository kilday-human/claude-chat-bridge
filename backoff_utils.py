import random
from tenacity.wait import wait_base

class wait_fibonacci_jitter(wait_base):
    """
    Fibonacci backoff with jitter. Each wait = fib(n) ± up to 50%.
    """
    def __init__(self, max_attempts=10):
        super().__init__()
        # Precompute Fibonacci numbers up to max_attempts
        fib = [1, 1]
        for _ in range(2, max_attempts):
            fib.append(fib[-1] + fib[-2])
        self.fib = fib

    def __call__(self, retry_state):
        # retry_state.attempt_number starts at 1
        idx = retry_state.attempt_number - 1
        base = self.fib[min(idx, len(self.fib) - 1)]
        # jitter ±50%
        return base * (0.5 + random.random())

