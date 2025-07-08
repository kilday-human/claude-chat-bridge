import random

def wait_fibonacci_jitter(max_attempts=5):
    # Precompute Fibonacci numbers up to max_attempts
    fib_vals = [1, 1]
    for i in range(2, max_attempts):
        fib_vals.append(fib_vals[i-1] + fib_vals[i-2])

    def wait_fn(retry_state):
        # Determine the base Fib value for this attempt
        attempt = retry_state.attempt_number
        idx = min(attempt - 1, len(fib_vals) - 1)
        base = fib_vals[idx]
        # Use a deterministic RNG seeded by attempt number
        rng = random.Random(attempt)
        # Uniform jitter between 0.5×base and 1.5×base
        return rng.uniform(0.5 * base, 1.5 * base)

    return wait_fn
