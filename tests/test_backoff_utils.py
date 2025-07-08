import pytest
import types
from backoff_utils import wait_fibonacci_jitter

def test_fibonacci_jitter_sequence():
    wait = wait_fibonacci_jitter(max_attempts=6)
    delays = []
    fib_vals = [1, 1, 2, 3, 5, 8]
    for attempt in range(1, 7):
        # Create a dummy object with the needed attribute
        state = types.SimpleNamespace(attempt_number=attempt)
        d = wait(state)
        base = fib_vals[attempt - 1]
        # Should be within [0.5*fib, 1.5*fib]
        assert 0.5 * base <= d <= 1.5 * base
        delays.append(d)
    # Check that a later attempt delay is larger than an earlier one
    assert delays[2] > delays[0]

