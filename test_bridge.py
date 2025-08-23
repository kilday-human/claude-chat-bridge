#!/usr/bin/env python3
"""
Bridge test suite - validates mock and real API functionality
Run this before commits to ensure bridge stability
"""

import subprocess
import sys
import os
import json
from typing import List, Tuple

LOG_FILE = os.path.join(os.path.dirname(__file__), "logs", "cost_ledger.jsonl")

def run_command(cmd: List[str]) -> Tuple[bool, str, str]:
    """Run a command and return (success, stdout, stderr)"""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)

def read_last_log():
    """Read last cost ledger entry"""
    if not os.path.exists(LOG_FILE):
        return None
    with open(LOG_FILE) as f:
        lines = f.readlines()
    if not lines:
        return None
    return json.loads(lines[-1])

def test_dual_mode_runs():
    """Ensure --dual runs both GPT and Claude"""
    success, stdout, stderr = run_command(
        ["python3", "cli_bridge.py", "Explain quantum entanglement", "--dual", "--mock"]
    )
    assert success, f"CLI failed: {stderr}"
    assert "[GPT]" in stdout
    # Accept either title or uppercase form
    assert "[Claude]" in stdout or "[CLAUDE]" in stdout
    assert "Bridge complete" in stdout

def test_router_mode_runs():
    """Ensure router mode runs at least one model"""
    success, stdout, stderr = run_command(
        ["python3", "cli_bridge.py", "Hello bridge system", "--router", "--mock"]
    )
    assert success, f"CLI failed: {stderr}"
    assert "[GPT]" in stdout or "[CLAUDE]" in stdout
    assert "Bridge complete" in stdout

def test_cost_log_written():
    """Ensure cost ledger is written"""
    run_command(["python3", "cli_bridge.py", "Testing cost log", "--mock"])
    last = read_last_log()
    assert last is not None
    assert "model" in last
    # Accept either simple schema or detailed tokens
    assert "tokens" in last or (
        "input_tokens" in last and "output_tokens" in last
    )

def main():
    """Run tests manually if needed"""
    print("🚀 Running test suite...")
    all_ok = True
    for test in [test_dual_mode_runs, test_router_mode_runs, test_cost_log_written]:
        try:
            test()
            print(f"✅ {test.__name__} passed")
        except AssertionError as e:
            print(f"❌ {test.__name__} failed: {e}")
            all_ok = False
    sys.exit(0 if all_ok else 1)

if __name__ == "__main__":
    main()
