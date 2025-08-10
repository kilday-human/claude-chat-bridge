#!/usr/bin/env python3
"""
Bridge test suite - validates mock and real API functionality
Run this before commits to ensure bridge stability
"""

import subprocess
import sys
import os
from typing import List, Tuple

def run_command(cmd: List[str]) -> Tuple[bool, str, str]:
    """Run a command and return (success, stdout, stderr)"""
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=30,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)

def test_compilation():
    """Test that all Python files compile"""
    print("🔍 Testing compilation...")
    
    files = ["chatgpt_wrapper.py", "claude_wrapper.py", "cli_bridge.py"]
    for file in files:
        success, stdout, stderr = run_command(["python3", "-m", "py_compile", file])
        if not success:
            print(f"❌ {file} compilation failed: {stderr}")
            return False
        print(f"✅ {file} compiles")
    
    return True

def test_mock_scenarios():
    """Test all mock scenarios"""
    print("\n🎭 Testing mock scenarios...")
    
    scenarios = [
        {
            "name": "Basic bridge-ok",
            "cmd": ["python3", "cli_bridge.py", "Reply 'bridge-ok' only.", "1", "--mock"]
        },
        {
            "name": "Multi-turn sequential", 
            "cmd": ["python3", "cli_bridge.py", "One-sentence proof you're alive; then echo 'done'.", "2", "--mock", "--no-parallel"]
        },
        {
            "name": "Parallel execution",
            "cmd": ["python3", "cli_bridge.py", "Reply 'bridge-ok' only.", "1", "--mock", "--parallel"]
        },
        {
            "name": "Math operation",
            "cmd": ["python3", "cli_bridge.py", "What's 17 * 234 + 892?", "1", "--mock"]
        }
    ]
    
    all_passed = True
    for scenario in scenarios:
        print(f"  Testing: {scenario['name']}")
        success, stdout, stderr = run_command(scenario["cmd"])
        
        if not success:
            print(f"❌ {scenario['name']} failed to run: {stderr}")
            all_passed = False
            continue
        
        # Check that we get completion and reasonable output
        if "Bridge complete" not in stdout:
            print(f"❌ {scenario['name']} didn't complete properly")
            print(f"   Output: {stdout}")
            all_passed = False
        elif len(stdout.strip()) < 50:  # Too short, probably errored
            print(f"❌ {scenario['name']} output too short, may have errored")
            print(f"   Output: {stdout}")
            all_passed = False
        else:
            print(f"✅ {scenario['name']} passed")
    
    return all_passed

def test_real_api():
    """Test real API if keys are available"""
    print("\n🌐 Testing real API...")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  No OPENAI_API_KEY found, skipping real API tests")
        return True
    
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  No ANTHROPIC_API_KEY found, skipping real API tests")
        return True
    
    print("  Testing single real call...")
    success, stdout, stderr = run_command([
        "python3", "cli_bridge.py", 
        "Reply 'test-ok' only.", "1", 
        "--no-parallel", "--max-tokens", "64"
    ])
    
    if not success:
        print(f"❌ Real API test failed: {stderr}")
        return False
    
    if "test-ok" not in stdout.lower() and "ok" not in stdout.lower():
        print(f"❌ Real API test didn't get expected response")
        print(f"   Output: {stdout}")
        return False
    
    print("✅ Real API test passed")
    return True

def main():
    """Run all tests"""
    print("🚀 Running bridge test suite...")
    
    tests = [
        ("Compilation", test_compilation),
        ("Mock scenarios", test_mock_scenarios),
        ("Real API", test_real_api)
    ]
    
    all_passed = True
    results = []
    
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
            all_passed = False
    
    # Summary
    print(f"\n📊 Test Results:")
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} {test_name}")
    
    if all_passed:
        print(f"\n🎉 All tests passed! Bridge is ready.")
        sys.exit(0)
    else:
        print(f"\n💥 Some tests failed. Check output above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
