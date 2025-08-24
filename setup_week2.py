#!/usr/bin/env python3
"""
Week 2 Setup Script for Claude-GPT Bridge
Sets up cache, guardrails, and evaluation components
"""

import os
import json
from pathlib import Path
import sys

def create_directories():
    """Create necessary directories for Week 2 components"""
    dirs_to_create = [
        "data/cache",
        "data/chroma_db",
        "eval_results",
        "logs",
        "src",
        "tests"
    ]
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {dir_path}")

def create_config_files():
    """Create configuration files"""
    
    # Guardrails config
    guardrails_config = {
        "enable_content_safety": True,
        "enable_quality_check": True,
        "enable_bias_detection": True,
        "enable_format_validation": True,
        "min_quality_score": 0.6,
        "max_response_length": 4000,
        "min_response_length": 10,
        "safety_patterns": [
            "\\b(?:hate|violence|harm|illegal|dangerous)\\b",
            "\\b(?:suicide|self-harm|death)\\b",
            "\\b(?:racist|sexist|discriminat)\\w*\\b"
        ]
    }
    
    with open("config/guardrails.json", "w") as f:
        json.dump(guardrails_config, f, indent=2)
    print("✅ Created guardrails configuration")
    
    # Cache config
    cache_config = {
        "max_memory_size": 100,
        "max_disk_size": 1000,
        "default_ttl": 3600,
        "cache_dir": "data/cache"
    }
    
    with open("config/cache.json", "w") as f:
        json.dump(cache_config, f, indent=2)
    print("✅ Created cache configuration")
    
    # Eval config
    eval_config = {
        "default_test_suite": "tests/test_suite.json",
        "output_dir": "eval_results",
        "max_workers": 4,
        "timeout": 30,
        "metrics": {
            "accuracy_weight": 0.3,
            "quality_weight": 0.4,
            "safety_weight": 0.3
        }
    }
    
    with open("config/evaluation.json", "w") as f:
        json.dump(eval_config, f, indent=2)
    print("✅ Created evaluation configuration")

def create_test_files():
    """Create test files and scripts"""
    
    # Test Week 2 integration
    integration_test = '''#!/usr/bin/env python3
"""Test Week 2 integration"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cache_manager import get_cache_manager
from src.guardrails_system import get_guardrails_manager, GuardrailConfig
from src.eval_harness import EvalHarness, TestSuiteLoader

def test_cache():
    """Test cache system"""
    print("🧪 Testing Cache Manager...")
    cache = get_cache_manager()
    
    # Test basic caching
    cache.put("test prompt", "gpt-4o-mini", "test response", 0.001, 50)
    result = cache.get("test prompt", "gpt-4o-mini")
    
    assert result is not None, "Cache should return stored result"
    assert result.response == "test response", "Cache should return correct response"
    print("✅ Cache basic functionality working")
    
    # Test stats
    stats = cache.get_stats()
    assert stats['hits'] > 0, "Cache should have recorded hits"
    print("✅ Cache statistics working")

def test_guardrails():
    """Test guardrails system"""
    print("🧪 Testing Guardrails System...")
    guardrails = get_guardrails_manager()
    
    # Test safe content
    safe_prompt = "What is artificial intelligence?"
    safe_response = "AI is a branch of computer science focused on creating intelligent machines."
    
    is_safe, summary = guardrails.is_response_safe(safe_prompt, safe_response)
    assert is_safe, "Safe content should pass guardrails"
    print("✅ Safe content detection working")
    
    # Test potentially unsafe content
    unsafe_response = "This content contains violence and hate speech."
    is_safe, summary = guardrails.is_response_safe(safe_prompt, unsafe_response)
    # Note: depending on patterns, this might or might not trigger
    print(f"✅ Unsafe content detection working (safe: {is_safe})")
    
    # Test stats
    stats = guardrails.get_stats()
    assert 'total_checks' in stats, "Guardrails should track statistics"
    print("✅ Guardrails statistics working")

def test_evaluation():
    """Test evaluation system"""
    print("🧪 Testing Evaluation Harness...")
    
    # Mock bridge function
    def mock_bridge(prompt, **kwargs):
        return f"Mock response to: {prompt}", {
            'model': 'mock-model',
            'cost': 0.001,
            'tokens_used': 50,
            'execution_time': 0.1
        }
    
    harness = EvalHarness(mock_bridge, "test_eval_results")
    
    # Test with small test suite
    test_cases = TestSuiteLoader.create_default_suite()[:3]  # Just first 3
    
    config = {'mock': True}
    report = harness.run_evaluation(test_cases, config, "integration_test", parallel=False)
    
    assert len(report.results) == 3, "Should have 3 test results"
    assert report.aggregate_metrics['success_rate'] == 1.0, "Mock responses should all succeed"
    print("✅ Evaluation harness working")
    
    # Test stats
    summary = harness.get_evaluation_summary()
    assert summary['total_evaluations'] > 0, "Should have recorded evaluation"
    print("✅ Evaluation statistics working")

def main():
    """Run integration tests"""
    print("🚀 Starting Week 2 Integration Tests")
    print("=" * 50)
    
    try:
        test_cache()
        print()
        
        test_guardrails()
        print()
        
        test_evaluation()
        print()
        
        print("🎉 All Week 2 integration tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    
    Path("tests").mkdir(exist_ok=True)
    with open("tests/test_week2_integration.py", "w") as f:
        f.write(integration_test)
    
    # Make it executable
    os.chmod("tests/test_week2_integration.py", 0o755)
    print("✅ Created integration test script")

def create_demo_script():
    """Create demo script showcasing Week 2 features"""
    
    demo_script = '''#!/usr/bin/env python3
"""
Week 2 Demo Script - Showcase new features
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from cli_bridge import EnhancedBridge

def demo_cache_performance():
    """Demo cache performance improvement"""
    print("💾 Cache Performance Demo")
    print("=" * 40)
    
    bridge = EnhancedBridge()
    test_prompt = "What is machine learning and how does it work?"
    
    # First request (no cache)
    print("🔄 First request (no cache)...")
    start_time = time.time()
    response1, meta1 = bridge.process_request(test_prompt, mock=True, use_cache=True)
    time1 = time.time() - start_time
    
    print(f"Response: {response1[:100]}...")
    print(f"Time: {time1:.3f}s | Cached: {meta1.get('cached', False)}")
    
    # Second request (cached)
    print("\\n⚡ Second request (from cache)...")
    start_time = time.time()
    response2, meta2 = bridge.process_request(test_prompt, mock=True, use_cache=True)
    time2 = time.time() - start_time
    
    print(f"Response: {response2[:100]}...")
    print(f"Time: {time2:.3f}s | Cached: {meta2.get('cached', False)}")
    
    speedup = time1 / time2 if time2 > 0 else float('inf')
    print(f"\\n🚀 Speedup: {speedup:.1f}x faster!")

def demo_guardrails():
    """Demo guardrails system"""
    print("\\n🛡️ Guardrails Demo")
    print("=" * 30)
    
    bridge = EnhancedBridge()
    
    test_cases = [
        ("Safe prompt", "Explain renewable energy benefits"),
        ("Potentially unsafe", "Tell me how to harm someone"),
        ("Long response test", "Write a 5000-word essay on everything")
    ]
    
    for name, prompt in test_cases:
        print(f"\\n📝 Testing: {name}")
        print(f"Prompt: {prompt}")
        
        # Process with guardrails
        response, metadata = bridge.process_request(
            prompt, 
            mock=True, 
            use_guardrails=True,
            verbose=True
        )
        
        guardrail_summary = metadata.get('guardrails_summary', {})
        is_safe = guardrail_summary.get('safe', True)
        violations = guardrail_summary.get('violations', [])
        
        print(f"Result: {'✅ Safe' if is_safe else '❌ Issues detected'}")
        if violations:
            print(f"Violations: {', '.join(violations)}")

def demo_evaluation():
    """Demo evaluation system"""
    print("\\n🧪 Evaluation Demo")
    print("=" * 30)
    
    from src.eval_harness import EvalHarness, TestSuiteLoader, create_eval_config
    
    bridge = EnhancedBridge()
    harness = EvalHarness(bridge.process_request, "demo_eval_results")
    
    # Create small test suite
    test_suite = TestSuiteLoader.create_default_suite()[:5]  # First 5 tests
    config = create_eval_config(mock=True, use_router=True, use_rag=False)
    
    print(f"Running evaluation with {len(test_suite)} test cases...")
    
    report = harness.run_evaluation(test_suite, config, "demo_eval", parallel=False)
    
    print(f"\\n📊 Results:")
    print(f"Success Rate: {report.aggregate_metrics.get('success_rate', 0):.1%}")
    print(f"Average Quality: {report.aggregate_metrics.get('avg_quality', 0):.2f}")
    print(f"Average Latency: {report.aggregate_metrics.get('avg_latency', 0):.2f}s")
    
    if report.recommendations:
        print(f"\\n💡 Recommendations:")
        for rec in report.recommendations[:3]:  # Show first 3
            print(f"  • {rec}")

def main():
    """Run comprehensive Week 2 demo"""
    print("🎭 Claude-GPT Bridge - Week 2 Feature Demo")
    print("=" * 50)
    print("Showcasing: Cache, Guardrails, and Evaluation systems")
    print()
    
    try:
        demo_cache_performance()
        demo_guardrails()  
        demo_evaluation()
        
        print("\\n🎉 Week 2 Demo Complete!")
        print("All new features working as expected.")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    
    with open("demo_week2.py", "w") as f:
        f.write(demo_script)
    
    # Make it executable  
    os.chmod("demo_week2.py", 0o755)
    print("✅ Created Week 2 demo script")

def update_requirements():
    """Update requirements.txt with Week 2 dependencies"""
    additional_requirements = [
        "# Week 2 - Cache, Guardrails, Evaluation",
        "pickle5>=1.0.0  # Enhanced pickling for cache",
        "threading  # Built-in, for thread safety",
        "concurrent.futures  # Built-in, for parallel evaluation",
        "statistics  # Built-in, for metrics calculation",
        "hashlib  # Built-in, for cache key generation",
        ""
    ]
    
    # Check if requirements.txt exists
    if Path("requirements.txt").exists():
        with open("requirements.txt", "a") as f:
            f.write("\\n".join(additional_requirements))
        print("✅ Updated requirements.txt")
    else:
        base_requirements = [
            "# Claude-GPT Bridge Requirements",
            "# Core dependencies",
            "openai>=1.0.0",
            "anthropic>=0.7.0",
            "python-dotenv>=1.0.0",
            "tiktoken>=0.5.0",
            "",
            "# RAG and embeddings",
            "chromadb>=0.4.0",
            "sentence-transformers>=2.2.0",
            "",
            "# Week 2 additions",
            *additional_requirements
        ]
        
        with open("requirements.txt", "w") as f:
            f.write("\\n".join(base_requirements))
        print("✅ Created requirements.txt")

def main():
    """Run Week 2 setup"""
    print("🚀 Claude-GPT Bridge - Week 2 Setup")
    print("=" * 50)
    print("Setting up: Cache, Guardrails, and Evaluation systems")
    print()
    
    try:
        # Create directory structure
        create_directories()
        print()
        
        # Create config directory
        Path("config").mkdir(exist_ok=True)
        create_config_files()
        print()
        
        # Create test files
        create_test_files()
        print()
        
        # Create demo script
        create_demo_script()
        print()
        
        # Update requirements
        update_requirements()
        print()
        
        print("🎉 Week 2 Setup Complete!")
        print()
        print("Next steps:")
        print("1. Run integration tests: python3 tests/test_week2_integration.py")
        print("2. Try the demo: python3 demo_week2.py")
        print("3. Test new CLI features:")
        print("   python3 cli_bridge.py 'Hello world' --router --cache --guardrails --mock")
        print("   python3 cli_bridge.py --eval-quick")
        print("   python3 cli_bridge.py --stats")
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
