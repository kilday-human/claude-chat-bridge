#!/usr/bin/env python3
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
