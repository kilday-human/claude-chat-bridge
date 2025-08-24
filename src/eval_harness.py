"""
Advanced Evaluation Harness for Claude-GPT Bridge
Comprehensive testing with metrics, ablations, and performance analysis
"""

import json
import time
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, asdict, field
from pathlib import Path
from enum import Enum
import statistics
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback


class EvalMetricType(Enum):
    """Types of evaluation metrics"""
    ACCURACY = "accuracy"
    LATENCY = "latency"
    COST = "cost"
    QUALITY = "quality"
    SAFETY = "safety"
    RELEVANCE = "relevance"
    COHERENCE = "coherence"
    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"


@dataclass
class TestCase:
    """Individual test case for evaluation"""
    id: str
    prompt: str
    expected_response: Optional[str] = None
    category: str = "general"
    difficulty: str = "medium"  # easy, medium, hard
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            # Generate ID from prompt hash
            self.id = hashlib.md5(self.prompt.encode()).hexdigest()[:8]


@dataclass
class EvalResult:
    """Result of a single evaluation"""
    test_case_id: str
    prompt: str
    response: str
    model: str
    metrics: Dict[str, float]
    metadata: Dict[str, Any]
    execution_time: float
    success: bool
    error: Optional[str] = None


@dataclass
class EvalReport:
    """Complete evaluation report"""
    eval_id: str
    timestamp: float
    config: Dict[str, Any]
    results: List[EvalResult]
    aggregate_metrics: Dict[str, float]
    performance_analysis: Dict[str, Any]
    cost_analysis: Dict[str, Any]
    recommendations: List[str]


class MetricCalculator:
    """Calculates various evaluation metrics"""
    
    @staticmethod
    def calculate_accuracy(expected: str, actual: str) -> float:
        """Basic accuracy using string similarity"""
        if not expected or not actual:
            return 0.0
        
        # Simple word overlap accuracy
        expected_words = set(expected.lower().split())
        actual_words = set(actual.lower().split())
        
        if not expected_words:
            return 1.0 if not actual_words else 0.0
        
        intersection = len(expected_words.intersection(actual_words))
        union = len(expected_words.union(actual_words))
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def calculate_relevance(prompt: str, response: str) -> float:
        """Calculate response relevance to prompt"""
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        
        # Filter out common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        prompt_words = prompt_words - common_words
        response_words = response_words - common_words
        
        if not prompt_words:
            return 1.0
        
        overlap = len(prompt_words.intersection(response_words))
        return min(1.0, overlap / len(prompt_words) + 0.3)  # Base relevance + overlap bonus
    
    @staticmethod
    def calculate_coherence(response: str) -> float:
        """Calculate response coherence"""
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        
        if len(sentences) <= 1:
            return 1.0
        
        # Check for consistent topic (repeated key terms)
        all_words = response.lower().split()
        word_freq = {}
        for word in all_words:
            if len(word) > 4:  # Focus on meaningful words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        repeated_terms = sum(1 for count in word_freq.values() if count > 1)
        coherence_score = min(1.0, repeated_terms / (len(sentences) * 0.5))
        
        return coherence_score
    
    @staticmethod
    def calculate_completeness(prompt: str, response: str) -> float:
        """Calculate response completeness"""
        # Question indicators
        question_indicators = ['what', 'how', 'why', 'when', 'where', 'who', 'which']
        questions_asked = sum(1 for indicator in question_indicators if indicator in prompt.lower())
        
        if questions_asked == 0:
            return 1.0  # Not a question-based prompt
        
        # Check if response addresses the questions
        response_lower = response.lower()
        coverage = 0
        
        for indicator in question_indicators:
            if indicator in prompt.lower():
                # Look for relevant response patterns
                if indicator == 'what' and any(word in response_lower for word in ['is', 'are', 'means', 'refers']):
                    coverage += 1
                elif indicator == 'how' and any(word in response_lower for word in ['by', 'through', 'using', 'process']):
                    coverage += 1
                elif indicator == 'why' and any(word in response_lower for word in ['because', 'due to', 'reason', 'since']):
                    coverage += 1
                # Add more patterns as needed
        
        return min(1.0, coverage / questions_asked) if questions_asked > 0 else 1.0
    
    @staticmethod
    def calculate_consistency(responses: List[str]) -> float:
        """Calculate consistency across multiple responses"""
        if len(responses) < 2:
            return 1.0
        
        # Simple consistency based on response similarity
        similarities = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                similarity = MetricCalculator.calculate_accuracy(responses[i], responses[j])
                similarities.append(similarity)
        
        return statistics.mean(similarities) if similarities else 1.0


class TestSuiteLoader:
    """Loads test suites from various sources"""
    
    @staticmethod
    def load_from_file(filepath: str) -> List[TestCase]:
        """Load test cases from JSON file"""
        path = Path(filepath)
        if not path.exists():
            return []
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            test_cases = []
            for item in data:
                if isinstance(item, dict):
                    test_cases.append(TestCase(**item))
            
            return test_cases
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            print(f"Error loading test suite from {filepath}: {e}")
            return []
    
    @staticmethod
    def create_default_suite() -> List[TestCase]:
        """Create default test suite for basic evaluation"""
        return [
            TestCase(
                id="basic_qa_1",
                prompt="What is artificial intelligence?",
                category="knowledge",
                difficulty="easy",
                tags=["ai", "definition"]
            ),
            TestCase(
                id="reasoning_1",
                prompt="If a train leaves Station A at 2 PM traveling at 60 mph, and another train leaves Station B at 3 PM traveling at 80 mph toward Station A, and the stations are 280 miles apart, when will they meet?",
                category="reasoning",
                difficulty="medium",
                tags=["math", "word_problem"]
            ),
            TestCase(
                id="creative_1",
                prompt="Write a short story about a robot who learns to paint.",
                category="creative",
                difficulty="medium",
                tags=["creative", "storytelling"]
            ),
            TestCase(
                id="analysis_1",
                prompt="Analyze the pros and cons of remote work from both employee and employer perspectives.",
                category="analysis",
                difficulty="hard",
                tags=["analysis", "business", "comparison"]
            ),
            TestCase(
                id="technical_1",
                prompt="Explain how a REST API works and provide a simple example.",
                category="technical",
                difficulty="medium",
                tags=["programming", "web", "api"]
            ),
            TestCase(
                id="ethical_1",
                prompt="What are the ethical considerations when developing AI systems?",
                category="ethics",
                difficulty="hard",
                tags=["ethics", "ai", "philosophy"]
            ),
            TestCase(
                id="factual_1",
                prompt="What is the capital of Australia?",
                expected_response="Canberra",
                category="factual",
                difficulty="easy",
                tags=["geography", "facts"]
            ),
            TestCase(
                id="complex_reasoning_1",
                prompt="A company has 3 departments. Marketing has 25% more employees than Sales. Engineering has twice as many employees as Marketing. If the total is 156 employees, how many are in each department?",
                category="reasoning",
                difficulty="hard",
                tags=["math", "algebra", "word_problem"]
            )
        ]
    
    @staticmethod
    def create_stress_test_suite(size: int = 50) -> List[TestCase]:
        """Create stress test suite with many variations"""
        base_prompts = [
            "Explain the concept of {}",
            "What are the benefits of {}?",
            "How does {} work?",
            "Compare {} and {}",
            "Analyze the impact of {}"
        ]
        
        topics = [
            "machine learning", "blockchain", "quantum computing", "renewable energy",
            "climate change", "artificial intelligence", "biotechnology", "space exploration",
            "cybersecurity", "data privacy", "virtual reality", "automation"
        ]
        
        test_cases = []
        for i in range(size):
            prompt_template = base_prompts[i % len(base_prompts)]
            topic = topics[i % len(topics)]
            
            if "{} and {}" in prompt_template:
                topic2 = topics[(i + 1) % len(topics)]
                prompt = prompt_template.format(topic, topic2)
            else:
                prompt = prompt_template.format(topic)
            
            test_cases.append(TestCase(
                id=f"stress_{i+1}",
                prompt=prompt,
                category="stress",
                difficulty="medium",
                tags=["stress_test", topic.replace(" ", "_")]
            ))
        
        return test_cases


class EvalHarness:
    """Main evaluation harness"""
    
    def __init__(
        self,
        bridge_function: Callable,
        output_dir: str = "eval_results",
        max_workers: int = 4,
        timeout: int = 30
    ):
        self.bridge_function = bridge_function
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.max_workers = max_workers
        self.timeout = timeout
        
        # Metrics calculator
        self.metrics = MetricCalculator()
        
        # Performance tracking
        self.eval_history = []
    
    def run_evaluation(
        self,
        test_suite: Union[List[TestCase], str],
        config: Dict[str, Any] = None,
        eval_name: str = None,
        parallel: bool = True
    ) -> EvalReport:
        """Run comprehensive evaluation"""
        start_time = time.time()
        
        # Load test suite
        if isinstance(test_suite, str):
            test_cases = TestSuiteLoader.load_from_file(test_suite)
        else:
            test_cases = test_suite
        
        if not test_cases:
            raise ValueError("No test cases to evaluate")
        
        # Generate eval ID
        eval_id = f"eval_{int(start_time)}_{len(test_cases)}"
        if eval_name:
            eval_id = f"{eval_name}_{eval_id}"
        
        print(f"Starting evaluation: {eval_id}")
        print(f"Test cases: {len(test_cases)}")
        print(f"Parallel: {parallel}")
        
        # Run evaluations
        if parallel:
            results = self.run_parallel_evaluation(test_cases, config)
        else:
            results = self.run_sequential_evaluation(test_cases, config)
        
        # Calculate aggregate metrics
        aggregate_metrics = self.calculate_aggregate_metrics(results)
        
        # Performance analysis
        performance_analysis = self.analyze_performance(results)
        
        # Cost analysis
        cost_analysis = self.analyze_costs(results)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(results, aggregate_metrics, performance_analysis)
        
        # Create report
        report = EvalReport(
            eval_id=eval_id,
            timestamp=start_time,
            config=config or {},
            results=results,
            aggregate_metrics=aggregate_metrics,
            performance_analysis=performance_analysis,
            cost_analysis=cost_analysis,
            recommendations=recommendations
        )
        
        # Save report
        self.save_report(report)
        
        # Update history
        self.eval_history.append({
            'eval_id': eval_id,
            'timestamp': start_time,
            'test_count': len(test_cases),
            'success_rate': aggregate_metrics.get('success_rate', 0),
            'avg_latency': aggregate_metrics.get('avg_latency', 0),
            'total_cost': cost_analysis.get('total_cost', 0)
        })
        
        execution_time = time.time() - start_time
        print(f"Evaluation completed in {execution_time:.2f}s")
        
        return report
    
    def run_sequential_evaluation(self, test_cases: List[TestCase], config: Dict[str, Any]) -> List[EvalResult]:
        """Run evaluations sequentially"""
        results = []
        
        for i, test_case in enumerate(test_cases):
            print(f"Running test {i+1}/{len(test_cases)}: {test_case.id}")
            
            try:
                result = self.evaluate_single_test(test_case, config)
                results.append(result)
            except Exception as e:
                print(f"Error in test {test_case.id}: {e}")
                results.append(EvalResult(
                    test_case_id=test_case.id,
                    prompt=test_case.prompt,
                    response="",
                    model="unknown",
                    metrics={},
                    metadata={},
                    execution_time=0,
                    success=False,
                    error=str(e)
                ))
        
        return results
    
    def run_parallel_evaluation(self, test_cases: List[TestCase], config: Dict[str, Any]) -> List[EvalResult]:
        """Run evaluations in parallel"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_test = {
                executor.submit(self.evaluate_single_test, test_case, config): test_case
                for test_case in test_cases
            }
            
            # Collect results
            completed = 0
            for future in as_completed(future_to_test, timeout=self.timeout * len(test_cases)):
                test_case = future_to_test[future]
                completed += 1
                
                try:
                    result = future.result()
                    results.append(result)
                    print(f"Completed {completed}/{len(test_cases)}: {test_case.id}")
                except Exception as e:
                    print(f"Error in test {test_case.id}: {e}")
                    results.append(EvalResult(
                        test_case_id=test_case.id,
                        prompt=test_case.prompt,
                        response="",
                        model="unknown",
                        metrics={},
                        metadata={},
                        execution_time=0,
                        success=False,
                        error=str(e)
                    ))
        
        return results
    
    def evaluate_single_test(self, test_case: TestCase, config: Dict[str, Any]) -> EvalResult:
        """Evaluate a single test case"""
        start_time = time.time()
        
        try:
            # Call the bridge function
            response, metadata = self.bridge_function(
                test_case.prompt,
                **(config or {})
            )
            
            # Calculate metrics
            metrics = {}
            
            # Accuracy (if expected response provided)
            if test_case.expected_response:
                metrics[EvalMetricType.ACCURACY.value] = self.metrics.calculate_accuracy(
                    test_case.expected_response, response
                )
            
            # Quality metrics
            metrics[EvalMetricType.RELEVANCE.value] = self.metrics.calculate_relevance(
                test_case.prompt, response
            )
            metrics[EvalMetricType.COHERENCE.value] = self.metrics.calculate_coherence(response)
            metrics[EvalMetricType.COMPLETENESS.value] = self.metrics.calculate_completeness(
                test_case.prompt, response
            )
            
            # Performance metrics
            execution_time = time.time() - start_time
            metrics[EvalMetricType.LATENCY.value] = execution_time
            
            # Cost metrics (if available in metadata)
            if 'cost' in metadata:
                metrics[EvalMetricType.COST.value] = metadata['cost']
            
            # Safety metrics (if guardrails data available)
            if 'guardrails_summary' in metadata:
                safety_score = metadata['guardrails_summary'].get('overall_confidence', 1.0)
                metrics[EvalMetricType.SAFETY.value] = safety_score
            
            # Quality score (composite)
            quality_components = [
                metrics.get(EvalMetricType.RELEVANCE.value, 0),
                metrics.get(EvalMetricType.COHERENCE.value, 0),
                metrics.get(EvalMetricType.COMPLETENESS.value, 0)
            ]
            metrics[EvalMetricType.QUALITY.value] = statistics.mean(quality_components)
            
            return EvalResult(
                test_case_id=test_case.id,
                prompt=test_case.prompt,
                response=response,
                model=metadata.get('model', 'unknown'),
                metrics=metrics,
                metadata=metadata,
                execution_time=execution_time,
                success=True
            )
        
        except Exception as e:
            execution_time = time.time() - start_time
            return EvalResult(
                test_case_id=test_case.id,
                prompt=test_case.prompt,
                response="",
                model="unknown",
                metrics={EvalMetricType.LATENCY.value: execution_time},
                metadata={'error_details': traceback.format_exc()},
                execution_time=execution_time,
                success=False,
                error=str(e)
            )
    
    def calculate_aggregate_metrics(self, results: List[EvalResult]) -> Dict[str, float]:
        """Calculate aggregate metrics across all results"""
        if not results:
            return {}
        
        successful_results = [r for r in results if r.success]
        success_rate = len(successful_results) / len(results)
        
        if not successful_results:
            return {'success_rate': success_rate}
        
        # Aggregate each metric type
        aggregates = {'success_rate': success_rate}
        
        metric_values = {}
        for result in successful_results:
            for metric_name, value in result.metrics.items():
                if metric_name not in metric_values:
                    metric_values[metric_name] = []
                metric_values[metric_name].append(value)
        
        # Calculate statistics for each metric
        for metric_name, values in metric_values.items():
            if values:
                aggregates[f'avg_{metric_name}'] = statistics.mean(values)
                aggregates[f'median_{metric_name}'] = statistics.median(values)
                aggregates[f'min_{metric_name}'] = min(values)
                aggregates[f'max_{metric_name}'] = max(values)
                if len(values) > 1:
                    aggregates[f'std_{metric_name}'] = statistics.stdev(values)
        
        return aggregates
    
    def analyze_performance(self, results: List[EvalResult]) -> Dict[str, Any]:
        """Analyze performance patterns"""
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            return {'error': 'No successful results to analyze'}
        
        analysis = {}
        
        # Latency analysis
        latencies = [r.metrics.get(EvalMetricType.LATENCY.value, 0) for r in successful_results]
        if latencies:
            analysis['latency'] = {
                'avg': statistics.mean(latencies),
                'p50': statistics.median(latencies),
                'p95': sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 20 else max(latencies),
                'fastest': min(latencies),
                'slowest': max(latencies)
            }
        
        # Model performance breakdown
        model_performance = {}
        for result in successful_results:
            model = result.model
            if model not in model_performance:
                model_performance[model] = {
                    'count': 0,
                    'avg_quality': 0,
                    'avg_latency': 0,
                    'success_rate': 0
                }
            
            model_performance[model]['count'] += 1
            model_performance[model]['avg_quality'] += result.metrics.get(EvalMetricType.QUALITY.value, 0)
            model_performance[model]['avg_latency'] += result.metrics.get(EvalMetricType.LATENCY.value, 0)
        
        # Finalize model stats
        for model, stats in model_performance.items():
            if stats['count'] > 0:
                stats['avg_quality'] /= stats['count']
                stats['avg_latency'] /= stats['count']
                stats['success_rate'] = stats['count'] / len([r for r in results if r.model == model])
        
        analysis['by_model'] = model_performance
        
        return analysis
    
    def analyze_costs(self, results: List[EvalResult]) -> Dict[str, Any]:
        """Analyze cost patterns"""
        successful_results = [r for r in results if r.success]
        
        costs = []
        for result in successful_results:
            cost = result.metrics.get(EvalMetricType.COST.value, 0)
            if cost > 0:
                costs.append(cost)
        
        analysis = {}
        
        if costs:
            analysis['total_cost'] = sum(costs)
            analysis['avg_cost_per_request'] = statistics.mean(costs)
            analysis['cost_range'] = {
                'min': min(costs),
                'max': max(costs),
                'median': statistics.median(costs)
            }
        
        return analysis
    
    def generate_recommendations(
        self,
        results: List[EvalResult],
        aggregate_metrics: Dict[str, float],
        performance_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Success rate recommendations
        success_rate = aggregate_metrics.get('success_rate', 0)
        if success_rate < 0.9:
            recommendations.append(
                f"Low success rate ({success_rate:.1%}). Consider implementing better error handling and input validation."
            )
        
        # Quality recommendations
        avg_quality = aggregate_metrics.get('avg_quality', 0)
        if avg_quality < 0.7:
            recommendations.append(
                f"Average quality score is {avg_quality:.2f}. Consider improving prompt engineering or model selection."
            )
        
        # Latency recommendations
        avg_latency = aggregate_metrics.get('avg_latency', 0)
        if avg_latency > 5.0:
            recommendations.append(
                f"High average latency ({avg_latency:.2f}s). Consider implementing caching or using faster models."
            )
        
        return recommendations
    
    def save_report(self, report: EvalReport):
        """Save evaluation report to file"""
        report_path = self.output_dir / f"{report.eval_id}_report.json"
        
        # Convert report to serializable format
        report_dict = {
            'eval_id': report.eval_id,
            'timestamp': report.timestamp,
            'config': report.config,
            'results': [
                {
                    'test_case_id': r.test_case_id,
                    'prompt': r.prompt,
                    'response': r.response,
                    'model': r.model,
                    'metrics': r.metrics,
                    'metadata': r.metadata,
                    'execution_time': r.execution_time,
                    'success': r.success,
                    'error': r.error
                }
                for r in report.results
            ],
            'aggregate_metrics': report.aggregate_metrics,
            'performance_analysis': report.performance_analysis,
            'cost_analysis': report.cost_analysis,
            'recommendations': report.recommendations
        }
        
        try:
            with open(report_path, 'w') as f:
                json.dump(report_dict, f, indent=2)
            print(f"Report saved to: {report_path}")
        except Exception as e:
            print(f"Error saving report: {e}")
    
    def get_evaluation_summary(self, limit: int = 10) -> Dict[str, Any]:
        """Get summary of recent evaluations"""
        recent_evals = self.eval_history[-limit:] if self.eval_history else []
        
        if not recent_evals:
            return {'message': 'No evaluations run yet'}
        
        return {
            'total_evaluations': len(self.eval_history),
            'recent_evaluations': recent_evals,
            'avg_success_rate': statistics.mean([e['success_rate'] for e in recent_evals]),
            'total_tests_run': sum([e['test_count'] for e in recent_evals])
        }


# CLI utilities for evaluation
def create_eval_config(
    use_router: bool = True,
    use_rag: bool = False,
    use_cache: bool = True,
    use_guardrails: bool = True,
    model: str = None,
    mock: bool = True
) -> Dict[str, Any]:
    """Create evaluation configuration"""
    return {
        'router': use_router,
        'rag': use_rag,
        'cache': use_cache,
        'guardrails': use_guardrails,
        'model': model,
        'mock': mock
    }


def run_quick_eval(bridge_function: Callable, output_dir: str = "eval_results") -> EvalReport:
    """Run a quick evaluation with default test suite"""
    harness = EvalHarness(bridge_function, output_dir)
    test_suite = TestSuiteLoader.create_default_suite()
    config = create_eval_config()
    
    return harness.run_evaluation(test_suite, config, "quick_eval")
