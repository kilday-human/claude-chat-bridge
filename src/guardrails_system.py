"""
Guardrails System for Claude-GPT Bridge
Provides content safety, quality validation, and response filtering
"""

import re
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
from pathlib import Path


class GuardrailViolationType(Enum):
    """Types of guardrail violations"""
    CONTENT_SAFETY = "content_safety"
    QUALITY_LOW = "quality_low"
    FACTUAL_ERROR = "factual_error"
    BIAS_DETECTED = "bias_detected"
    HALLUCINATION = "hallucination"
    FORMAT_INVALID = "format_invalid"
    LENGTH_VIOLATION = "length_violation"
    LANGUAGE_INAPPROPRIATE = "language_inappropriate"


@dataclass
class GuardrailResult:
    """Result of guardrail evaluation"""
    passed: bool
    confidence: float
    violations: List[GuardrailViolationType]
    messages: List[str]
    metadata: Dict[str, Any]
    processing_time: float


@dataclass
class GuardrailConfig:
    """Configuration for guardrail checks"""
    enable_content_safety: bool = True
    enable_quality_check: bool = True
    enable_bias_detection: bool = True
    enable_hallucination_check: bool = False  # Requires additional models
    enable_format_validation: bool = True
    
    # Thresholds
    min_quality_score: float = 0.6
    max_response_length: int = 4000
    min_response_length: int = 10
    
    # Content safety patterns
    safety_patterns: List[str] = None
    
    def __post_init__(self):
        if self.safety_patterns is None:
            self.safety_patterns = [
                r'\b(?:hate|violence|harm|illegal|dangerous)\b',
                r'\b(?:suicide|self-harm|death)\b',
                r'\b(?:racist|sexist|discriminat)\w*\b',
                r'\b(?:exploit|hack|crack|piracy)\b'
            ]


class BaseGuardrail:
    """Base class for all guardrails"""
    
    def __init__(self, name: str, config: GuardrailConfig):
        self.name = name
        self.config = config
    
    def check(self, prompt: str, response: str, metadata: Dict[str, Any] = None) -> GuardrailResult:
        """Check if response passes this guardrail"""
        raise NotImplementedError
    
    def _create_result(
        self,
        passed: bool,
        confidence: float,
        violations: List[GuardrailViolationType] = None,
        messages: List[str] = None,
        metadata: Dict[str, Any] = None,
        processing_time: float = 0.0
    ) -> GuardrailResult:
        """Helper to create GuardrailResult"""
        return GuardrailResult(
            passed=passed,
            confidence=confidence,
            violations=violations or [],
            messages=messages or [],
            metadata=metadata or {},
            processing_time=processing_time
        )


class ContentSafetyGuardrail(BaseGuardrail):
    """Content safety checking using pattern matching and heuristics"""
    
    def __init__(self, config: GuardrailConfig):
        super().__init__("content_safety", config)
        self.safety_patterns = [re.compile(pattern, re.IGNORECASE) 
                               for pattern in config.safety_patterns]
        
        # Additional safety checks
        self.harmful_keywords = {
            'violence': ['kill', 'murder', 'assault', 'attack', 'weapon', 'bomb'],
            'hate': ['racist', 'sexist', 'homophobic', 'hate speech'],
            'illegal': ['drugs', 'piracy', 'hacking', 'fraud', 'scam'],
            'self_harm': ['suicide', 'self-harm', 'cutting', 'overdose']
        }
    
    def check(self, prompt: str, response: str, metadata: Dict[str, Any] = None) -> GuardrailResult:
        start_time = time.time()
        
        violations = []
        messages = []
        safety_score = 1.0
        
        # Pattern-based checking
        for pattern in self.safety_patterns:
            matches = pattern.findall(response.lower())
            if matches:
                violations.append(GuardrailViolationType.CONTENT_SAFETY)
                messages.append(f"Unsafe content detected: {', '.join(matches)}")
                safety_score *= 0.5
        
        # Keyword-based checking with context
        for category, keywords in self.harmful_keywords.items():
            found_keywords = []
            for keyword in keywords:
                if keyword in response.lower():
                    found_keywords.append(keyword)
            
            if found_keywords:
                # Context check - reduce false positives
                if not self._is_educational_context(response, found_keywords):
                    violations.append(GuardrailViolationType.CONTENT_SAFETY)
                    messages.append(f"Potentially harmful {category} content: {', '.join(found_keywords)}")
                    safety_score *= 0.7
        
        # Length-based safety (extremely long responses might be problematic)
        if len(response) > self.config.max_response_length * 2:
            violations.append(GuardrailViolationType.LENGTH_VIOLATION)
            messages.append(f"Response extremely long ({len(response)} chars)")
            safety_score *= 0.8
        
        processing_time = time.time() - start_time
        passed = len(violations) == 0 and safety_score > 0.5
        
        return self._create_result(
            passed=passed,
            confidence=safety_score,
            violations=violations,
            messages=messages,
            metadata={'safety_score': safety_score, 'checks_performed': len(self.safety_patterns)},
            processing_time=processing_time
        )
    
    def _is_educational_context(self, response: str, keywords: List[str]) -> bool:
        """Check if harmful keywords appear in educational/informational context"""
        educational_indicators = [
            'education', 'information', 'awareness', 'prevention',
            'history', 'research', 'study', 'academic', 'scholarly',
            'definition', 'explanation', 'understand', 'learn'
        ]
        
        response_lower = response.lower()
        return any(indicator in response_lower for indicator in educational_indicators)


class QualityGuardrail(BaseGuardrail):
    """Response quality validation"""
    
    def __init__(self, config: GuardrailConfig):
        super().__init__("quality", config)
    
    def check(self, prompt: str, response: str, metadata: Dict[str, Any] = None) -> GuardrailResult:
        start_time = time.time()
        
        violations = []
        messages = []
        quality_metrics = {}
        
        # Length checks
        if len(response.strip()) < self.config.min_response_length:
            violations.append(GuardrailViolationType.QUALITY_LOW)
            messages.append(f"Response too short ({len(response)} chars)")
        
        if len(response) > self.config.max_response_length:
            violations.append(GuardrailViolationType.LENGTH_VIOLATION)
            messages.append(f"Response too long ({len(response)} chars)")
        
        # Content quality metrics
        quality_metrics.update(self._analyze_content_quality(response))
        
        # Coherence check
        coherence_score = self._check_coherence(response)
        quality_metrics['coherence'] = coherence_score
        
        # Relevance check (basic keyword overlap)
        relevance_score = self._check_relevance(prompt, response)
        quality_metrics['relevance'] = relevance_score
        
        # Overall quality score
        overall_quality = (
            quality_metrics['readability'] * 0.3 +
            coherence_score * 0.4 +
            relevance_score * 0.3
        )
        
        if overall_quality < self.config.min_quality_score:
            violations.append(GuardrailViolationType.QUALITY_LOW)
            messages.append(f"Low quality score: {overall_quality:.2f}")
        
        processing_time = time.time() - start_time
        passed = len(violations) == 0
        
        return self._create_result(
            passed=passed,
            confidence=overall_quality,
            violations=violations,
            messages=messages,
            metadata=quality_metrics,
            processing_time=processing_time
        )
    
    def _analyze_content_quality(self, response: str) -> Dict[str, float]:
        """Analyze various quality metrics"""
        words = response.split()
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not words or not sentences:
            return {'readability': 0.0, 'complexity': 0.0, 'structure': 0.0}
        
        # Basic readability (Flesch approximation)
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        readability = max(0, min(1, (206.835 - 1.015 * avg_sentence_length - 84.6 * (avg_word_length / 6)) / 100))
        
        # Structural quality
        structure_score = 1.0
        if len(sentences) == 1 and len(words) > 50:
            structure_score *= 0.8  # Penalize very long single sentences
        
        # Vocabulary diversity
        unique_words = len(set(word.lower() for word in words))
        diversity = unique_words / len(words) if words else 0
        
        return {
            'readability': readability,
            'complexity': min(1.0, avg_word_length / 8),
            'structure': structure_score,
            'diversity': diversity,
            'word_count': len(words),
            'sentence_count': len(sentences)
        }
    
    def _check_coherence(self, response: str) -> float:
        """Basic coherence checking"""
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return 1.0  # Single sentence is coherent by definition
        
        # Simple coherence heuristics
        coherence_score = 1.0
        
        # Check for topic consistency (repeated key terms)
        all_words = response.lower().split()
        word_freq = {}
        for word in all_words:
            if len(word) > 4:  # Focus on meaningful words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Coherent responses have some repeated key terms
        repeated_terms = sum(1 for count in word_freq.values() if count > 1)
        if repeated_terms < len(sentences) * 0.2:
            coherence_score *= 0.8
        
        return coherence_score
    
    def _check_relevance(self, prompt: str, response: str) -> float:
        """Basic relevance checking using keyword overlap"""
        prompt_words = set(word.lower().strip('.,!?') for word in prompt.split() if len(word) > 3)
        response_words = set(word.lower().strip('.,!?') for word in response.split() if len(word) > 3)
        
        if not prompt_words:
            return 1.0  # No specific terms to match
        
        overlap = len(prompt_words.intersection(response_words))
        relevance = overlap / len(prompt_words)
        
        return min(1.0, relevance + 0.3)  # Base relevance + bonus


class BiasDetectionGuardrail(BaseGuardrail):
    """Basic bias detection in responses"""
    
    def __init__(self, config: GuardrailConfig):
        super().__init__("bias_detection", config)
        
        # Bias indicator patterns
        self.bias_patterns = {
            'gender': [
                r'\b(?:all|most|typical) (?:men|women|boys|girls)\b',
                r'\b(?:men|women) (?:are|should|always|never)\b'
            ],
            'racial': [
                r'\b(?:all|most|typical) (?:black|white|asian|hispanic) people\b',
                r'\b(?:naturally|inherently) (?:good|bad) at\b'
            ],
            'age': [
                r'\b(?:old|young) people (?:are|can\'t|should)\b',
                r'\bmillennials|boomers (?:are|always)\b'
            ],
            'economic': [
                r'\b(?:poor|rich) people (?:are|always|never)\b',
                r'\bdeserve (?:poverty|wealth)\b'
            ]
        }
    
    def check(self, prompt: str, response: str, metadata: Dict[str, Any] = None) -> GuardrailResult:
        start_time = time.time()
        
        violations = []
        messages = []
        bias_indicators = {}
        
        for bias_type, patterns in self.bias_patterns.items():
            matches = []
            for pattern in patterns:
                found = re.findall(pattern, response, re.IGNORECASE)
                matches.extend(found)
            
            if matches:
                bias_indicators[bias_type] = matches
                violations.append(GuardrailViolationType.BIAS_DETECTED)
                messages.append(f"Potential {bias_type} bias detected: {', '.join(matches)}")
        
        # Calculate bias score (lower is better)
        bias_score = 1.0 - (len(violations) * 0.2)
        bias_score = max(0.0, bias_score)
        
        processing_time = time.time() - start_time
        passed = len(violations) == 0
        
        return self._create_result(
            passed=passed,
            confidence=bias_score,
            violations=violations,
            messages=messages,
            metadata={'bias_indicators': bias_indicators, 'bias_score': bias_score},
            processing_time=processing_time
        )


class FormatValidationGuardrail(BaseGuardrail):
    """Format and structure validation"""
    
    def __init__(self, config: GuardrailConfig):
        super().__init__("format_validation", config)
    
    def check(self, prompt: str, response: str, metadata: Dict[str, Any] = None) -> GuardrailResult:
        start_time = time.time()
        
        violations = []
        messages = []
        format_metrics = {}
        
        # Basic format checks
        if not response.strip():
            violations.append(GuardrailViolationType.FORMAT_INVALID)
            messages.append("Empty response")
        
        # Check for proper sentence structure
        sentences = re.split(r'[.!?]+', response)
        valid_sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]
        
        if len(valid_sentences) == 0 and len(response.strip()) > 10:
            violations.append(GuardrailViolationType.FORMAT_INVALID)
            messages.append("No proper sentence structure detected")
        
        # Check for encoding issues
        try:
            response.encode('utf-8')
        except UnicodeEncodeError:
            violations.append(GuardrailViolationType.FORMAT_INVALID)
            messages.append("Text encoding issues detected")
        
        # Check for extremely repetitive content
        repetition_score = self._check_repetition(response)
        format_metrics['repetition_score'] = repetition_score
        
        if repetition_score > 0.7:
            violations.append(GuardrailViolationType.FORMAT_INVALID)
            messages.append(f"High repetition detected: {repetition_score:.2f}")
        
        processing_time = time.time() - start_time
        passed = len(violations) == 0
        format_score = 1.0 - (len(violations) * 0.3)
        
        return self._create_result(
            passed=passed,
            confidence=max(0.0, format_score),
            violations=violations,
            messages=messages,
            metadata=format_metrics,
            processing_time=processing_time
        )
    
    def _check_repetition(self, response: str) -> float:
        """Check for repetitive content"""
        words = response.split()
        if len(words) < 10:
            return 0.0
        
        # Check for repeated phrases
        phrases = []
        for i in range(len(words) - 2):
            phrase = ' '.join(words[i:i+3])
            phrases.append(phrase)
        
        if not phrases:
            return 0.0
        
        phrase_counts = {}
        for phrase in phrases:
            phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
        
        # Calculate repetition ratio
        repeated_phrases = sum(count - 1 for count in phrase_counts.values() if count > 1)
        return repeated_phrases / len(phrases) if phrases else 0.0


class GuardrailsManager:
    """Main guardrails system managing multiple guardrails"""
    
    def __init__(self, config: GuardrailConfig = None):
        self.config = config or GuardrailConfig()
        self.guardrails: List[BaseGuardrail] = []
        self._initialize_guardrails()
        
        # Performance tracking
        self.stats = {
            'total_checks': 0,
            'violations_found': 0,
            'checks_by_type': {},
            'avg_processing_time': 0.0,
            'violation_history': []
        }
    
    def _initialize_guardrails(self):
        """Initialize enabled guardrails"""
        if self.config.enable_content_safety:
            self.guardrails.append(ContentSafetyGuardrail(self.config))
        
        if self.config.enable_quality_check:
            self.guardrails.append(QualityGuardrail(self.config))
        
        if self.config.enable_bias_detection:
            self.guardrails.append(BiasDetectionGuardrail(self.config))
        
        if self.config.enable_format_validation:
            self.guardrails.append(FormatValidationGuardrail(self.config))
    
    def check_response(
        self,
        prompt: str,
        response: str,
        metadata: Dict[str, Any] = None,
        required_guardrails: List[str] = None
    ) -> Dict[str, GuardrailResult]:
        """Run all enabled guardrails on response"""
        start_time = time.time()
        results = {}
        
        guardrails_to_run = self.guardrails
        if required_guardrails:
            guardrails_to_run = [
                g for g in self.guardrails 
                if g.name in required_guardrails
            ]
        
        for guardrail in guardrails_to_run:
            try:
                result = guardrail.check(prompt, response, metadata)
                results[guardrail.name] = result
                
                # Update stats
                self.stats['checks_by_type'][guardrail.name] = (
                    self.stats['checks_by_type'].get(guardrail.name, 0) + 1
                )
                
                if not result.passed:
                    self.stats['violations_found'] += 1
                    self.stats['violation_history'].append({
                        'timestamp': time.time(),
                        'guardrail': guardrail.name,
                        'violations': [v.value for v in result.violations],
                        'confidence': result.confidence
                    })
            
            except Exception as e:
                # Graceful degradation
                results[guardrail.name] = GuardrailResult(
                    passed=False,
                    confidence=0.0,
                    violations=[GuardrailViolationType.FORMAT_INVALID],
                    messages=[f"Guardrail error: {str(e)}"],
                    metadata={'error': str(e)},
                    processing_time=0.0
                )
        
        # Update overall stats
        total_time = time.time() - start_time
        self.stats['total_checks'] += 1
        self.stats['avg_processing_time'] = (
            (self.stats['avg_processing_time'] * (self.stats['total_checks'] - 1) + total_time) /
            self.stats['total_checks']
        )
        
        return results
    
    def is_response_safe(
        self,
        prompt: str,
        response: str,
        metadata: Dict[str, Any] = None,
        min_confidence: float = 0.7
    ) -> Tuple[bool, Dict[str, Any]]:
        """Quick safety check with summary"""
        results = self.check_response(prompt, response, metadata)
        
        # Aggregate results
        all_passed = all(result.passed for result in results.values())
        min_confidence_met = all(
            result.confidence >= min_confidence for result in results.values()
        )
        
        # Collect all violations
        all_violations = []
        all_messages = []
        confidence_scores = {}
        
        for name, result in results.items():
            all_violations.extend(result.violations)
            all_messages.extend(result.messages)
            confidence_scores[name] = result.confidence
        
        overall_safe = all_passed and min_confidence_met
        
        summary = {
            'safe': overall_safe,
            'violations': [v.value for v in set(all_violations)],
            'messages': all_messages,
            'confidence_scores': confidence_scores,
            'overall_confidence': min(confidence_scores.values()) if confidence_scores else 0.0,
            'guardrails_run': list(results.keys())
        }
        
        return overall_safe, summary
    
    def filter_response(
        self,
        prompt: str,
        response: str,
        metadata: Dict[str, Any] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Filter/modify response based on guardrail results"""
        is_safe, summary = self.is_response_safe(prompt, response, metadata)
        
        if is_safe:
            return response, summary
        
        # Apply filtering based on violation types
        filtered_response = response
        
        # Handle different violation types
        for violation_type in summary['violations']:
            if violation_type == GuardrailViolationType.CONTENT_SAFETY.value:
                filtered_response = self._apply_content_filter(filtered_response)
            elif violation_type == GuardrailViolationType.LENGTH_VIOLATION.value:
                filtered_response = self._apply_length_filter(filtered_response)
            elif violation_type == GuardrailViolationType.FORMAT_INVALID.value:
                filtered_response = self._apply_format_filter(filtered_response)
        
        # Re-check filtered response
        recheck_safe, recheck_summary = self.is_response_safe(prompt, filtered_response, metadata)
        
        summary.update({
            'filtered': True,
            'original_length': len(response),
            'filtered_length': len(filtered_response),
            'recheck_safe': recheck_safe,
            'recheck_summary': recheck_summary
        })
        
        return filtered_response, summary
    
    def _apply_content_filter(self, response: str) -> str:
        """Apply content safety filtering"""
        # Simple content filtering - remove problematic sentences
        sentences = re.split(r'[.!?]+', response)
        safe_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if sentence contains problematic content
            is_safe = True
            for pattern in self.config.safety_patterns:
                if re.search(pattern, sentence, re.IGNORECASE):
                    is_safe = False
                    break
            
            if is_safe:
                safe_sentences.append(sentence)
        
        filtered = '. '.join(safe_sentences)
        if filtered and not filtered.endswith('.'):
            filtered += '.'
        
        return filtered or "I cannot provide a response to that request."
    
    def _apply_length_filter(self, response: str) -> str:
        """Apply length filtering"""
        if len(response) <= self.config.max_response_length:
            return response
        
        # Truncate at sentence boundaries
        sentences = re.split(r'[.!?]+', response)
        truncated_sentences = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if current_length + len(sentence) + 2 <= self.config.max_response_length:
                truncated_sentences.append(sentence)
                current_length += len(sentence) + 2
            else:
                break
        
        truncated = '. '.join(truncated_sentences)
        if truncated and not truncated.endswith('.'):
            truncated += '.'
        
        return truncated
    
    def _apply_format_filter(self, response: str) -> str:
        """Apply format filtering"""
        # Basic format cleaning
        lines = response.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line and len(line) > 2:  # Filter out very short lines
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get guardrails performance statistics"""
        return {
            **self.stats,
            'violation_rate': (
                self.stats['violations_found'] / self.stats['total_checks']
                if self.stats['total_checks'] > 0 else 0.0
            ),
            'enabled_guardrails': [g.name for g in self.guardrails],
            'recent_violations': self.stats['violation_history'][-10:]  # Last 10
        }
    
    def update_config(self, new_config: GuardrailConfig):
        """Update configuration and reinitialize guardrails"""
        self.config = new_config
        self.guardrails = []
        self._initialize_guardrails()
    
    def add_custom_guardrail(self, guardrail: BaseGuardrail):
        """Add custom guardrail"""
        self.guardrails.append(guardrail)


# Global guardrails manager instance
_guardrails_manager = None


def get_guardrails_manager(config: GuardrailConfig = None) -> GuardrailsManager:
    """Get or create global guardrails manager"""
    global _guardrails_manager
    if _guardrails_manager is None:
        _guardrails_manager = GuardrailsManager(config)
    return _guardrails_manager


def validate_response(func):
    """Decorator for automatic response validation"""
    def wrapper(*args, **kwargs):
        guardrails = get_guardrails_manager()
        
        # Get original response
        response, metadata = func(*args, **kwargs)
        
        # Extract prompt
        prompt = args[0] if args else kwargs.get('prompt', '')
        
        # Validate response
        is_safe, summary = guardrails.is_response_safe(prompt, response, metadata)
        
        # Add validation metadata
        enhanced_metadata = {
            **metadata,
            'guardrails_passed': is_safe,
            'guardrails_summary': summary,
            'validation_time': sum(
                r.processing_time for r in guardrails.check_response(prompt, response, metadata).values()
            )
        }
        
        return response, enhanced_metadata
    
    return wrapper


def filter_response(func):
    """Decorator for automatic response filtering"""
    def wrapper(*args, **kwargs):
        guardrails = get_guardrails_manager()
        
        # Get original response
        response, metadata = func(*args, **kwargs)
        
        # Extract prompt
        prompt = args[0] if args else kwargs.get('prompt', '')
        
        # Filter response
        filtered_response, filter_summary = guardrails.filter_response(prompt, response, metadata)
        
        # Add filtering metadata
        enhanced_metadata = {
            **metadata,
            'guardrails_filtered': filter_summary.get('filtered', False),
            'guardrails_summary': filter_summary
        }
        
        return filtered_response, enhanced_metadata
    
    return wrapper
