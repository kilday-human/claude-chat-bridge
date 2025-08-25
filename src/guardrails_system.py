"""
Guardrails System for Claude-GPT Bridge
Content safety, quality validation, bias detection, and response filtering
"""

import re
import json
import time
import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)

class GuardrailSeverity(Enum):
    """Severity levels for guardrail violations"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class GuardrailViolation:
    """Represents a guardrail violation"""
    rule_id: str
    rule_name: str
    severity: GuardrailSeverity
    message: str
    confidence: float
    matched_content: Optional[str] = None
    suggestion: Optional[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class GuardrailResult:
    """Result of guardrail evaluation"""
    passed: bool
    violations: List[GuardrailViolation]
    overall_score: float
    safety_score: float
    quality_score: float
    bias_score: float
    processing_time_ms: int
    metadata: Dict[str, Any] = None

class ContentSafetyGuard:
    """Content safety detection and filtering"""
    
    def __init__(self):
        # Harmful content patterns
        self.harmful_patterns = {
            'violence': [
                r'\b(kill|murder|assault|attack|violence|harm|hurt|injure)\b.*\b(people|person|human|someone)\b',
                r'\b(bomb|weapon|gun|knife|explosive)\b.*\b(make|build|create|construct)\b',
                r'\b(suicide|self-harm|cutting)\b.*\b(how to|instructions|guide)\b'
            ],
            'hate_speech': [
                r'\b(hate|despise|inferior)\b.*\b(race|ethnicity|religion|gender|sexual)\b',
                r'\b(terrorist|extremist)\b.*\b(group|organization|movement)\b',
                r'\b(discriminat|prejudic|racist|sexist|homophobic)\b'
            ],
            'adult_content': [
                r'\b(sexual|explicit|adult|pornographic)\b.*\b(content|material|images)\b',
                r'\b(nude|naked|sex|intercourse)\b.*\b(minors|children|kids|teens)\b'
            ],
            'illegal_activity': [
                r'\b(drugs|illegal|criminal|fraud|scam)\b.*\b(how to|guide|instructions|tutorial)\b',
                r'\b(hack|breach|exploit|unauthorized)\b.*\b(access|entry|system|network)\b',
                r'\b(money laundering|tax evasion|identity theft)\b'
            ]
        }
    
    def evaluate(self, text: str) -> List[GuardrailViolation]:
        """Evaluate text for content safety violations"""
        violations = []
        text_lower = text.lower()
        
        for category, patterns in self.harmful_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    violation = GuardrailViolation(
                        rule_id=f"safety_{category}",
                        rule_name=f"Content Safety - {category.title().replace('_', ' ')}",
                        severity=GuardrailSeverity.HIGH,
                        message=f"Potential {category.replace('_', ' ')} content detected",
                        confidence=0.8,
                        matched_content=match.group(),
                        suggestion="Consider rephrasing to avoid potentially harmful content"
                    )
                    violations.append(violation)
        
        return violations

class QualityGuard:
    """Response quality validation"""
    
    def __init__(self):
        # Quality assessment patterns
        self.quality_indicators = {
            'coherence': [
                r'\b(however|therefore|furthermore|moreover|additionally|consequently)\b',
                r'\b(first|second|third|finally|in conclusion)\b',
                r'\b(because|since|although|while|despite)\b'
            ],
            'completeness': [
                r'\b(comprehensive|detailed|thorough|complete|extensive)\b',
                r'\b(example|instance|specifically|particularly)\b',
                r'\b(include|contains|covers|addresses)\b'
            ],
            'clarity': [
                r'\b(clear|obvious|evident|apparent|straightforward)\b',
                r'\b(explain|clarify|demonstrate|illustrate)\b',
                r'\b(simple|easy|understand|comprehend)\b'
            ]
        }
        
        # Quality detractors
        self.quality_detractors = {
            'vague': [
                r'\b(maybe|perhaps|possibly|might|could be|seems like)\b.*\b(maybe|perhaps|possibly|might|could be|seems like)\b',
                r'\b(some|various|certain|several)\b.*\b(some|various|certain|several)\b',
                r'\b(thing|stuff|something|anything|everything)\b'
            ],
            'repetitive': [
                r'\b(\w+)\s+\1\b',  # Repeated words
                r'(\b\w+\b)(?:\s+\w+){0,5}\s+\1'  # Words repeated within 5 words
            ],
            'incomplete': [
                r'\b(I don\'t know|I\'m not sure|I can\'t say|unclear|uncertain)\b',
                r'\b(more information needed|need more context|depends on)\b',
                r'\.{3,}|…'  # Multiple ellipses indicating incompleteness
            ]
        }
    
    def evaluate(self, text: str, prompt: str = "") -> Tuple[float, List[GuardrailViolation]]:
        """Evaluate response quality and return score with violations"""
        violations = []
        
        # Basic quality metrics
        word_count = len(text.split())
        sentence_count = len(re.findall(r'[.!?]+', text))
        avg_sentence_length = word_count / max(sentence_count, 1)
        
        # Quality indicators score
        indicator_score = 0
        for category, patterns in self.quality_indicators.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                indicator_score += matches * 0.1
        
        # Quality detractors
        detractor_score = 0
        for category, patterns in self.quality_detractors.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    detractor_score += 0.2
                    
                    if category == 'vague' and len(list(re.finditer(pattern, text, re.IGNORECASE))) > 2:
                        violation = GuardrailViolation(
                            rule_id=f"quality_{category}",
                            rule_name=f"Quality Issue - {category.title()}",
                            severity=GuardrailSeverity.MEDIUM,
                            message=f"Response contains {category} language",
                            confidence=0.7,
                            matched_content=match.group(),
                            suggestion=f"Use more specific and concrete language"
                        )
                        violations.append(violation)
        
        # Length penalties/bonuses
        length_score = 0
        if word_count < 10:
            length_score = -0.5
            violations.append(GuardrailViolation(
                rule_id="quality_length_short",
                rule_name="Quality Issue - Too Short",
                severity=GuardrailSeverity.MEDIUM,
                message="Response is very short and may be incomplete",
                confidence=0.8,
                suggestion="Provide more detailed and comprehensive information"
            ))
        elif word_count > 500:
            length_score = -0.2
        elif 50 <= word_count <= 200:
            length_score = 0.1
        
        # Sentence structure score
        structure_score = 0
        if avg_sentence_length < 5:
            structure_score = -0.1
        elif avg_sentence_length > 30:
            structure_score = -0.1
        else:
            structure_score = 0.1
        
        # Calculate final quality score
        base_score = 0.5  # Neutral starting point
        final_score = base_score + indicator_score - detractor_score + length_score + structure_score
        final_score = max(0.0, min(1.0, final_score))
        
        return final_score, violations

class BiasDetectionGuard:
    """Bias detection in responses"""
    
    def __init__(self):
        # Bias detection patterns
        self.bias_patterns = {
            'gender': [
                r'\b(men|male|guy|boy)\b.*\b(better|superior|stronger|smarter)\b.*\b(women|female|girl|lady)\b',
                r'\b(women|female|girl|lady)\b.*\b(emotional|irrational|weak|inferior)\b',
                r'\b(typical|natural|normal)\b.*\b(male|female|man|woman)\b.*\b(behavior|trait|characteristic)\b'
            ],
            'racial': [
                r'\b(race|ethnic|cultural)\b.*\b(superior|inferior|better|worse|natural|typical)\b',
                r'\b(all|most|typically)\b.*\b(people|individuals|members)\b.*\b(from|of)\b.*\b(country|culture|ethnicity)\b.*\b(are|do|have)\b'
            ],
            'age': [
                r'\b(young|old|elderly|senior)\b.*\b(people|individuals|generation)\b.*\b(always|never|typically|usually)\b',
                r'\b(millennials|boomers|gen z|generation)\b.*\b(lazy|entitled|confused|outdated)\b'
            ],
            'socioeconomic': [
                r'\b(poor|rich|wealthy|low-income|high-income)\b.*\b(people|families|individuals)\b.*\b(always|never|typically|usually)\b.*\b(lazy|hardworking|smart|stupid)\b'
            ]
        }
    
    def evaluate(self, text: str) -> Tuple[float, List[GuardrailViolation]]:
        """Evaluate text for potential bias"""
        violations = []
        bias_score = 1.0  # Start with perfect score
        
        for category, patterns in self.bias_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    bias_score -= 0.2
                    violation = GuardrailViolation(
                        rule_id=f"bias_{category}",
                        rule_name=f"Potential Bias - {category.title()}",
                        severity=GuardrailSeverity.MEDIUM,
                        message=f"Potential {category} bias detected",
                        confidence=0.6,
                        matched_content=match.group(),
                        suggestion="Consider using more neutral and inclusive language"
                    )
                    violations.append(violation)
        
        # Check for absolute statements that might indicate bias
        absolute_patterns = [
            r'\b(all|every|never|always|none|everyone|nobody)\b.*\b(people|person|individual|group)\b'
        ]
        
        for pattern in absolute_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(violations) < 5:  # Limit violations
                    bias_score -= 0.1
                    violation = GuardrailViolation(
                        rule_id="bias_absolute",
                        rule_name="Potential Bias - Absolute Statement",
                        severity=GuardrailSeverity.LOW,
                        message="Absolute statements may indicate bias",
                        confidence=0.4,
                        matched_content=match.group(),
                        suggestion="Consider using more nuanced language with qualifiers"
                    )
                    violations.append(violation)
        
        bias_score = max(0.0, bias_score)
        return bias_score, violations

class FormatValidationGuard:
    """Response format and structure validation"""
    
    def evaluate(self, text: str, expected_format: Optional[str] = None) -> List[GuardrailViolation]:
        """Validate response format"""
        violations = []
        
        # Basic format checks
        if not text.strip():
            violations.append(GuardrailViolation(
                rule_id="format_empty",
                rule_name="Format Issue - Empty Response",
                severity=GuardrailSeverity.HIGH,
                message="Response is empty",
                confidence=1.0,
                suggestion="Provide a meaningful response"
            ))
        
        # Check for malformed encoding
        try:
            text.encode('utf-8')
        except UnicodeEncodeError:
            violations.append(GuardrailViolation(
                rule_id="format_encoding",
                rule_name="Format Issue - Encoding Error",
                severity=GuardrailSeverity.HIGH,
                message="Response contains invalid characters",
                confidence=1.0,
                suggestion="Ensure response uses valid UTF-8 encoding"
            ))
        
        # Check for excessive whitespace
        if re.search(r'\s{10,}', text):
            violations.append(GuardrailViolation(
                rule_id="format_whitespace",
                rule_name="Format Issue - Excessive Whitespace",
                severity=GuardrailSeverity.LOW,
                message="Response contains excessive whitespace",
                confidence=0.8,
                suggestion="Clean up formatting by removing excessive spaces"
            ))
        
        # Check for broken markdown if it looks like markdown
        if '```' in text and text.count('```') % 2 != 0:
            violations.append(GuardrailViolation(
                rule_id="format_markdown",
                rule_name="Format Issue - Broken Markdown",
                severity=GuardrailSeverity.MEDIUM,
                message="Markdown code blocks are not properly closed",
                confidence=0.9,
                suggestion="Ensure all code blocks are properly closed with matching backticks"
            ))
        
        return violations

class GuardrailsSystem:
    """Main guardrails orchestration system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize guardrail components
        self.content_safety = ContentSafetyGuard()
        self.quality_guard = QualityGuard()
        self.bias_detection = BiasDetectionGuard()
        self.format_validation = FormatValidationGuard()
        
        # Configuration settings
        self.enable_content_safety = config.get('enable_content_safety', True)
        self.enable_quality_check = config.get('enable_quality_check', True)
        self.enable_bias_detection = config.get('enable_bias_detection', True)
        self.enable_format_validation = config.get('enable_format_validation', True)
        
        # Thresholds
        self.min_quality_score = config.get('min_quality_score', 0.6)
        self.min_safety_score = config.get('min_safety_score', 0.8)
        self.min_bias_score = config.get('min_bias_score', 0.7)
        
        # Scoring weights
        self.quality_weight = config.get('quality_weight', 0.4)
        self.safety_weight = config.get('safety_weight', 0.3)
        self.bias_weight = config.get('bias_weight', 0.3)
        
        # Statistics
        self.stats = {
            'total_evaluations': 0,
            'passed_evaluations': 0,
            'failed_evaluations': 0,
            'safety_violations': 0,
            'quality_violations': 0,
            'bias_violations': 0,
            'format_violations': 0,
            'avg_processing_time_ms': 0,
            'total_processing_time_ms': 0
        }
    
    def evaluate_response(self, text: str, prompt: str = "", model: str = "") -> GuardrailResult:
        """Comprehensive response evaluation"""
        start_time = time.time()
        
        all_violations = []
        safety_score = 1.0
        quality_score = 0.8  # Default
        bias_score = 1.0
        
        # Content safety evaluation
        if self.enable_content_safety:
            safety_violations = self.content_safety.evaluate(text)
            all_violations.extend(safety_violations)
            
            # Calculate safety score based on violations
            if safety_violations:
                severity_weights = {
                    GuardrailSeverity.LOW: 0.1,
                    GuardrailSeverity.MEDIUM: 0.25,
                    GuardrailSeverity.HIGH: 0.5,
                    GuardrailSeverity.CRITICAL: 1.0
                }
                
                safety_penalty = sum(severity_weights[v.severity] for v in safety_violations)
                safety_score = max(0.0, 1.0 - safety_penalty)
                self.stats['safety_violations'] += len(safety_violations)
        
        # Quality evaluation
        if self.enable_quality_check:
            quality_score, quality_violations = self.quality_guard.evaluate(text, prompt)
            all_violations.extend(quality_violations)
            self.stats['quality_violations'] += len(quality_violations)
        
        # Bias detection
        if self.enable_bias_detection:
            bias_score, bias_violations = self.bias_detection.evaluate(text)
            all_violations.extend(bias_violations)
            self.stats['bias_violations'] += len(bias_violations)
        
        # Format validation
        if self.enable_format_validation:
            format_violations = self.format_validation.evaluate(text)
            all_violations.extend(format_violations)
            self.stats['format_violations'] += len(format_violations)
        
        # Calculate overall score
        overall_score = (
            quality_score * self.quality_weight +
            safety_score * self.safety_weight +
            bias_score * self.bias_weight
        )
        
        # Determine if response passes
        passes_safety = safety_score >= self.min_safety_score
        passes_quality = quality_score >= self.min_quality_score  
        passes_bias = bias_score >= self.min_bias_score
        passes_format = len([v for v in all_violations if v.rule_id.startswith('format') and v.severity in [GuardrailSeverity.HIGH, GuardrailSeverity.CRITICAL]]) == 0
        
        passed = passes_safety and passes_quality and passes_bias and passes_format
        
        # Processing time
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Update statistics
        self.stats['total_evaluations'] += 1
        if passed:
            self.stats['passed_evaluations'] += 1
        else:
            self.stats['failed_evaluations'] += 1
        
        self.stats['total_processing_time_ms'] += processing_time_ms
        self.stats['avg_processing_time_ms'] = self.stats['total_processing_time_ms'] / self.stats['total_evaluations']
        
        # Create result
        result = GuardrailResult(
            passed=passed,
            violations=all_violations,
            overall_score=overall_score,
            safety_score=safety_score,
            quality_score=quality_score,
            bias_score=bias_score,
            processing_time_ms=processing_time_ms,
            metadata={
                'model': model,
                'prompt_length': len(prompt),
                'response_length': len(text),
                'word_count': len(text.split()),
                'passes_safety': passes_safety,
                'passes_quality': passes_quality,
                'passes_bias': passes_bias,
                'passes_format': passes_format
            }
        )
        
        return result
    
    def evaluate_prompt(self, prompt: str) -> GuardrailResult:
        """Evaluate input prompt for safety and appropriateness"""
        start_time = time.time()
        
        violations = []
        safety_score = 1.0
        
        # Check prompt for safety issues
        if self.enable_content_safety:
            safety_violations = self.content_safety.evaluate(prompt)
            violations.extend(safety_violations)
            
            if safety_violations:
                severity_weights = {
                    GuardrailSeverity.LOW: 0.1,
                    GuardrailSeverity.MEDIUM: 0.25,
                    GuardrailSeverity.HIGH: 0.5,
                    GuardrailSeverity.CRITICAL: 1.0
                }
                
                safety_penalty = sum(severity_weights[v.severity] for v in safety_violations)
                safety_score = max(0.0, 1.0 - safety_penalty)
        
        # Check for prompt injection attempts
        injection_patterns = [
            r'ignore\s+previous\s+instructions',
            r'system\s*:\s*you\s+are',
            r'override\s+your\s+programming',
            r'jailbreak|roleplay\s+as|pretend\s+you\s+are',
            r'developer\s+mode|admin\s+mode|god\s+mode'
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                violations.append(GuardrailViolation(
                    rule_id="prompt_injection",
                    rule_name="Prompt Injection Attempt",
                    severity=GuardrailSeverity.HIGH,
                    message="Potential prompt injection or system manipulation attempt",
                    confidence=0.8,
                    suggestion="Use standard queries without attempting to modify system behavior"
                ))
                safety_score = min(safety_score, 0.3)
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        passed = safety_score >= self.min_safety_score and len([v for v in violations if v.severity in [GuardrailSeverity.HIGH, GuardrailSeverity.CRITICAL]]) == 0
        
        return GuardrailResult(
            passed=passed,
            violations=violations,
            overall_score=safety_score,
            safety_score=safety_score,
            quality_score=1.0,  # Not applicable for prompts
            bias_score=1.0,     # Not applicable for prompts
            processing_time_ms=processing_time_ms,
            metadata={
                'prompt_length': len(prompt),
                'word_count': len(prompt.split()),
                'evaluation_type': 'prompt'
            }
        )
    
    def get_violation_summary(self, violations: List[GuardrailViolation]) -> Dict[str, Any]:
        """Get summary statistics of violations"""
        if not violations:
            return {'total': 0, 'by_severity': {}, 'by_category': {}}
        
        by_severity = {}
        by_category = {}
        
        for violation in violations:
            # Count by severity
            severity_key = violation.severity.value
            by_severity[severity_key] = by_severity.get(severity_key, 0) + 1
            
            # Count by category (extracted from rule_id)
            category = violation.rule_id.split('_')[0] if '_' in violation.rule_id else 'other'
            by_category[category] = by_category.get(category, 0) + 1
        
        return {
            'total': len(violations),
            'by_severity': by_severity,
            'by_category': by_category,
            'critical_count': by_severity.get('critical', 0),
            'high_count': by_severity.get('high', 0)
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get guardrails system statistics"""
        pass_rate = (self.stats['passed_evaluations'] / self.stats['total_evaluations'] * 100) if self.stats['total_evaluations'] > 0 else 0
        
        return {
            **self.stats,
            'pass_rate_percent': round(pass_rate, 2),
            'config': {
                'content_safety_enabled': self.enable_content_safety,
                'quality_check_enabled': self.enable_quality_check,
                'bias_detection_enabled': self.enable_bias_detection,
                'format_validation_enabled': self.enable_format_validation,
                'min_quality_score': self.min_quality_score,
                'min_safety_score': self.min_safety_score,
                'min_bias_score': self.min_bias_score
            }
        }
    
    def update_thresholds(self, **kwargs):
        """Update guardrail thresholds"""
        if 'min_quality_score' in kwargs:
            self.min_quality_score = kwargs['min_quality_score']
        if 'min_safety_score' in kwargs:
            self.min_safety_score = kwargs['min_safety_score']
        if 'min_bias_score' in kwargs:
            self.min_bias_score = kwargs['min_bias_score']
        
        logger.info(f"Updated thresholds: quality={self.min_quality_score}, safety={self.min_safety_score}, bias={self.min_bias_score}")

def create_guardrails_system(config_path: str = "config/guardrails_config.json") -> GuardrailsSystem:
    """Factory function to create guardrails system with configuration"""
    
    # Default configuration
    default_config = {
        "enable_content_safety": True,
        "enable_quality_check": True,
        "enable_bias_detection": True,
        "enable_format_validation": True,
        "min_quality_score": 0.6,
        "min_safety_score": 0.8,
        "min_bias_score": 0.7,
        "quality_weight": 0.4,
        "safety_weight": 0.3,
        "bias_weight": 0.3
    }
    
    # Load configuration if file exists
    import os
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                # Merge with defaults
                default_config.update(config)
        except Exception as e:
            logger.error(f"Error loading guardrails config: {e}")
    
    return GuardrailsSystem(default_config)
