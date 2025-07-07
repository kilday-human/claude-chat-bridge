# metrics.py
from typing import Dict
import re

DISALLOWED = {"hate", "kill", "bomb"}  # simple example

def keyword_coverage(output: str, keywords: set) -> float:
    hits = sum(1 for kw in keywords if re.search(rf"\b{re.escape(kw)}\b", output, re.IGNORECASE))
    return hits / len(keywords) if keywords else 1.0

def length_parity(a: str, b: str) -> float:
    la, lb = len(a.split()), len(b.split())
    return min(la, lb) / max(la, lb) if max(la, lb) > 0 else 1.0

def safety_check(output: str) -> bool:
    return not any(word in output.lower() for word in DISALLOWED)

def compare_responses(resp_a: str, resp_b: str, keywords: set) -> Dict:
    return {
        "coverage_a": keyword_coverage(resp_a, keywords),
        "coverage_b": keyword_coverage(resp_b, keywords),
        "length_parity": length_parity(resp_a, resp_b),
        "safe_a": safety_check(resp_a),
        "safe_b": safety_check(resp_b),
    }

