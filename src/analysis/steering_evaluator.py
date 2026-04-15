"""
Steering Evaluation for GSM8K experiments.

Metrics following DLM-Scope Section 4.2:
1. GSM8K Accuracy: correctness of extracted numerical answers
2. Concept improvement C(f): reasoning quality improvement  
3. Perplexity change P(f): fluency impact
4. Steering score S(f): combined metric

Additional reasoning-specific metrics:
5. CoT structure detection: does output contain step-by-step reasoning?
6. Reasoning length: verbosity of generated solutions
"""

import re
import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy import stats as scipy_stats
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# Reasoning structure indicators
REASONING_MARKERS = [
    r'step\s*\d', r'first[,.]', r'then[,.]', r'next[,.]', r'finally[,.]',
    r'therefore', r'because', r'since', r'so\s+', r'thus',
    r'let\s+me', r'we\s+(?:can|need|have|know|get)',
    r'\d+\s*[\+\-\*\/\×\÷]\s*\d+', r'=\s*\d+',
    r'total', r'remaining', r'each', r'per\s+',
]


def extract_numerical_answer(text: str) -> Optional[float]:
    """
    Extract numerical answer from generated text.
    
    Handles formats: "#### 42", "answer is 42", "= 42", last number
    """
    if not text:
        return None
    
    # Try "#### number" format
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    if match:
        try:
            return float(match.group(1).replace(',', ''))
        except ValueError:
            pass
    
    # Try "answer is X" or "= X"
    match = re.search(
        r'(?:answer\s+is|answer:|result:|=)\s*\$?(-?[\d,]+\.?\d*)', 
        text, re.IGNORECASE
    )
    if match:
        try:
            return float(match.group(1).replace(',', ''))
        except ValueError:
            pass
    
    # Last number in text
    numbers = re.findall(r'-?[\d,]+\.?\d*', text)
    if numbers:
        try:
            return float(numbers[-1].replace(',', ''))
        except ValueError:
            pass
    
    return None


def compute_accuracy(
    results: List[Dict], 
    tolerance: float = 1e-2,
) -> Dict:
    """
    Compute GSM8K accuracy from experiment results.
    
    Args:
        results: List of result dicts with 'generated' and 'answer' keys
        tolerance: Numerical tolerance for answer matching
    
    Returns:
        Dict with accuracy metrics
    """
    correct = 0
    total = 0
    parsed = 0
    
    for r in results:
        if r.get('answer') is None:
            continue
        
        total += 1
        generated_text = r.get('generated', r.get('text', ''))
        pred = extract_numerical_answer(generated_text)
        
        if pred is not None:
            parsed += 1
            if abs(pred - r['answer']) < tolerance:
                correct += 1
    
    accuracy = correct / total if total > 0 else 0
    parse_rate = parsed / total if total > 0 else 0
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'parsed': parsed,
        'parse_rate': parse_rate,
    }


def compute_reasoning_score(text: str) -> Dict:
    """
    Score how much chain-of-thought reasoning is in the generated text.
    
    Returns:
        Dict with reasoning metrics
    """
    if not text:
        return {'score': 0, 'n_markers': 0, 'n_equations': 0, 'length': 0}
    
    text_lower = text.lower()
    
    # Count reasoning markers
    n_markers = sum(
        len(re.findall(pattern, text_lower)) 
        for pattern in REASONING_MARKERS
    )
    
    # Count mathematical equations/operations
    n_equations = len(re.findall(r'\d+\s*[\+\-\*\/\×\÷=]\s*\d+', text))
    
    # Sentence count (rough proxy for step count)
    n_sentences = len(re.split(r'[.!?\n]', text))
    
    # Word count
    n_words = len(text.split())
    
    # Composite score (weighted)
    score = (
        min(n_markers / 5.0, 1.0) * 0.4 +     # Reasoning language
        min(n_equations / 3.0, 1.0) * 0.3 +     # Math operations
        min(n_sentences / 5.0, 1.0) * 0.2 +     # Multi-step
        min(n_words / 50.0, 1.0) * 0.1           # Verbosity
    )
    
    return {
        'score': float(score),
        'n_markers': n_markers,
        'n_equations': n_equations,
        'n_sentences': n_sentences,
        'n_words': n_words,
    }


def evaluate_experiment(
    baseline_results: List[Dict],
    steered_results: List[Dict],
    condition_name: str = "steered",
) -> Dict:
    """
    Full evaluation comparing baseline vs. steered generation.
    
    Computes:
    1. Accuracy comparison
    2. Reasoning quality change (concept improvement proxy)
    3. Statistical significance
    
    Args:
        baseline_results: Results without steering
        steered_results: Results with steering
        condition_name: Name for this comparison
    
    Returns:
        Comprehensive evaluation dict
    """
    # Accuracy
    baseline_acc = compute_accuracy(baseline_results)
    steered_acc = compute_accuracy(steered_results)
    
    # Reasoning scores
    baseline_reasoning = [
        compute_reasoning_score(r.get('generated', r.get('text', '')))
        for r in baseline_results
    ]
    steered_reasoning = [
        compute_reasoning_score(r.get('generated', r.get('text', '')))
        for r in steered_results
    ]
    
    baseline_scores = [r['score'] for r in baseline_reasoning]
    steered_scores = [r['score'] for r in steered_reasoning]
    
    # Statistical test on reasoning scores
    if len(baseline_scores) > 1 and len(steered_scores) > 1:
        t_stat, p_value = scipy_stats.ttest_ind(steered_scores, baseline_scores)
    else:
        t_stat, p_value = 0.0, 1.0
    
    # Compute concept improvement C(f) — DLM-Scope metric
    mean_baseline_reasoning = float(np.mean(baseline_scores))
    mean_steered_reasoning = float(np.mean(steered_scores))
    
    # Normalized concept improvement (bounded [-1, 1])
    concept_improvement = (
        (mean_steered_reasoning - mean_baseline_reasoning) / 
        (max(mean_baseline_reasoning, 0.01) + 1e-8)
    )
    concept_improvement = max(-1.0, min(1.0, concept_improvement))
    
    # Compute relative length change (proxy for perplexity/fluency)
    baseline_lengths = [r['n_words'] for r in baseline_reasoning]
    steered_lengths = [r['n_words'] for r in steered_reasoning]
    
    mean_baseline_len = float(np.mean(baseline_lengths)) if baseline_lengths else 1.0
    mean_steered_len = float(np.mean(steered_lengths)) if steered_lengths else 1.0
    
    # Steering score S(f) = C(f) + λ * P(f) (DLM-Scope Eq. 19)
    lambda_weight = 0.3
    perplexity_proxy = -abs(mean_steered_len - mean_baseline_len) / (mean_baseline_len + 1e-8)
    steering_score = concept_improvement + lambda_weight * perplexity_proxy
    
    evaluation = {
        'condition': condition_name,
        'baseline_accuracy': baseline_acc,
        'steered_accuracy': steered_acc,
        'accuracy_delta': steered_acc['accuracy'] - baseline_acc['accuracy'],
        'concept_improvement': concept_improvement,
        'steering_score': steering_score,
        'mean_reasoning_score_baseline': mean_baseline_reasoning,
        'mean_reasoning_score_steered': mean_steered_reasoning,
        'reasoning_t_stat': float(t_stat),
        'reasoning_p_value': float(p_value),
        'reasoning_significant': bool(p_value < 0.05),
        'mean_length_baseline': mean_baseline_len,
        'mean_length_steered': mean_steered_len,
    }
    
    return evaluation


def evaluate_alpha_sweep(
    baseline_results: List[Dict],
    alpha_results: Dict[float, List[Dict]],
) -> List[Dict]:
    """Evaluate results across alpha sweep."""
    evaluations = []
    for alpha, results in sorted(alpha_results.items()):
        eval_result = evaluate_experiment(
            baseline_results, results, f"alpha_{alpha}"
        )
        eval_result['alpha'] = alpha
        evaluations.append(eval_result)
    return evaluations


def save_evaluation(evaluation: Dict, path: str):
    """Save evaluation results as JSON."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "evaluation.json", 'w') as f:
        json.dump(evaluation, f, indent=2, default=str)


def print_evaluation_summary(evaluation: Dict):
    """Print a formatted evaluation summary."""
    print(f"\n{'='*60}")
    print(f"Evaluation: {evaluation['condition']}")
    print(f"{'='*60}")
    print(f"  Baseline accuracy: {evaluation['baseline_accuracy']['accuracy']:.1%}")
    print(f"  Steered accuracy:  {evaluation['steered_accuracy']['accuracy']:.1%}")
    print(f"  Accuracy delta:    {evaluation['accuracy_delta']:+.1%}")
    print(f"  Concept improvement (C): {evaluation['concept_improvement']:.3f}")
    print(f"  Steering score (S):      {evaluation['steering_score']:.3f}")
    print(f"  Reasoning score: {evaluation['mean_reasoning_score_baseline']:.3f} → {evaluation['mean_reasoning_score_steered']:.3f}")
    if evaluation.get('reasoning_significant'):
        print(f"  ✓ Statistically significant (p={evaluation['reasoning_p_value']:.4f})")
    else:
        print(f"  ✗ Not significant (p={evaluation['reasoning_p_value']:.4f})")
    print(f"{'='*60}\n")
