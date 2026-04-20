"""Steering evaluation metrics for GSM8K experiments."""

import re
import numpy as np
from typing import List, Dict, Optional
from scipy import stats as scipy_stats
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

REASONING_MARKERS = [
    r'step\s*\d', r'first[,.]', r'then[,.]', r'next[,.]', r'finally[,.]',
    r'therefore', r'because', r'since', r'so\s+', r'thus',
    r'let\s+me', r'we\s+(?:can|need|have|know|get)',
    r'\d+\s*[\+\-\*\/\×\÷]\s*\d+', r'=\s*\d+',
    r'total', r'remaining', r'each', r'per\s+',
]


def extract_numerical_answer(text: str) -> Optional[float]:
    if not text:
        return None
    for pattern, group in [
        (r'####\s*(-?[\d,]+\.?\d*)', 1),
        (r'(?:answer\s+is|answer:|result:|=)\s*\$?(-?[\d,]+\.?\d*)', 1),
    ]:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(group).replace(',', ''))
            except ValueError:
                continue

    numbers = re.findall(r'-?[\d,]+\.?\d*', text)
    if numbers:
        try:
            return float(numbers[-1].replace(',', ''))
        except ValueError:
            pass
    return None


def compute_accuracy(results: List[Dict], tolerance: float = 1e-2) -> Dict:
    correct, total, parsed = 0, 0, 0
    for r in results:
        if r.get('answer') is None:
            continue
        total += 1
        pred = extract_numerical_answer(r.get('generated', r.get('text', '')))
        if pred is not None:
            parsed += 1
            if abs(pred - r['answer']) < tolerance:
                correct += 1
    return {
        'accuracy': correct / total if total > 0 else 0,
        'correct': correct, 'total': total,
        'parsed': parsed, 'parse_rate': parsed / total if total > 0 else 0,
    }


def compute_reasoning_score(text: str) -> Dict:
    if not text:
        return {'score': 0, 'n_markers': 0, 'n_equations': 0, 'length': 0}
    text_lower = text.lower()
    n_markers = sum(len(re.findall(p, text_lower)) for p in REASONING_MARKERS)
    n_equations = len(re.findall(r'\d+\s*[\+\-\*\/\×\÷=]\s*\d+', text))
    n_sentences = len(re.split(r'[.!?\n]', text))
    n_words = len(text.split())

    score = (
        min(n_markers / 5.0, 1.0) * 0.4 +
        min(n_equations / 3.0, 1.0) * 0.3 +
        min(n_sentences / 5.0, 1.0) * 0.2 +
        min(n_words / 50.0, 1.0) * 0.1
    )
    return {'score': float(score), 'n_markers': n_markers,
            'n_equations': n_equations, 'n_sentences': n_sentences, 'n_words': n_words}


def evaluate_experiment(baseline_results, steered_results, condition_name="steered") -> Dict:
    baseline_acc = compute_accuracy(baseline_results)
    steered_acc = compute_accuracy(steered_results)

    baseline_reasoning = [compute_reasoning_score(r.get('generated', r.get('text', ''))) for r in baseline_results]
    steered_reasoning = [compute_reasoning_score(r.get('generated', r.get('text', ''))) for r in steered_results]

    baseline_scores = [r['score'] for r in baseline_reasoning]
    steered_scores = [r['score'] for r in steered_reasoning]

    if len(baseline_scores) > 1 and len(steered_scores) > 1:
        t_stat, p_value = scipy_stats.ttest_ind(steered_scores, baseline_scores)
    else:
        t_stat, p_value = 0.0, 1.0

    mean_b = float(np.mean(baseline_scores))
    mean_s = float(np.mean(steered_scores))

    concept_improvement = max(-1.0, min(1.0,
        (mean_s - mean_b) / (max(mean_b, 0.01) + 1e-8)
    ))

    baseline_lens = [r['n_words'] for r in baseline_reasoning]
    steered_lens = [r['n_words'] for r in steered_reasoning]
    mean_bl = float(np.mean(baseline_lens)) if baseline_lens else 1.0
    mean_sl = float(np.mean(steered_lens)) if steered_lens else 1.0

    perplexity_proxy = -abs(mean_sl - mean_bl) / (mean_bl + 1e-8)
    steering_score = concept_improvement + 0.3 * perplexity_proxy

    return {
        'condition': condition_name,
        'baseline_accuracy': baseline_acc, 'steered_accuracy': steered_acc,
        'accuracy_delta': steered_acc['accuracy'] - baseline_acc['accuracy'],
        'concept_improvement': concept_improvement,
        'steering_score': steering_score,
        'mean_reasoning_score_baseline': mean_b,
        'mean_reasoning_score_steered': mean_s,
        'reasoning_t_stat': float(t_stat), 'reasoning_p_value': float(p_value),
        'reasoning_significant': bool(p_value < 0.05),
        'mean_length_baseline': mean_bl, 'mean_length_steered': mean_sl,
    }


def evaluate_alpha_sweep(baseline_results, alpha_results) -> List[Dict]:
    return [
        {**evaluate_experiment(baseline_results, results, f"alpha_{alpha}"), 'alpha': alpha}
        for alpha, results in sorted(alpha_results.items())
    ]


def save_evaluation(evaluation, path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "evaluation.json", 'w') as f:
        json.dump(evaluation, f, indent=2, default=str)


def print_evaluation_summary(ev):
    print(f"\n{'='*50}")
    print(f"  {ev['condition']}")
    print(f"  Baseline acc: {ev['baseline_accuracy']['accuracy']:.1%}")
    print(f"  Steered acc:  {ev['steered_accuracy']['accuracy']:.1%} ({ev['accuracy_delta']:+.1%})")
    print(f"  Reasoning: {ev['mean_reasoning_score_baseline']:.3f} -> {ev['mean_reasoning_score_steered']:.3f}")
    print(f"  Steering score: {ev['steering_score']:.3f}")
    sig = "significant" if ev.get('reasoning_significant') else "not significant"
    print(f"  p={ev['reasoning_p_value']:.4f} ({sig})")
    print(f"{'='*50}")
