"""
Contrastive feature discovery: identifies SAE features differentially
activated during CoT reasoning vs direct answering.
"""

import torch
import numpy as np
from scipy import stats
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json
import logging
from tqdm import tqdm

try:
    from src.models.topk_sae import TopKSAE
except ImportError:
    from ..models.topk_sae import TopKSAE

logger = logging.getLogger(__name__)


class ContrastiveFeatureDiscovery:
    """
    Compares SAE activations between CoT and Direct prompting conditions.
    Uses Welch's t-test with multiple comparison correction to find
    features that encode reasoning-relevant information.
    """

    def __init__(self, sae: TopKSAE, significance_alpha=0.05,
                 correction="bonferroni", min_effect_size=0.2):
        self.sae = sae
        self.alpha = significance_alpha
        self.correction = correction
        self.min_effect_size = min_effect_size
        self.cot_activations = None
        self.direct_activations = None
        self.results = None

    def compute_feature_activations(self, activations: torch.Tensor) -> torch.Tensor:
        self.sae.eval()
        device = next(self.sae.parameters()).device
        all_features = []
        for i in range(0, len(activations), 512):
            batch = activations[i:i + 512].to(device)
            with torch.no_grad():
                features = self.sae.encode(batch)
            all_features.append(features.cpu())
        return torch.cat(all_features, dim=0)

    def analyze(self, cot_activations: torch.Tensor, direct_activations: torch.Tensor) -> Dict:
        logger.info(f"Contrastive analysis: CoT={cot_activations.shape[0]}, Direct={direct_activations.shape[0]} tokens")

        cot_features = self.compute_feature_activations(cot_activations)
        direct_features = self.compute_feature_activations(direct_activations)
        self.cot_activations = cot_features
        self.direct_activations = direct_features

        n_features = cot_features.shape[1]
        differential_scores, p_values, effect_sizes = [], [], []
        cot_means, direct_means, cot_stds, direct_stds = [], [], [], []

        for f_idx in tqdm(range(n_features), desc="Analyzing features"):
            cot_f = cot_features[:, f_idx].numpy()
            direct_f = direct_features[:, f_idx].numpy()

            cot_mean = float(np.mean(cot_f))
            direct_mean = float(np.mean(direct_f))
            cot_std = float(np.std(cot_f)) + 1e-8
            direct_std = float(np.std(direct_f)) + 1e-8

            diff = cot_mean - direct_mean

            if cot_std > 1e-7 or direct_std > 1e-7:
                t_stat, p_val = stats.ttest_ind(cot_f, direct_f, equal_var=False)
                p_val_one_sided = p_val / 2 if t_stat > 0 else 1 - p_val / 2
            else:
                p_val_one_sided = 1.0

            pooled_std = np.sqrt((cot_std**2 + direct_std**2) / 2)
            cohens_d = diff / (pooled_std + 1e-8)

            differential_scores.append(diff)
            p_values.append(p_val_one_sided)
            effect_sizes.append(cohens_d)
            cot_means.append(cot_mean)
            direct_means.append(direct_mean)
            cot_stds.append(cot_std)
            direct_stds.append(direct_std)

        differential_scores = np.array(differential_scores)
        p_values = np.array(p_values)
        effect_sizes = np.array(effect_sizes)

        # Multiple comparison correction
        if self.correction == "bonferroni":
            significant = p_values < (self.alpha / n_features)
        elif self.correction == "fdr":
            sorted_idx = np.argsort(p_values)
            n = len(p_values)
            thresholds = np.arange(1, n + 1) / n * self.alpha
            significant = np.zeros(n, dtype=bool)
            sorted_p = p_values[sorted_idx]
            k = np.max(np.where(sorted_p <= thresholds)[0], initial=-1)
            if k >= 0:
                significant[sorted_idx[:k + 1]] = True
        else:
            significant = p_values < self.alpha

        large_effect = np.abs(effect_sizes) >= self.min_effect_size
        reasoning_features = significant & large_effect & (differential_scores > 0)

        reasoning_indices = np.where(reasoning_features)[0]
        reasoning_ranked = reasoning_indices[np.argsort(effect_sizes[reasoning_indices])[::-1]]

        anti_reasoning = significant & large_effect & (differential_scores < 0)
        anti_indices = np.where(anti_reasoning)[0]
        anti_ranked = anti_indices[np.argsort(effect_sizes[anti_indices])]

        self.results = {
            'reasoning_features': reasoning_ranked.tolist(),
            'anti_reasoning_features': anti_ranked.tolist(),
            'n_significant': int(significant.sum()),
            'n_reasoning': int(reasoning_features.sum()),
            'n_anti_reasoning': int(anti_reasoning.sum()),
            'differential_scores': differential_scores.tolist(),
            'p_values': p_values.tolist(),
            'effect_sizes': effect_sizes.tolist(),
            'cot_means': cot_means, 'direct_means': direct_means,
            'cot_stds': cot_stds, 'direct_stds': direct_stds,
            'correction': self.correction, 'alpha': self.alpha,
        }

        logger.info(f"Found {len(reasoning_ranked)} reasoning, {len(anti_ranked)} anti-reasoning features")
        return self.results

    def get_top_reasoning_features(self, n: int = 50) -> List[int]:
        if self.results is None:
            raise ValueError("Run analyze() first")
        return self.results['reasoning_features'][:n]

    def get_feature_summary(self, feature_idx: int) -> Dict:
        if self.results is None:
            raise ValueError("Run analyze() first")
        return {
            'feature_idx': feature_idx,
            'differential_score': self.results['differential_scores'][feature_idx],
            'p_value': self.results['p_values'][feature_idx],
            'effect_size': self.results['effect_sizes'][feature_idx],
            'cot_mean': self.results['cot_means'][feature_idx],
            'direct_mean': self.results['direct_means'][feature_idx],
            'is_reasoning': feature_idx in self.results['reasoning_features'],
        }

    def save(self, path: str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "contrastive_results.json", 'w') as f:
            json.dump(self.results, f, indent=2)
        if self.cot_activations is not None:
            torch.save(self.cot_activations, path / "cot_features.pt")
            torch.save(self.direct_activations, path / "direct_features.pt")

    @classmethod
    def load_results(cls, path: str) -> Dict:
        with open(Path(path) / "contrastive_results.json") as f:
            return json.load(f)
