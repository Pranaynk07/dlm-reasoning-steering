"""
Contrastive Reasoning Feature Discovery.

NOVEL CONTRIBUTION: Identifies SAE features that are differentially 
activated during chain-of-thought reasoning vs. direct answer generation.

Methodology:
1. Collect SAE feature activations for CoT prompts (Set A)
2. Collect SAE feature activations for Direct prompts (Set B)
3. For each feature: compute differential activation = mean(A) - mean(B)
4. Statistical testing: Welch's t-test with Bonferroni correction
5. Rank features by effect size (Cohen's d)

Features with significantly higher activation during CoT are 
"reasoning-associated" — these are candidates for steering interventions.
"""

import torch
import numpy as np
from scipy import stats
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json
import logging
from tqdm import tqdm

from ..models.topk_sae import TopKSAE

logger = logging.getLogger(__name__)


class ContrastiveFeatureDiscovery:
    """
    Discovers reasoning-associated SAE features via contrastive analysis.
    
    Key insight: If a feature activates significantly more during 
    chain-of-thought reasoning than during direct answering, it likely 
    encodes reasoning-relevant information that we can amplify via steering.
    """
    
    def __init__(
        self,
        sae: TopKSAE,
        significance_alpha: float = 0.05,
        correction: str = "bonferroni",
        min_effect_size: float = 0.2,
    ):
        """
        Args:
            sae: Trained Top-K SAE
            significance_alpha: p-value threshold for significance
            correction: Multiple comparison correction method
            min_effect_size: Minimum Cohen's d for a feature to be considered
        """
        self.sae = sae
        self.alpha = significance_alpha
        self.correction = correction
        self.min_effect_size = min_effect_size
        
        # Results
        self.cot_activations = None
        self.direct_activations = None
        self.results = None
    
    def compute_feature_activations(
        self,
        activations: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode activations through SAE and get per-feature activation values.
        
        Args:
            activations: Raw model activations [n_tokens, d_model]
        
        Returns:
            Feature activations [n_tokens, d_dict]
        """
        self.sae.eval()
        device = next(self.sae.parameters()).device
        
        # Process in batches to avoid OOM
        batch_size = 512
        all_features = []
        
        for i in range(0, len(activations), batch_size):
            batch = activations[i:i + batch_size].to(device)
            with torch.no_grad():
                features = self.sae.encode(batch)
            all_features.append(features.cpu())
        
        return torch.cat(all_features, dim=0)
    
    def analyze(
        self,
        cot_activations: torch.Tensor,
        direct_activations: torch.Tensor,
    ) -> Dict:
        """
        Run full contrastive analysis to identify reasoning features.
        
        Args:
            cot_activations: Activations from CoT prompts [n_cot, d_model]
            direct_activations: Activations from Direct prompts [n_direct, d_model]
        
        Returns:
            Dict with analysis results including ranked features
        """
        logger.info(
            f"Running contrastive analysis: "
            f"CoT={cot_activations.shape[0]} tokens, "
            f"Direct={direct_activations.shape[0]} tokens"
        )
        
        # Encode through SAE
        logger.info("Encoding CoT activations through SAE...")
        cot_features = self.compute_feature_activations(cot_activations)
        
        logger.info("Encoding Direct activations through SAE...")
        direct_features = self.compute_feature_activations(direct_activations)
        
        self.cot_activations = cot_features
        self.direct_activations = direct_features
        
        n_features = cot_features.shape[1]
        
        # Per-feature analysis
        logger.info(f"Analyzing {n_features} features...")
        
        differential_scores = []
        p_values = []
        effect_sizes = []
        cot_means = []
        direct_means = []
        cot_stds = []
        direct_stds = []
        
        for f_idx in tqdm(range(n_features), desc="Feature analysis"):
            cot_f = cot_features[:, f_idx].numpy()
            direct_f = direct_features[:, f_idx].numpy()
            
            # Mean activations
            cot_mean = float(np.mean(cot_f))
            direct_mean = float(np.mean(direct_f))
            cot_std = float(np.std(cot_f)) + 1e-8
            direct_std = float(np.std(direct_f)) + 1e-8
            
            # Differential score
            diff = cot_mean - direct_mean
            
            # Welch's t-test (unequal variances)
            if cot_std > 1e-7 or direct_std > 1e-7:
                t_stat, p_val = stats.ttest_ind(cot_f, direct_f, equal_var=False)
                # One-sided: we care about CoT > Direct
                p_val_one_sided = p_val / 2 if t_stat > 0 else 1 - p_val / 2
            else:
                t_stat = 0.0
                p_val_one_sided = 1.0
            
            # Cohen's d (effect size)
            pooled_std = np.sqrt((cot_std**2 + direct_std**2) / 2)
            cohens_d = diff / (pooled_std + 1e-8)
            
            differential_scores.append(diff)
            p_values.append(p_val_one_sided)
            effect_sizes.append(cohens_d)
            cot_means.append(cot_mean)
            direct_means.append(direct_mean)
            cot_stds.append(cot_std)
            direct_stds.append(direct_std)
        
        # Convert to arrays
        differential_scores = np.array(differential_scores)
        p_values = np.array(p_values)
        effect_sizes = np.array(effect_sizes)
        
        # Multiple comparison correction
        if self.correction == "bonferroni":
            adjusted_alpha = self.alpha / n_features
            significant = p_values < adjusted_alpha
        elif self.correction == "fdr":
            # Benjamini-Hochberg FDR
            sorted_idx = np.argsort(p_values)
            n = len(p_values)
            thresholds = np.arange(1, n + 1) / n * self.alpha
            significant = np.zeros(n, dtype=bool)
            # Find largest k where p_(k) <= k/n * alpha
            sorted_p = p_values[sorted_idx]
            k = np.max(np.where(sorted_p <= thresholds)[0], initial=-1)
            if k >= 0:
                significant[sorted_idx[:k + 1]] = True
        else:
            significant = p_values < self.alpha
        
        # Additional filter: minimum effect size
        large_effect = np.abs(effect_sizes) >= self.min_effect_size
        reasoning_features = significant & large_effect & (differential_scores > 0)
        
        # Rank by effect size
        reasoning_indices = np.where(reasoning_features)[0]
        reasoning_ranked = reasoning_indices[
            np.argsort(effect_sizes[reasoning_indices])[::-1]
        ]
        
        # Also identify "anti-reasoning" features (significantly lower in CoT)
        anti_reasoning = significant & large_effect & (differential_scores < 0)
        anti_indices = np.where(anti_reasoning)[0]
        anti_ranked = anti_indices[
            np.argsort(effect_sizes[anti_indices])
        ]
        
        self.results = {
            'reasoning_features': reasoning_ranked.tolist(),
            'anti_reasoning_features': anti_ranked.tolist(),
            'n_significant': int(significant.sum()),
            'n_reasoning': int(reasoning_features.sum()),
            'n_anti_reasoning': int(anti_reasoning.sum()),
            'differential_scores': differential_scores.tolist(),
            'p_values': p_values.tolist(),
            'effect_sizes': effect_sizes.tolist(),
            'cot_means': cot_means,
            'direct_means': direct_means,
            'cot_stds': cot_stds,
            'direct_stds': direct_stds,
            'correction': self.correction,
            'alpha': self.alpha,
        }
        
        logger.info(
            f"Found {len(reasoning_ranked)} reasoning features, "
            f"{len(anti_ranked)} anti-reasoning features "
            f"(out of {n_features} total, {significant.sum()} significant)"
        )
        
        return self.results
    
    def get_top_reasoning_features(self, n: int = 50) -> List[int]:
        """Get top-N reasoning feature indices, ranked by effect size."""
        if self.results is None:
            raise ValueError("Run analyze() first")
        return self.results['reasoning_features'][:n]
    
    def get_feature_summary(self, feature_idx: int) -> Dict:
        """Get detailed stats for a specific feature."""
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
        """Save analysis results."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        with open(path / "contrastive_results.json", 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save feature activations for later visualization
        if self.cot_activations is not None:
            torch.save(self.cot_activations, path / "cot_features.pt")
            torch.save(self.direct_activations, path / "direct_features.pt")
    
    @classmethod
    def load_results(cls, path: str) -> Dict:
        """Load previously saved results."""
        with open(Path(path) / "contrastive_results.json") as f:
            return json.load(f)
