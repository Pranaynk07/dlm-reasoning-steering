"""
Diffusion-time steering via SAE feature injection.
At each denoising step, injects learned feature directions into the residual stream.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Dict, Callable
import logging

try:
    from src.models.topk_sae import TopKSAE
except ImportError:
    from ..models.topk_sae import TopKSAE

logger = logging.getLogger(__name__)


class DiffusionSteerer:
    """
    Steers generation by adding SAE decoder directions to hidden states
    during denoising. Follows Eq. 13 from DLM-Scope: X[s] += alpha * m_f * v_f
    """

    def __init__(self, sae, reasoning_features, target_layer, alpha=2.0,
                 token_scope="all", normalize=True):
        self.sae = sae
        self.reasoning_features = reasoning_features
        self.target_layer = target_layer
        self.alpha = alpha
        self.token_scope = token_scope
        self.normalize = normalize
        self.steering_direction = self._compute_steering_direction()
        self.step_history = []

    def _compute_steering_direction(self):
        device = next(self.sae.parameters()).device
        directions = self.sae.get_feature_directions(self.reasoning_features)
        aggregate = directions.sum(dim=1)
        if self.normalize:
            aggregate = aggregate / (aggregate.norm() + 1e-8)
        return aggregate.to(device)

    def create_steering_hook(self):
        def hook(hidden_states, mask):
            device = hidden_states.device
            steering_vec = self.steering_direction.to(device)

            with torch.no_grad():
                features = self.sae.encode(hidden_states.float())
                target_acts = features[:, :, self.reasoning_features]
                m_f = max(target_acts.abs().mean().item(), 1.0)

            if self.token_scope == "all":
                intervention = self.alpha * m_f * steering_vec
                hidden_states = hidden_states + intervention.unsqueeze(0).unsqueeze(0)
            elif self.token_scope == "update":
                intervention = self.alpha * m_f * steering_vec
                mask_expanded = mask.unsqueeze(-1).float().to(device)
                hidden_states = hidden_states + intervention * mask_expanded

            self.step_history.append({
                'm_f': m_f,
                'norm': (self.alpha * m_f * steering_vec).norm().item(),
            })
            return hidden_states

        return hook

    def steer_generation(self, dlm_wrapper, prompt, max_new_tokens=128,
                         num_steps=30, temperature=1.0):
        self.step_history = []
        hook = self.create_steering_hook()
        output = dlm_wrapper.generate(
            prompt=prompt, max_new_tokens=max_new_tokens, num_steps=num_steps,
            temperature=temperature, steering_hook=hook, steering_layer=self.target_layer,
        )
        return {
            'text': output, 'prompt': prompt, 'alpha': self.alpha,
            'target_layer': self.target_layer, 'n_features': len(self.reasoning_features),
            'token_scope': self.token_scope, 'n_steps': num_steps,
            'step_history': self.step_history,
        }

    def update_alpha(self, new_alpha):
        self.alpha = new_alpha

    def update_features(self, new_features):
        self.reasoning_features = new_features
        self.steering_direction = self._compute_steering_direction()


class SteeringExperiment:
    """Runs baseline, steered, random-control, and alpha-sweep experiments."""

    def __init__(self, dlm_wrapper, sae, reasoning_features, target_layer):
        self.dlm = dlm_wrapper
        self.sae = sae
        self.reasoning_features = reasoning_features
        self.target_layer = target_layer

    def run_baseline(self, prompts, **kwargs):
        results = []
        for p in prompts:
            text = self.dlm.generate(prompt=p['prompt'], **kwargs)
            results.append({
                'prompt': p['prompt'], 'generated': text,
                'answer': p.get('answer'), 'condition': 'baseline',
            })
        return results

    def run_steered(self, prompts, alpha=2.0, features=None,
                    condition_name="steered", **kwargs):
        features = features or self.reasoning_features
        steerer = DiffusionSteerer(
            sae=self.sae, reasoning_features=features,
            target_layer=self.target_layer, alpha=alpha,
        )
        results = []
        for p in prompts:
            result = steerer.steer_generation(self.dlm, p['prompt'], **kwargs)
            result['answer'] = p.get('answer')
            result['condition'] = condition_name
            results.append(result)
        return results

    def run_random_control(self, prompts, n_features=50, n_random_sets=5,
                           alpha=2.0, **kwargs):
        import random
        all_results = []
        for seed in range(n_random_sets):
            random.seed(seed)
            rand_feats = random.sample(range(self.sae.d_dict), n_features)
            results = self.run_steered(
                prompts, alpha=alpha, features=rand_feats,
                condition_name=f"random_{seed}", **kwargs,
            )
            all_results.extend(results)
        return all_results

    def run_alpha_sweep(self, prompts, alpha_values=None, **kwargs):
        if alpha_values is None:
            alpha_values = [0.5, 1.0, 2.0, 5.0, 10.0]
        results = {}
        for alpha in alpha_values:
            logger.info(f"Alpha sweep: {alpha}")
            results[alpha] = self.run_steered(
                prompts, alpha=alpha, condition_name=f"alpha_{alpha}", **kwargs,
            )
        return results
