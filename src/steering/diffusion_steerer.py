"""
Diffusion-Time Steering via SAE Feature Intervention.

Implements the core steering mechanism from DLM-Scope Section 4,
extended for reasoning-specific feature intervention.

Key equations (from DLM-Scope):
  Eq. 13: X_{l,k}[s_k] += α * m_f * v_f  (per-step steering)
  Eq. 14: s_k position selection (all-tokens vs update-tokens)

Our extension: Instead of steering toward topics/styles, we steer
toward chain-of-thought reasoning by amplifying reasoning-associated
SAE features identified via contrastive analysis.
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
    Steers DLM generation toward reasoning by injecting SAE feature 
    directions during the denoising process.
    
    Following DLM-Scope Eq. 13:
    At each denoising step k, for target layer l:
        X_{l,k}[s_k] += α * m_f * v_f
    
    where:
        v_f = decoder direction for feature f
        m_f = per-sample scale (mean activation magnitude)
        α = steering strength hyperparameter
        s_k = position selector (all or masked-only)
    """
    
    def __init__(
        self,
        sae: TopKSAE,
        reasoning_features: List[int],
        target_layer: int,
        alpha: float = 2.0,
        token_scope: str = "all",
        normalize: bool = True,
    ):
        """
        Args:
            sae: Trained Top-K SAE
            reasoning_features: List of feature indices to amplify
            target_layer: Layer index for intervention
            alpha: Steering strength (positive = amplify reasoning)
            token_scope: "all" (all positions) or "update" (masked only)
            normalize: Whether to normalize steering vector
        """
        self.sae = sae
        self.reasoning_features = reasoning_features
        self.target_layer = target_layer
        self.alpha = alpha
        self.token_scope = token_scope
        self.normalize = normalize
        
        # Pre-compute the aggregate steering direction
        # This is the sum of decoder atoms for all reasoning features
        self.steering_direction = self._compute_steering_direction()
        
        # Per-step tracking for analysis
        self.step_history = []
    
    def _compute_steering_direction(self) -> torch.Tensor:
        """
        Compute the aggregate steering direction from reasoning features.
        
        For multiple features, we sum their decoder directions (v_f),
        following DLM-Scope's approach of single-feature steering
        extended to multi-feature intervention.
        
        Returns:
            Steering vector [d_model]
        """
        device = next(self.sae.parameters()).device
        
        directions = self.sae.get_feature_directions(self.reasoning_features)
        # directions shape: [d_model, n_features]
        
        # Sum and optionally normalize
        aggregate = directions.sum(dim=1)  # [d_model]
        
        if self.normalize:
            aggregate = aggregate / (aggregate.norm() + 1e-8)
        
        return aggregate.to(device)
    
    def create_steering_hook(self) -> Callable:
        """
        Create a hook function for use with DLMWrapper.denoising_loop.
        
        Returns a callable that modifies hidden states during denoising.
        """
        def steering_hook(hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            """
            Apply SAE feature steering to hidden states.
            
            Args:
                hidden_states: [batch, seq_len, d_model]
                mask: [batch, seq_len] boolean (True = masked)
            
            Returns:
                Modified hidden states
            """
            device = hidden_states.device
            steering_vec = self.steering_direction.to(device)
            
            # Compute per-sample scale m_f (DLM-Scope Eq. 4)
            # m_f = mean activation magnitude across the sequence
            with torch.no_grad():
                features = self.sae.encode(hidden_states.float())
                # Average activation of our target features
                target_acts = features[:, :, self.reasoning_features]
                m_f = target_acts.abs().mean()
                m_f = max(m_f.item(), 1.0)  # Floor at 1.0 to ensure steering effect
            
            # Create position selector s_k (DLM-Scope Eq. 14)
            if self.token_scope == "all":
                # Steer all positions
                intervention = self.alpha * m_f * steering_vec
                hidden_states = hidden_states + intervention.unsqueeze(0).unsqueeze(0)
            elif self.token_scope == "update":
                # Steer only masked (to-be-updated) positions
                intervention = self.alpha * m_f * steering_vec  # [d_model]
                mask_expanded = mask.unsqueeze(-1).float().to(device)  # [batch, seq, 1]
                hidden_states = hidden_states + intervention * mask_expanded
            else:
                raise ValueError(f"Unknown token_scope: {self.token_scope}")
            
            # Track for analysis
            self.step_history.append({
                'm_f': m_f,
                'intervention_norm': (self.alpha * m_f * steering_vec).norm().item(),
            })
            
            return hidden_states
        
        return steering_hook
    
    def steer_generation(
        self,
        dlm_wrapper,
        prompt: str,
        max_new_tokens: int = 128,
        num_steps: int = 30,
        temperature: float = 1.0,
    ) -> Dict:
        """
        Generate text with reasoning steering applied.
        
        Args:
            dlm_wrapper: Initialized DLMWrapper
            prompt: Input prompt
            max_new_tokens: Tokens to generate
            num_steps: Denoising steps
            temperature: Sampling temperature
        
        Returns:
            Dict with generated text and steering metadata
        """
        self.step_history = []  # Reset tracking
        
        # Create steering hook
        hook = self.create_steering_hook()
        
        # Generate with steering
        output_text = dlm_wrapper.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            num_steps=num_steps,
            temperature=temperature,
            steering_hook=hook,
            steering_layer=self.target_layer,
        )
        
        return {
            'text': output_text,
            'prompt': prompt,
            'alpha': self.alpha,
            'target_layer': self.target_layer,
            'n_features': len(self.reasoning_features),
            'token_scope': self.token_scope,
            'n_steps': num_steps,
            'step_history': self.step_history,
        }
    
    def update_alpha(self, new_alpha: float):
        """Update steering strength."""
        self.alpha = new_alpha
    
    def update_features(self, new_features: List[int]):
        """Update the set of reasoning features."""
        self.reasoning_features = new_features
        self.steering_direction = self._compute_steering_direction()


class SteeringExperiment:
    """
    Runs a full steering experiment with multiple configurations.
    
    Manages the experimental matrix:
    - Baseline (no steering)
    - Positive steering (amplify reasoning)
    - Negative steering (suppress reasoning)
    - Random feature control
    - Ablations (alpha sweep, layer sweep, feature count sweep)
    """
    
    def __init__(
        self,
        dlm_wrapper,
        sae: TopKSAE,
        reasoning_features: List[int],
        target_layer: int,
    ):
        self.dlm = dlm_wrapper
        self.sae = sae
        self.reasoning_features = reasoning_features
        self.target_layer = target_layer
    
    def run_baseline(self, prompts: List[Dict], **kwargs) -> List[Dict]:
        """Generate without steering (baseline)."""
        results = []
        for p in prompts:
            text = self.dlm.generate(prompt=p['prompt'], **kwargs)
            results.append({
                'prompt': p['prompt'],
                'generated': text,
                'answer': p.get('answer'),
                'condition': 'baseline',
            })
        return results
    
    def run_steered(
        self, 
        prompts: List[Dict], 
        alpha: float = 2.0,
        features: Optional[List[int]] = None,
        condition_name: str = "steered",
        **kwargs,
    ) -> List[Dict]:
        """Generate with steering."""
        features = features or self.reasoning_features
        steerer = DiffusionSteerer(
            sae=self.sae,
            reasoning_features=features,
            target_layer=self.target_layer,
            alpha=alpha,
        )
        
        results = []
        for p in prompts:
            result = steerer.steer_generation(
                dlm_wrapper=self.dlm,
                prompt=p['prompt'],
                **kwargs,
            )
            result['answer'] = p.get('answer')
            result['condition'] = condition_name
            results.append(result)
        
        return results
    
    def run_random_control(
        self, 
        prompts: List[Dict], 
        n_features: int = 50,
        n_random_sets: int = 5,
        alpha: float = 2.0,
        **kwargs,
    ) -> List[Dict]:
        """Control experiment: steer with random features."""
        import random
        
        all_results = []
        d_dict = self.sae.d_dict
        
        for seed in range(n_random_sets):
            random.seed(seed)
            random_features = random.sample(range(d_dict), n_features)
            
            results = self.run_steered(
                prompts=prompts,
                alpha=alpha,
                features=random_features,
                condition_name=f"random_control_{seed}",
                **kwargs,
            )
            all_results.extend(results)
        
        return all_results
    
    def run_alpha_sweep(
        self,
        prompts: List[Dict],
        alpha_values: List[float] = [0.5, 1.0, 2.0, 5.0, 10.0],
        **kwargs,
    ) -> Dict[float, List[Dict]]:
        """Sweep over steering strengths."""
        results = {}
        for alpha in alpha_values:
            logger.info(f"Alpha sweep: α={alpha}")
            results[alpha] = self.run_steered(
                prompts=prompts,
                alpha=alpha,
                condition_name=f"alpha_{alpha}",
                **kwargs,
            )
        return results
