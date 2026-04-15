"""
Top-K Sparse Autoencoder for Diffusion Language Models.

Implements the SAE architecture from DLM-Scope (ICLR 2026):
  "DLM-Scope: Mechanistic Interpretability of Diffusion Language Models 
   via Sparse Autoencoders" — Wang et al., 2026

The Top-K SAE enforces exact sparsity by retaining only the k largest
activations in the encoded representation. This follows the architecture
from Gao et al. (2024) "Scaling and Evaluating Sparse Autoencoders"
as adapted for DLMs by DLM-Scope.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math
import json
from pathlib import Path


class TopKSAE(nn.Module):
    """
    Top-K Sparse Autoencoder following DLM-Scope architecture.
    
    Architecture:
        encode: h = TopK(W_enc @ (x - b_dec) + b_enc)
        decode: x_hat = W_dec @ h + b_dec
    
    Key Design Choices (from DLM-Scope):
        - Pre-encoder bias subtraction (subtract decoder bias before encoding)
        - Top-K activation function (enforces exact L0 sparsity)
        - Unit-norm decoder columns (following Anthropic's approach)
        - Expansion factor typically 4x or 8x the model dimension
    
    Args:
        d_model: Dimension of the input activations (residual stream width)
        d_dict: Dictionary size (number of SAE features)
        k: Number of active features per input (sparsity level)
    """
    
    def __init__(self, d_model: int, d_dict: int, k: int):
        super().__init__()
        self.d_model = d_model
        self.d_dict = d_dict
        self.k = k
        
        # Encoder: maps input to sparse feature space
        self.W_enc = nn.Linear(d_model, d_dict, bias=True)
        
        # Decoder: maps sparse features back to activation space
        self.W_dec = nn.Linear(d_dict, d_model, bias=True)
        
        # Initialize weights following Anthropic/DLM-Scope conventions
        self._init_weights()
        
        # Track dead features for monitoring
        self.register_buffer(
            'feature_activation_count', 
            torch.zeros(d_dict, dtype=torch.long)
        )
        self.register_buffer('total_samples', torch.tensor(0, dtype=torch.long))
    
    def _init_weights(self):
        """Initialize weights with proper scaling."""
        # Xavier uniform for encoder
        nn.init.xavier_uniform_(self.W_enc.weight)
        nn.init.zeros_(self.W_enc.bias)
        
        # Decoder initialized as transpose of encoder, then normalized
        with torch.no_grad():
            self.W_dec.weight.copy_(self.W_enc.weight.T)
            self._normalize_decoder()
        nn.init.zeros_(self.W_dec.bias)
    
    def _normalize_decoder(self):
        """Normalize decoder columns to unit norm (Anthropic convention)."""
        with torch.no_grad():
            norms = self.W_dec.weight.norm(dim=0, keepdim=True)
            norms = torch.clamp(norms, min=1e-8)
            self.W_dec.weight.div_(norms)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input activations to sparse feature space.
        
        Following DLM-Scope: h = TopK(W_enc @ (x - b_dec) + b_enc)
        
        Args:
            x: Input activations [batch, ..., d_model]
        
        Returns:
            Sparse feature activations [batch, ..., d_dict]
        """
        # Pre-encoder bias subtraction (DLM-Scope convention)
        x_centered = x - self.W_dec.bias
        
        # Project to feature space
        pre_acts = self.W_enc(x_centered)  # [..., d_dict]
        
        # Apply ReLU then Top-K selection
        pre_acts = F.relu(pre_acts)
        
        # Top-K: keep only k largest activations, zero out the rest
        topk_values, topk_indices = torch.topk(pre_acts, self.k, dim=-1)
        
        # Create sparse activation tensor
        sparse_acts = torch.zeros_like(pre_acts)
        sparse_acts.scatter_(-1, topk_indices, topk_values)
        
        return sparse_acts
    
    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """
        Decode sparse features back to activation space.
        
        x_hat = W_dec @ h + b_dec
        
        Args:
            h: Sparse feature activations [..., d_dict]
        
        Returns:
            Reconstructed activations [..., d_model]
        """
        return self.W_dec(h)
    
    def forward(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Full forward pass: encode → decode with loss computation.
        
        Args:
            x: Input activations [batch, seq_len, d_model] or [batch, d_model]
        
        Returns:
            x_hat: Reconstructed activations (same shape as x)
            h: Sparse features [..., d_dict]
            metrics: Dict with 'recon_loss', 'l0', 'explained_variance'
        """
        h = self.encode(x)
        x_hat = self.decode(h)
        
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_hat, x)
        
        # Compute metrics
        with torch.no_grad():
            # L0: average number of active features
            l0 = (h > 0).float().sum(dim=-1).mean()
            
            # Explained variance
            total_var = (x - x.mean(dim=0, keepdim=True)).pow(2).sum()
            residual_var = (x - x_hat).pow(2).sum()
            explained_var = 1.0 - (residual_var / (total_var + 1e-8))
            
            # Track feature usage (for dead feature detection)
            if self.training:
                active_features = (h > 0).any(dim=tuple(range(h.dim() - 1)))
                self.feature_activation_count += active_features.long()
                self.total_samples += 1
        
        metrics = {
            'recon_loss': recon_loss,
            'l0': l0,
            'explained_variance': explained_var,
        }
        
        return x_hat, h, metrics
    
    def get_feature_direction(self, feature_idx: int) -> torch.Tensor:
        """
        Get the decoder direction (atom) for a specific feature.
        This is the direction v_f used for steering (DLM-Scope Eq. 4).
        
        Args:
            feature_idx: Index of the SAE feature
        
        Returns:
            Feature direction vector [d_model]
        """
        return self.W_dec.weight[:, feature_idx].detach()
    
    def get_feature_directions(self, feature_indices: list) -> torch.Tensor:
        """
        Get decoder directions for multiple features.
        
        Args:
            feature_indices: List of feature indices
        
        Returns:
            Feature direction matrix [d_model, n_features]
        """
        return self.W_dec.weight[:, feature_indices].detach()
    
    def get_dead_features(self, threshold: int = 15) -> torch.Tensor:
        """
        Identify features that have never/rarely activated.
        Threshold from DLM-Scope Appendix D: dead_latent_threshold=15.
        
        Returns:
            Boolean tensor [d_dict] — True for dead features
        """
        return self.feature_activation_count < threshold
    
    @property
    def num_dead_features(self) -> int:
        """Number of features that haven't activated enough."""
        return self.get_dead_features().sum().item()
    
    def save(self, path: str):
        """Save SAE weights and config."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save weights
        torch.save(self.state_dict(), path / "sae_weights.pt")
        
        # Save config
        config = {
            'd_model': self.d_model,
            'd_dict': self.d_dict,
            'k': self.k,
            'total_samples': self.total_samples.item(),
            'num_dead_features': self.num_dead_features,
        }
        with open(path / "sae_config.json", 'w') as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'TopKSAE':
        """Load SAE from saved weights and config."""
        path = Path(path)
        
        with open(path / "sae_config.json", 'r') as f:
            config = json.load(f)
        
        sae = cls(
            d_model=config['d_model'],
            d_dict=config['d_dict'],
            k=config['k'],
        )
        
        state_dict = torch.load(path / "sae_weights.pt", map_location=device)
        sae.load_state_dict(state_dict)
        
        return sae
    
    def __repr__(self):
        dead = self.num_dead_features
        return (
            f"TopKSAE(d_model={self.d_model}, d_dict={self.d_dict}, "
            f"k={self.k}, dead_features={dead}/{self.d_dict})"
        )


class SAEInsertionWrapper(nn.Module):
    """
    Wraps a trained SAE for insertion into a DLM's residual stream.
    
    This implements the "SAE insertion" evaluation from DLM-Scope Section 3.2:
    during inference, the residual stream at a target layer is replaced with
    the SAE reconstruction (x ← x_hat).
    
    Key finding from DLM-Scope: inserting SAEs in early layers of DLMs can
    actually REDUCE cross-entropy loss, unlike in autoregressive LLMs.
    """
    
    def __init__(self, sae: TopKSAE):
        super().__init__()
        self.sae = sae
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Replace activations with SAE reconstruction."""
        x_hat, _, _ = self.sae(x)
        return x_hat


def create_sae_for_model(
    d_model: int, 
    expansion_factor: int = 4, 
    k: int = 64
) -> TopKSAE:
    """
    Create a Top-K SAE with standard configuration.
    
    Args:
        d_model: Model hidden dimension
        expansion_factor: Dictionary size multiplier (4 or 8)
        k: Sparsity level
    
    Returns:
        Initialized TopKSAE
    """
    d_dict = d_model * expansion_factor
    return TopKSAE(d_model=d_model, d_dict=d_dict, k=k)
