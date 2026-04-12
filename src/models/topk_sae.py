"""
Top-K Sparse Autoencoder for DLMs.
Based on Gao et al. (2024) and adapted for diffusion models per DLM-Scope.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import json
from pathlib import Path


class TopKSAE(nn.Module):
    """
    Top-K SAE: encode h = TopK(W_enc(x - b_dec) + b_enc), decode x_hat = W_dec(h) + b_dec.
    Uses unit-norm decoder columns and pre-encoder bias subtraction.
    """

    def __init__(self, d_model: int, d_dict: int, k: int):
        super().__init__()
        self.d_model = d_model
        self.d_dict = d_dict
        self.k = k

        self.W_enc = nn.Linear(d_model, d_dict, bias=True)
        self.W_dec = nn.Linear(d_dict, d_model, bias=True)

        self._init_weights()

        self.register_buffer(
            'feature_activation_count',
            torch.zeros(d_dict, dtype=torch.long)
        )
        self.register_buffer('total_samples', torch.tensor(0, dtype=torch.long))

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_enc.weight)
        nn.init.zeros_(self.W_enc.bias)
        with torch.no_grad():
            self.W_dec.weight.copy_(self.W_enc.weight.T)
            self._normalize_decoder()
        nn.init.zeros_(self.W_dec.bias)

    def _normalize_decoder(self):
        """Normalize decoder columns to unit norm."""
        with torch.no_grad():
            norms = self.W_dec.weight.norm(dim=0, keepdim=True).clamp(min=1e-8)
            self.W_dec.weight.div_(norms)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_centered = x - self.W_dec.bias
        pre_acts = F.relu(self.W_enc(x_centered))

        topk_values, topk_indices = torch.topk(pre_acts, self.k, dim=-1)
        sparse_acts = torch.zeros_like(pre_acts)
        sparse_acts.scatter_(-1, topk_indices, topk_values)
        return sparse_acts

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        return self.W_dec(h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        h = self.encode(x)
        x_hat = self.decode(h)

        recon_loss = F.mse_loss(x_hat, x)

        with torch.no_grad():
            l0 = (h > 0).float().sum(dim=-1).mean()

            total_var = (x - x.mean(dim=0, keepdim=True)).pow(2).sum()
            residual_var = (x - x_hat).pow(2).sum()
            explained_var = 1.0 - (residual_var / (total_var + 1e-8))

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
        return self.W_dec.weight[:, feature_idx].detach()

    def get_feature_directions(self, feature_indices: list) -> torch.Tensor:
        return self.W_dec.weight[:, feature_indices].detach()

    def get_dead_features(self, threshold: int = 15) -> torch.Tensor:
        return self.feature_activation_count < threshold

    @property
    def num_dead_features(self) -> int:
        return self.get_dead_features().sum().item()

    def save(self, path: str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "sae_weights.pt")
        config = {
            'd_model': self.d_model, 'd_dict': self.d_dict, 'k': self.k,
            'total_samples': self.total_samples.item(),
            'num_dead_features': self.num_dead_features,
        }
        with open(path / "sae_config.json", 'w') as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'TopKSAE':
        path = Path(path)
        with open(path / "sae_config.json") as f:
            config = json.load(f)
        sae = cls(d_model=config['d_model'], d_dict=config['d_dict'], k=config['k'])
        sae.load_state_dict(torch.load(path / "sae_weights.pt", map_location=device))
        return sae

    def __repr__(self):
        return f"TopKSAE(d={self.d_model}, dict={self.d_dict}, k={self.k}, dead={self.num_dead_features})"


class SAEInsertionWrapper(nn.Module):
    """Replaces residual stream activations with SAE reconstructions for eval."""

    def __init__(self, sae: TopKSAE):
        super().__init__()
        self.sae = sae

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_hat, _, _ = self.sae(x)
        return x_hat


def create_sae_for_model(d_model: int, expansion_factor: int = 4, k: int = 64) -> TopKSAE:
    return TopKSAE(d_model=d_model, d_dict=d_model * expansion_factor, k=k)
