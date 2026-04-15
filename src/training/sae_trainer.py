"""
SAE Trainer for Diffusion Language Models.

Implements DLM-specific SAE training following DLM-Scope Section 3.1:
- Samples activations from denoising at various corruption levels
- Trains separate Mask-SAE (masked positions) and Unmask-SAE (unmasked)
- Uses Top-K architecture with reconstruction loss
- Tracks training metrics: L0, explained variance, dead features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Optional
from pathlib import Path
from tqdm import tqdm
import json
import logging
import time

try:
    from src.models.topk_sae import TopKSAE, create_sae_for_model
except ImportError:
    from ..models.topk_sae import TopKSAE, create_sae_for_model

logger = logging.getLogger(__name__)


class SAETrainer:
    """
    Trains Top-K SAEs on DLM activations.
    
    Training details from DLM-Scope Appendix B:
    - Context length: 2048 (we use 1024 for memory)
    - Batch size: 8
    - Token budget: 1M-5M tokens
    - Adam optimizer
    """
    
    def __init__(
        self,
        d_model: int,
        expansion_factor: int = 4,
        k: int = 64,
        learning_rate: float = 3e-4,
        batch_size: int = 256,
        device: str = "auto",
    ):
        """
        Args:
            d_model: Model hidden dimension
            expansion_factor: Dictionary size = expansion_factor * d_model
            k: Top-K sparsity level
            learning_rate: Adam learning rate
            batch_size: Training batch size (in tokens)
            device: Training device
        """
        self.d_model = d_model
        self.expansion_factor = expansion_factor
        self.k = k
        self.lr = learning_rate
        self.batch_size = batch_size
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Create SAE
        self.sae = create_sae_for_model(d_model, expansion_factor, k)
        self.sae = self.sae.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.sae.parameters(), lr=self.lr)
        
        # Training state
        self.step = 0
        self.train_history = []
    
    def train(
        self,
        activations: torch.Tensor,
        n_epochs: int = 5,
        log_every: int = 100,
        save_dir: Optional[str] = None,
        save_every: int = 1000,
    ) -> Dict:
        """
        Train the SAE on collected activations.
        
        Args:
            activations: Training data [n_tokens, d_model]
            n_epochs: Number of passes through the data
            log_every: Log metrics every N steps
            save_dir: Directory for saving checkpoints
            save_every: Save checkpoint every N steps
        
        Returns:
            Dict with final training metrics
        """
        logger.info(
            f"Training SAE: {activations.shape[0]} tokens, "
            f"d_model={self.d_model}, d_dict={self.d_model * self.expansion_factor}, "
            f"k={self.k}"
        )
        
        # Create data loader
        dataset = TensorDataset(activations)
        loader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            drop_last=True,
        )
        
        self.sae.train()
        best_loss = float('inf')
        start_time = time.time()
        
        for epoch in range(n_epochs):
            epoch_losses = []
            
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{n_epochs}")
            for batch in pbar:
                x = batch[0].to(self.device)
                
                # Forward pass
                x_hat, h, metrics = self.sae(x)
                loss = metrics['recon_loss']
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.sae.parameters(), 1.0)
                
                self.optimizer.step()
                
                # Normalize decoder columns (Anthropic convention)
                self.sae._normalize_decoder()
                
                # Track metrics
                self.step += 1
                epoch_losses.append(loss.item())
                
                self.train_history.append({
                    'step': self.step,
                    'loss': loss.item(),
                    'l0': metrics['l0'].item(),
                    'explained_variance': metrics['explained_variance'].item(),
                })
                
                # Log
                if self.step % log_every == 0:
                    pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'EV': f"{metrics['explained_variance'].item():.3f}",
                        'L0': f"{metrics['l0'].item():.1f}",
                        'dead': self.sae.num_dead_features,
                    })
                
                # Save checkpoint
                if save_dir and self.step % save_every == 0:
                    self._save_checkpoint(save_dir)
            
            # Epoch summary
            mean_loss = sum(epoch_losses) / len(epoch_losses)
            logger.info(
                f"Epoch {epoch+1}: loss={mean_loss:.4f}, "
                f"EV={metrics['explained_variance'].item():.3f}, "
                f"dead_features={self.sae.num_dead_features}/{self.sae.d_dict}"
            )
            
            if mean_loss < best_loss:
                best_loss = mean_loss
                if save_dir:
                    self.sae.save(Path(save_dir) / "best")
        
        elapsed = time.time() - start_time
        
        # Final save
        if save_dir:
            self.sae.save(Path(save_dir) / "final")
            self._save_training_log(save_dir)
        
        self.sae.eval()
        
        final_metrics = {
            'final_loss': mean_loss,
            'final_ev': metrics['explained_variance'].item(),
            'final_l0': metrics['l0'].item(),
            'dead_features': self.sae.num_dead_features,
            'total_steps': self.step,
            'training_time_seconds': elapsed,
        }
        
        logger.info(f"Training complete in {elapsed:.0f}s: {final_metrics}")
        return final_metrics
    
    def _save_checkpoint(self, save_dir: str):
        """Save training checkpoint."""
        path = Path(save_dir) / "checkpoint"
        path.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'sae_state_dict': self.sae.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': self.step,
        }, path / "trainer_checkpoint.pt")
    
    def _save_training_log(self, save_dir: str):
        """Save training history."""
        path = Path(save_dir)
        with open(path / "training_log.json", 'w') as f:
            json.dump(self.train_history, f)
    
    def evaluate(self, activations: torch.Tensor) -> Dict:
        """
        Evaluate SAE on held-out activations.
        
        Computes DLM-Scope metrics:
        - Explained variance (Eq. 15)
        - L0 sparsity
        - Dead feature count
        """
        self.sae.eval()
        
        dataset = TensorDataset(activations)
        loader = DataLoader(dataset, batch_size=self.batch_size)
        
        total_ev = 0
        total_l0 = 0
        n_batches = 0
        
        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(self.device)
                x_hat, h, metrics = self.sae(x)
                total_ev += metrics['explained_variance'].item()
                total_l0 += metrics['l0'].item()
                n_batches += 1
        
        return {
            'explained_variance': total_ev / n_batches,
            'l0': total_l0 / n_batches,
            'dead_features': self.sae.num_dead_features,
            'd_dict': self.sae.d_dict,
        }


def train_sae_for_layer(
    activations: torch.Tensor,
    d_model: int,
    layer_idx: int,
    save_dir: str,
    expansion_factor: int = 4,
    k: int = 64,
    n_epochs: int = 5,
    **kwargs,
) -> TopKSAE:
    """
    Convenience function: train a single SAE for one layer.
    
    Returns the trained SAE.
    """
    trainer = SAETrainer(
        d_model=d_model,
        expansion_factor=expansion_factor,
        k=k,
        **kwargs,
    )
    
    layer_save_dir = Path(save_dir) / f"layer_{layer_idx}"
    
    metrics = trainer.train(
        activations=activations,
        n_epochs=n_epochs,
        save_dir=str(layer_save_dir),
    )
    
    logger.info(f"Layer {layer_idx} SAE: EV={metrics['final_ev']:.3f}, L0={metrics['final_l0']:.1f}")
    
    return trainer.sae
