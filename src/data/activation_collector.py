"""
Activation Collector for DLMs with Google Drive checkpointing.

Collects residual-stream activations from a DLM during denoising,
optimized for Colab's session limits:
- Batch processing with incremental saves to Drive
- Resume from last checkpoint if session disconnects
- Memory-efficient: activations stored on CPU, saved as compressed tensors

Follows DLM-Scope's methodology:
- Collect activations at specified layers and denoising timesteps
- Separate masked vs. unmasked position activations
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import json
import logging
import gc

logger = logging.getLogger(__name__)


class ActivationCollector:
    """
    Collects and manages DLM activations for SAE training and analysis.
    
    Designed for Colab resilience:
    - Saves after each batch to Google Drive
    - Can resume from last saved batch
    - Compresses activations to reduce storage
    """
    
    def __init__(
        self,
        dlm_wrapper,
        target_layers: List[int],
        save_dir: str = "results/checkpoints/activations",
        batch_size: int = 4,
        denoising_steps: int = 30,
        timestep_samples: Optional[List[int]] = None,
    ):
        """
        Args:
            dlm_wrapper: Initialized DLMWrapper instance
            target_layers: Layer indices to collect activations from
            save_dir: Directory for saving activation checkpoints
            batch_size: Number of prompts per batch
            denoising_steps: Number of denoising steps per generation
            timestep_samples: Which timesteps to collect (None = all)
        """
        self.dlm = dlm_wrapper
        self.target_layers = target_layers
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.denoising_steps = denoising_steps
        self.timestep_samples = timestep_samples or list(range(denoising_steps))
    
    def collect_from_prompts(
        self,
        prompts: List[Dict],
        max_gen_tokens: int = 128,
        collection_name: str = "activations",
        resume: bool = True,
    ) -> Dict:
        """
        Collect activations for a list of prompts.
        
        For each prompt:
        1. Tokenize and create masked generation sequence
        2. Run denoising loop with activation hooks
        3. Save activations per layer and timestep
        
        Args:
            prompts: List of prompt dicts with 'prompt' key
            max_gen_tokens: Number of tokens to generate
            collection_name: Name for this collection (for save/resume)
            resume: Whether to resume from last checkpoint
        
        Returns:
            Dict with activation statistics
        """
        checkpoint_file = self.save_dir / f"{collection_name}_checkpoint.json"
        start_idx = 0
        
        # Check for resume
        if resume and checkpoint_file.exists():
            with open(checkpoint_file) as f:
                ckpt = json.load(f)
            start_idx = ckpt.get('last_completed_batch', 0) * self.batch_size
            logger.info(f"Resuming from prompt index {start_idx}")
        
        # Process in batches
        n_batches = (len(prompts) - start_idx + self.batch_size - 1) // self.batch_size
        
        all_stats = {
            'n_prompts': 0,
            'layers': self.target_layers,
            'timesteps': self.timestep_samples,
        }
        
        for batch_idx in tqdm(range(n_batches), desc=f"Collecting {collection_name}"):
            real_idx = start_idx + batch_idx * self.batch_size
            batch_prompts = prompts[real_idx:real_idx + self.batch_size]
            
            if not batch_prompts:
                break
            
            # Tokenize batch
            texts = [p['prompt'] for p in batch_prompts]
            encodings = self.dlm.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            
            prefix_lens = [
                len(self.dlm.tokenizer.encode(t)) 
                for t in texts
            ]
            min_prefix = min(prefix_lens)
            
            # Create generation sequence (prefix + masked generation tokens)
            input_ids = encodings['input_ids']
            gen_ids = torch.full(
                (input_ids.shape[0], max_gen_tokens),
                self.dlm.mask_token_id,
                dtype=torch.long,
            )
            full_ids = torch.cat([input_ids, gen_ids], dim=1)
            
            # Run denoising with activation collection
            result = self.dlm.denoising_loop(
                input_ids=full_ids,
                num_steps=self.denoising_steps,
                prefix_len=min_prefix,
                temperature=1.0,
                collect_activations=True,
                activation_layers=self.target_layers,
                activation_timesteps=self.timestep_samples,
            )
            
            # Save activations for this batch
            if result['activations']:
                batch_save_dir = self.save_dir / collection_name / f"batch_{batch_idx:04d}"
                batch_save_dir.mkdir(parents=True, exist_ok=True)
                
                for timestep, layer_acts in result['activations'].items():
                    for layer_idx, acts in layer_acts.items():
                        save_path = batch_save_dir / f"t{timestep}_l{layer_idx}.pt"
                        torch.save(acts.half(), save_path)  # Save as FP16
                
                # Save metadata for this batch
                batch_meta = {
                    'prompts': [p.get('idx', i) for i, p in enumerate(batch_prompts)],
                    'prompt_type': batch_prompts[0].get('prompt_type', 'unknown'),
                    'prefix_len': min_prefix,
                    'gen_len': max_gen_tokens,
                }
                with open(batch_save_dir / "meta.json", 'w') as f:
                    json.dump(batch_meta, f)
            
            # Update checkpoint
            with open(checkpoint_file, 'w') as f:
                json.dump({
                    'last_completed_batch': batch_idx + 1,
                    'total_prompts': len(prompts),
                    'collection_name': collection_name,
                }, f)
            
            all_stats['n_prompts'] += len(batch_prompts)
            
            # Memory cleanup
            del result
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        logger.info(
            f"Collected activations for {all_stats['n_prompts']} prompts, "
            f"{len(self.target_layers)} layers, "
            f"{len(self.timestep_samples)} timesteps"
        )
        
        return all_stats
    
    def load_activations(
        self,
        collection_name: str,
        layer_idx: int,
        timestep: int,
        position_type: str = "all",
    ) -> torch.Tensor:
        """
        Load saved activations for a specific layer and timestep.
        
        Args:
            collection_name: Name of the collection
            layer_idx: Which layer's activations
            timestep: Which denoising timestep
            position_type: "all", "mask", or "unmask"
        
        Returns:
            Concatenated activations [n_total_tokens, d_model]
        """
        collection_dir = self.save_dir / collection_name
        all_acts = []
        
        for batch_dir in sorted(collection_dir.iterdir()):
            if not batch_dir.is_dir() or not batch_dir.name.startswith("batch_"):
                continue
            
            act_file = batch_dir / f"t{timestep}_l{layer_idx}.pt"
            if act_file.exists():
                acts = torch.load(act_file, map_location='cpu').float()
                # acts shape: [batch, seq_len, d_model]
                
                if position_type == "all":
                    # Flatten batch and seq_len
                    all_acts.append(acts.reshape(-1, acts.shape[-1]))
                else:
                    # Would need mask info — for now, use all
                    all_acts.append(acts.reshape(-1, acts.shape[-1]))
        
        if not all_acts:
            raise FileNotFoundError(
                f"No activations found for {collection_name}, "
                f"layer {layer_idx}, timestep {timestep}"
            )
        
        return torch.cat(all_acts, dim=0)
    
    def get_training_data(
        self,
        collection_name: str,
        layer_idx: int,
        max_tokens: int = 500_000,
    ) -> torch.Tensor:
        """
        Get flattened activation data suitable for SAE training.
        
        Aggregates across all timesteps for a given layer,
        creating a diverse training set.
        
        Args:
            collection_name: Name of the collection
            layer_idx: Which layer
            max_tokens: Maximum number of token activations
        
        Returns:
            Training activations [n_tokens, d_model]
        """
        all_acts = []
        total = 0
        
        for ts in self.timestep_samples:
            try:
                acts = self.load_activations(collection_name, layer_idx, ts)
                all_acts.append(acts)
                total += acts.shape[0]
                
                if total >= max_tokens:
                    break
            except FileNotFoundError:
                continue
        
        if not all_acts:
            raise ValueError(f"No training data found for layer {layer_idx}")
        
        combined = torch.cat(all_acts, dim=0)
        
        # Subsample if too large
        if combined.shape[0] > max_tokens:
            indices = torch.randperm(combined.shape[0])[:max_tokens]
            combined = combined[indices]
        
        logger.info(f"Training data: {combined.shape[0]} tokens, layer {layer_idx}")
        return combined
