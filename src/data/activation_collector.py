"""Collects residual-stream activations during DLM denoising, with Drive checkpointing for Colab."""

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
import json
import logging
import gc

logger = logging.getLogger(__name__)


class ActivationCollector:
    def __init__(self, dlm_wrapper, target_layers, save_dir="results/checkpoints/activations",
                 batch_size=4, denoising_steps=30, timestep_samples=None):
        self.dlm = dlm_wrapper
        self.target_layers = target_layers
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.denoising_steps = denoising_steps
        self.timestep_samples = timestep_samples or list(range(denoising_steps))

    def collect_from_prompts(self, prompts, max_gen_tokens=128,
                             collection_name="activations", resume=True):
        checkpoint_file = self.save_dir / f"{collection_name}_checkpoint.json"
        start_idx = 0

        if resume and checkpoint_file.exists():
            with open(checkpoint_file) as f:
                start_idx = json.load(f).get('last_completed_batch', 0) * self.batch_size
            logger.info(f"Resuming from index {start_idx}")

        n_batches = (len(prompts) - start_idx + self.batch_size - 1) // self.batch_size
        stats = {'n_prompts': 0, 'layers': self.target_layers, 'timesteps': self.timestep_samples}

        for batch_idx in tqdm(range(n_batches), desc=f"Collecting {collection_name}"):
            real_idx = start_idx + batch_idx * self.batch_size
            batch_prompts = prompts[real_idx:real_idx + self.batch_size]
            if not batch_prompts:
                break

            texts = [p['prompt'] for p in batch_prompts]
            encodings = self.dlm.tokenizer(
                texts, padding=True, truncation=True, max_length=512, return_tensors="pt",
            )
            prefix_lens = [len(self.dlm.tokenizer.encode(t)) for t in texts]

            input_ids = encodings['input_ids']
            gen_ids = torch.full((input_ids.shape[0], max_gen_tokens), self.dlm.mask_token_id, dtype=torch.long)
            full_ids = torch.cat([input_ids, gen_ids], dim=1)

            result = self.dlm.denoising_loop(
                input_ids=full_ids, num_steps=self.denoising_steps,
                prefix_len=min(prefix_lens), temperature=1.0,
                collect_activations=True, activation_layers=self.target_layers,
                activation_timesteps=self.timestep_samples,
            )

            if result['activations']:
                batch_dir = self.save_dir / collection_name / f"batch_{batch_idx:04d}"
                batch_dir.mkdir(parents=True, exist_ok=True)
                for ts, layer_acts in result['activations'].items():
                    for layer_idx, acts in layer_acts.items():
                        torch.save(acts.half(), batch_dir / f"t{ts}_l{layer_idx}.pt")
                with open(batch_dir / "meta.json", 'w') as f:
                    json.dump({
                        'prompts': [p.get('idx', i) for i, p in enumerate(batch_prompts)],
                        'prompt_type': batch_prompts[0].get('prompt_type', 'unknown'),
                        'prefix_len': min(prefix_lens), 'gen_len': max_gen_tokens,
                    }, f)

            with open(checkpoint_file, 'w') as f:
                json.dump({
                    'last_completed_batch': batch_idx + 1,
                    'total_prompts': len(prompts), 'collection_name': collection_name,
                }, f)

            stats['n_prompts'] += len(batch_prompts)
            del result; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info(f"Collected {stats['n_prompts']} prompts, {len(self.target_layers)} layers")
        return stats

    def load_activations(self, collection_name, layer_idx, timestep, position_type="all"):
        collection_dir = self.save_dir / collection_name
        all_acts = []
        for batch_dir in sorted(collection_dir.iterdir()):
            if not batch_dir.is_dir() or not batch_dir.name.startswith("batch_"):
                continue
            act_file = batch_dir / f"t{timestep}_l{layer_idx}.pt"
            if act_file.exists():
                acts = torch.load(act_file, map_location='cpu').float()
                all_acts.append(acts.reshape(-1, acts.shape[-1]))
        if not all_acts:
            raise FileNotFoundError(f"No activations for {collection_name} layer {layer_idx} step {timestep}")
        return torch.cat(all_acts, dim=0)

    def get_training_data(self, collection_name, layer_idx, max_tokens=500_000):
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
            raise ValueError(f"No training data for layer {layer_idx}")
        combined = torch.cat(all_acts, dim=0)
        if combined.shape[0] > max_tokens:
            combined = combined[torch.randperm(combined.shape[0])[:max_tokens]]
        logger.info(f"Training data: {combined.shape[0]} tokens from layer {layer_idx}")
        return combined
