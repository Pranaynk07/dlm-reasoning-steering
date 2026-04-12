"""
Wrapper for DiffuGPT/Dream with hook-based activation extraction and steering.
Handles weight remapping from DiffuGPT checkpoint format to GPT-2 architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, Dict, List, Callable, Tuple
import numpy as np
from pathlib import Path
import sys
import os
import logging

logger = logging.getLogger(__name__)

DIFFUGPT_VOCAB_SIZE = 50258
DIFFUGPT_MASK_TOKEN_ID = 50257


class DLMWrapper:
    MASK_TOKEN = "[MASK]"

    def __init__(
        self,
        model_name: str = "diffusionfamily/diffugpt-m",
        base_model_name: str = "gpt2-medium",
        device: str = "auto",
        quantize_4bit: bool = False,
        cache_dir: Optional[str] = None,
    ):
        self.model_name = model_name
        self.base_model_name = base_model_name
        self.quantize_4bit = quantize_4bit
        self.cache_dir = cache_dir

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._activations: Dict[int, torch.Tensor] = {}
        self._hooks: List = []

        self._load_model()

        self.d_model = self.config.n_embd if hasattr(self.config, 'n_embd') else self.config.hidden_size
        self.n_layers = self.config.n_layer if hasattr(self.config, 'n_layer') else self.config.num_hidden_layers
        self.mask_token_id = self._mask_token_id

        logger.info(f"Loaded {model_name}: d_model={self.d_model}, n_layers={self.n_layers}, device={self.device}")

    def _load_model(self):
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, cache_dir=self.cache_dir, trust_remote_code=True,
            )
        except Exception:
            logger.info(f"Tokenizer not in {self.model_name}, falling back to {self.base_model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, cache_dir=self.cache_dir)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model and resize for DiffuGPT's extra mask token
        logger.info(f"Loading base model: {self.base_model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name, cache_dir=self.cache_dir,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
        )
        self.model.resize_token_embeddings(DIFFUGPT_VOCAB_SIZE)
        self._mask_token_id = DIFFUGPT_MASK_TOKEN_ID

        # Try loading DiffuGPT weights with name remapping
        if self.model_name != self.base_model_name:
            try:
                self._load_diffugpt_weights()
                logger.info("DiffuGPT weights loaded")
            except Exception as e:
                logger.warning(f"Could not load DiffuGPT weights: {e}. Using base GPT-2 weights.")
                with torch.no_grad():
                    self.model.transformer.wte.weight[self._mask_token_id] = 0.0

        self.model = self.model.to(self.device)
        self.config = self.model.config
        self.model.eval()

    def _load_diffugpt_weights(self):
        """Remap DiffuGPT checkpoint keys (denoise_model.* -> transformer.*) and load."""
        from safetensors.torch import load_file as load_safetensors
        from huggingface_hub import hf_hub_download

        try:
            ckpt_path = hf_hub_download(self.model_name, "model.safetensors", cache_dir=self.cache_dir)
            raw_state = load_safetensors(ckpt_path)
        except Exception:
            ckpt_path = hf_hub_download(self.model_name, "pytorch_model.bin", cache_dir=self.cache_dir)
            raw_state = torch.load(ckpt_path, map_location="cpu")

        mapped_state = {}
        for key, value in raw_state.items():
            if key.startswith("denoise_model."):
                mapped_state["transformer." + key[len("denoise_model."):]] = value
            elif key == "embed_tokens.weight":
                mapped_state["transformer.wte.weight"] = value
                mapped_state["lm_head.weight"] = value.clone()
            else:
                mapped_state[key] = value

        result = self.model.load_state_dict(mapped_state, strict=False)
        n_loaded = len(mapped_state) - len(result.unexpected_keys)
        logger.info(f"Weights mapped: {n_loaded} loaded, {len(result.missing_keys)} missing")

        if len(result.missing_keys) > 5:
            logger.warning("Many missing keys -- model may not work correctly")

    def _get_layer_module(self, layer_idx: int):
        if hasattr(self.model, 'transformer'):
            return self.model.transformer.h[layer_idx]
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers[layer_idx]
        raise ValueError(f"Unknown architecture: {self.model_name}")

    def register_activation_hooks(self, target_layers: List[int]):
        self.clear_hooks()
        self._activations = {}
        for layer_idx in target_layers:
            layer = self._get_layer_module(layer_idx)

            def make_hook(idx):
                def hook_fn(module, input, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    self._activations[idx] = hidden.detach().cpu()
                return hook_fn

            self._hooks.append(layer.register_forward_hook(make_hook(layer_idx)))

    def clear_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self._activations = {}

    def get_activations(self) -> Dict[int, torch.Tensor]:
        return dict(self._activations)

    def create_masked_input(
        self, input_ids: torch.Tensor, mask_rate: float = 0.5, preserve_prefix: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=input_ids.device)
        maskable_len = seq_len - preserve_prefix
        num_to_mask = int(maskable_len * mask_rate)

        for b in range(batch_size):
            perm = torch.randperm(maskable_len, device=input_ids.device)[:num_to_mask]
            mask[b, preserve_prefix + perm] = True

        masked_ids = input_ids.clone()
        masked_ids[mask] = self.mask_token_id
        return masked_ids, mask

    @torch.no_grad()
    def forward_pass(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        input_ids = input_ids.to(self.device)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = attention_mask.to(self.device)
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    @torch.no_grad()
    def denoising_step(
        self, masked_ids, mask, temperature=1.0,
        steering_hook=None, steering_layer=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        steer_handle = None
        if steering_hook is not None and steering_layer is not None:
            layer_module = self._get_layer_module(steering_layer)

            def _steer_fn(module, input, output):
                if isinstance(output, tuple):
                    return (steering_hook(output[0], mask),) + output[1:]
                return steering_hook(output, mask)

            steer_handle = layer_module.register_forward_hook(_steer_fn)

        try:
            logits = self.forward_pass(masked_ids)
            if temperature != 1.0:
                logits = logits / temperature

            probs = F.softmax(logits, dim=-1)
            if temperature > 0:
                predicted_ids = torch.multinomial(
                    probs.view(-1, probs.size(-1)), num_samples=1
                ).view(probs.shape[:-1])
            else:
                predicted_ids = logits.argmax(dim=-1)

            confidences = probs.gather(-1, predicted_ids.unsqueeze(-1)).squeeze(-1)
        finally:
            if steer_handle is not None:
                steer_handle.remove()

        return predicted_ids, confidences

    @torch.no_grad()
    def denoising_loop(
        self, input_ids, num_steps=30, prefix_len=0, temperature=1.0,
        steering_hook=None, steering_layer=None,
        collect_activations=False, activation_layers=None, activation_timesteps=None,
    ) -> Dict:
        """Run the full denoising loop with confidence-based remasking."""
        batch_size, seq_len = input_ids.shape
        gen_len = seq_len - prefix_len

        current_ids = input_ids.clone().to(self.device)
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=self.device)
        mask[:, prefix_len:] = True
        current_ids[:, prefix_len:] = self.mask_token_id

        trajectory = []
        all_activations = {} if collect_activations else None

        if collect_activations and activation_layers:
            self.register_activation_hooks(activation_layers)
        if activation_timesteps is None:
            activation_timesteps = list(range(num_steps))

        for step in range(num_steps):
            n_unmask = min(int(gen_len * (step + 1) / num_steps), gen_len)

            predicted_ids, confidences = self.denoising_step(
                current_ids, mask, temperature,
                steering_hook=steering_hook, steering_layer=steering_layer,
            )

            if collect_activations and step in activation_timesteps:
                all_activations[step] = {
                    layer: act.clone() for layer, act in self._activations.items()
                }

            filled_ids = current_ids.clone()
            filled_ids[mask] = predicted_ids[mask].to(filled_ids.device)

            if step < num_steps - 1:
                # Keep the most confident predictions, re-mask the rest
                gen_confidences = confidences[:, prefix_len:].clone()
                gen_confidences[~mask[:, prefix_len:]] = float('inf')
                _, keep_indices = gen_confidences.topk(n_unmask, dim=-1)

                new_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=self.device)
                new_mask[:, :prefix_len] = False
                for b in range(batch_size):
                    new_mask[b, prefix_len + keep_indices[b]] = False

                current_ids = filled_ids.clone()
                current_ids[new_mask] = self.mask_token_id
                mask = new_mask
            else:
                current_ids = filled_ids
                mask = torch.zeros_like(mask)

            trajectory.append(current_ids.cpu().clone())

        if collect_activations:
            self.clear_hooks()

        return {
            'output_ids': current_ids.cpu(),
            'trajectory': trajectory,
            'activations': all_activations,
        }

    def generate(
        self, prompt, max_new_tokens=128, num_steps=30, temperature=1.0,
        steering_hook=None, steering_layer=None,
    ) -> str:
        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        prefix_len = prompt_ids.shape[1]

        gen_ids = torch.full((1, max_new_tokens), self.mask_token_id, dtype=torch.long)
        input_ids = torch.cat([prompt_ids, gen_ids], dim=1)

        result = self.denoising_loop(
            input_ids=input_ids, num_steps=num_steps, prefix_len=prefix_len,
            temperature=temperature, steering_hook=steering_hook, steering_layer=steering_layer,
        )
        return self.tokenizer.decode(result['output_ids'][0], skip_special_tokens=True)

    def get_model_info(self) -> Dict:
        return {
            'model_name': self.model_name, 'd_model': self.d_model,
            'n_layers': self.n_layers, 'device': str(self.device),
            'quantized': self.quantize_4bit, 'mask_token_id': self.mask_token_id,
            'vocab_size': self.config.vocab_size,
        }
