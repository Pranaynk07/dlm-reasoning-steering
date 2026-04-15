"""
DLM Wrapper for DiffuGPT/Dream with activation extraction.

Provides a unified interface for:
1. Loading DiffuGPT-Medium or Dream-7B (with optional 4-bit quantization)
2. Running the denoising loop (DLM-Scope Eq. 8)
3. Extracting residual-stream activations via forward hooks
4. Generating text with optional steering hooks

References:
  - DiffuLLaMA: https://github.com/HKUNLP/DiffuLLaMA
  - DLM-Scope: arXiv:2602.05859
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


class DLMWrapper:
    """
    Wrapper for Diffusion Language Models (DiffuGPT / Dream) with 
    hook-based activation extraction and steering support.
    
    Design decisions:
    - Uses forward hooks to extract activations non-invasively
    - Implements the masked discrete diffusion denoising loop
    - Supports both DiffuGPT (GPT-2 based) and Dream (LLaMA based)
    - Handles Colab-specific memory management
    """
    
    # Special mask token ID (DiffuGPT uses a dedicated mask token)
    MASK_TOKEN = "[MASK]"
    
    def __init__(
        self,
        model_name: str = "diffusionfamily/diffugpt-m",
        base_model_name: str = "gpt2-medium",
        device: str = "auto",
        quantize_4bit: bool = False,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the DLM wrapper.
        
        Args:
            model_name: HuggingFace model name or local path
            base_model_name: Base model config name (for DiffuGPT)
            device: Device to load model on ("auto", "cuda", "cpu")
            quantize_4bit: Whether to use 4-bit quantization (for Dream-7B)
            cache_dir: Directory for model cache (useful on Colab with Drive)
        """
        self.model_name = model_name
        self.base_model_name = base_model_name
        self.quantize_4bit = quantize_4bit
        self.cache_dir = cache_dir
        
        # Resolve device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Activation storage (populated by hooks)
        self._activations: Dict[int, torch.Tensor] = {}
        self._hooks: List = []
        
        # Load model and tokenizer
        self._load_model()
        
        # Model architecture info
        self.d_model = self.config.n_embd if hasattr(self.config, 'n_embd') else self.config.hidden_size
        self.n_layers = self.config.n_layer if hasattr(self.config, 'n_layer') else self.config.num_hidden_layers
        
        # Get mask token id
        self.mask_token_id = self._get_mask_token_id()
        
        logger.info(
            f"Loaded {model_name}: d_model={self.d_model}, "
            f"n_layers={self.n_layers}, device={self.device}"
        )
    
    def _load_model(self):
        """Load the DLM model and tokenizer with robust fallback."""
        logger.info(f"Loading model: {self.model_name}")
        
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )
        except Exception:
            logger.info(f"Tokenizer not found for {self.model_name}, using {self.base_model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name,
                cache_dir=self.cache_dir,
            )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model — try primary, fall back to base
        model_loaded = False
        
        if self.quantize_4bit:
            try:
                from transformers import BitsAndBytesConfig
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=quant_config,
                    device_map="auto",
                    cache_dir=self.cache_dir,
                    trust_remote_code=True,
                )
                model_loaded = True
            except Exception as e:
                logger.warning(f"4-bit loading failed: {e}")
        
        if not model_loaded:
            # Try loading the primary model name
            for name in [self.model_name, self.base_model_name]:
                try:
                    logger.info(f"Trying to load: {name}")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        name,
                        cache_dir=self.cache_dir,
                        trust_remote_code=True,
                        torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                    )
                    self.model = self.model.to(self.device)
                    self.model_name = name  # Record which model actually loaded
                    model_loaded = True
                    logger.info(f"✅ Successfully loaded: {name}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load {name}: {e}")
                    continue
        
        if not model_loaded:
            raise RuntimeError(
                f"Could not load any model. Tried: {self.model_name}, {self.base_model_name}"
            )
        
        # Load config from whichever model loaded
        self.config = self.model.config
        self.model.eval()
    
    def _get_mask_token_id(self) -> int:
        """
        Get the mask token ID for diffusion masking.
        
        Strategy: Use the tokenizer's mask_token if available,
        otherwise use eos_token_id as a sentinel for masked positions.
        We clamp generated tokens to avoid this ID in output.
        """
        # Check for dedicated mask token
        if hasattr(self.tokenizer, 'mask_token_id') and self.tokenizer.mask_token_id is not None:
            return self.tokenizer.mask_token_id
        
        # Use eos_token_id as mask sentinel (safe — we filter it in decode)
        if self.tokenizer.eos_token_id is not None:
            return self.tokenizer.eos_token_id
        
        # Last resort: use last vocab token
        return self.config.vocab_size - 1
    
    def _get_layer_module(self, layer_idx: int):
        """Get the transformer layer module by index."""
        if hasattr(self.model, 'transformer'):
            # GPT-2 style (DiffuGPT)
            return self.model.transformer.h[layer_idx]
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            # LLaMA style (Dream)
            return self.model.model.layers[layer_idx]
        else:
            raise ValueError(f"Unknown model architecture for {self.model_name}")
    
    def _get_residual_stream_hook_point(self, layer_idx: int):
        """
        Get the hook point for the residual stream AFTER layer `layer_idx`.
        
        In DLM-Scope, SAEs are trained on residual-stream activations at 
        the output of each transformer block (post-layernorm, pre-next-layer).
        """
        layer = self._get_layer_module(layer_idx)
        return layer  # Hook on the full layer output
    
    def register_activation_hooks(self, target_layers: List[int]):
        """
        Register forward hooks to capture residual-stream activations.
        
        Args:
            target_layers: Layer indices to capture activations from
        """
        self.clear_hooks()
        self._activations = {}
        
        for layer_idx in target_layers:
            hook_point = self._get_residual_stream_hook_point(layer_idx)
            
            def make_hook(idx):
                def hook_fn(module, input, output):
                    # output is typically a tuple; first element is hidden states
                    if isinstance(output, tuple):
                        hidden_states = output[0]
                    else:
                        hidden_states = output
                    # Store on CPU to save GPU memory
                    self._activations[idx] = hidden_states.detach().cpu()
                return hook_fn
            
            handle = hook_point.register_forward_hook(make_hook(layer_idx))
            self._hooks.append(handle)
    
    def clear_hooks(self):
        """Remove all registered hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks = []
        self._activations = {}
    
    def get_activations(self) -> Dict[int, torch.Tensor]:
        """
        Get captured activations from the last forward pass.
        
        Returns:
            Dict mapping layer_idx -> activation tensor [batch, seq_len, d_model]
        """
        return dict(self._activations)
    
    def create_masked_input(
        self,
        input_ids: torch.Tensor,
        mask_rate: float = 0.5,
        preserve_prefix: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create a partially masked version of input_ids for diffusion.
        
        Following DLM-Scope: sample corruption level t and mask tokens
        at rate t. This creates x_t from x_0.
        
        Args:
            input_ids: Original token ids [batch, seq_len]
            mask_rate: Fraction of tokens to mask (t in DLM-Scope)
            preserve_prefix: Number of prefix tokens to never mask (for prompting)
        
        Returns:
            masked_ids: Token ids with some replaced by mask_token_id
            mask: Boolean tensor indicating which positions are masked
        """
        batch_size, seq_len = input_ids.shape
        
        # Create mask (True = masked)
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=input_ids.device)
        
        # Only mask positions after prefix
        maskable_len = seq_len - preserve_prefix
        num_to_mask = int(maskable_len * mask_rate)
        
        for b in range(batch_size):
            # Randomly select positions to mask (after prefix)
            perm = torch.randperm(maskable_len, device=input_ids.device)[:num_to_mask]
            mask[b, preserve_prefix + perm] = True
        
        # Apply mask
        masked_ids = input_ids.clone()
        masked_ids[mask] = self.mask_token_id
        
        return masked_ids, mask
    
    @torch.no_grad()
    def forward_pass(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Single forward pass through the model.
        
        Args:
            input_ids: [batch, seq_len]
            attention_mask: Optional attention mask
        
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        input_ids = input_ids.to(self.device)
        
        # DiffuGPT uses bidirectional attention (no causal mask)
        # We need to pass full attention mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = attention_mask.to(self.device)
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        return outputs.logits
    
    @torch.no_grad()  
    def denoising_step(
        self,
        masked_ids: torch.Tensor,
        mask: torch.Tensor,
        temperature: float = 1.0,
        steering_hook: Optional[Callable] = None,
        steering_layer: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single denoising step: predict tokens at masked positions.
        
        Implements one step of DLM-Scope Eq. 8:
        1. Forward pass on current masked sequence
        2. (Optional) Apply steering hook at target layer
        3. Sample predicted tokens for masked positions
        
        Args:
            masked_ids: Current partially masked sequence [batch, seq_len]
            mask: Boolean mask [batch, seq_len] (True = masked)
            temperature: Sampling temperature
            steering_hook: Optional function to modify activations at steering_layer
            steering_layer: Layer index for steering intervention
        
        Returns:
            predicted_ids: Best prediction for all positions [batch, seq_len]
            confidences: Prediction confidence scores [batch, seq_len]
        """
        # Set up steering hook if provided
        steer_handle = None
        if steering_hook is not None and steering_layer is not None:
            layer_module = self._get_residual_stream_hook_point(steering_layer)
            
            def _steer_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden_states = output[0]
                    steered = steering_hook(hidden_states, mask)
                    return (steered,) + output[1:]
                else:
                    return steering_hook(output, mask)
            
            steer_handle = layer_module.register_forward_hook(_steer_fn)
        
        try:
            # Forward pass
            logits = self.forward_pass(masked_ids)
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Get predictions
            probs = F.softmax(logits, dim=-1)
            
            # Sample or argmax
            if temperature > 0:
                predicted_ids = torch.multinomial(
                    probs.view(-1, probs.size(-1)), 
                    num_samples=1
                ).view(probs.shape[:-1])
            else:
                predicted_ids = logits.argmax(dim=-1)
            
            # Confidence = probability of predicted token
            confidences = probs.gather(-1, predicted_ids.unsqueeze(-1)).squeeze(-1)
            
        finally:
            # Clean up steering hook
            if steer_handle is not None:
                steer_handle.remove()
        
        return predicted_ids, confidences
    
    @torch.no_grad()
    def denoising_loop(
        self,
        input_ids: torch.Tensor,
        num_steps: int = 30,
        prefix_len: int = 0,
        temperature: float = 1.0,
        steering_hook: Optional[Callable] = None,
        steering_layer: Optional[int] = None,
        collect_activations: bool = False,
        activation_layers: Optional[List[int]] = None,
        activation_timesteps: Optional[List[int]] = None,
    ) -> Dict:
        """
        Full denoising loop: iteratively unmask tokens.
        
        Implements DLM-Scope Eq. 8:
        For each step k:
          1. Predict all masked tokens → x̃_0
          2. Fill masked positions with predictions
          3. Re-mask to match next step's mask rate (t_{k-1})
        
        Uses confidence-based remasking (entropy strategy from Dream):
        tokens with highest confidence are kept, lowest are re-masked.
        
        Args:
            input_ids: Initial fully masked sequence [batch, seq_len]
                       (prefix tokens should already be unmasked)
            num_steps: Number of denoising steps
            prefix_len: Number of prefix tokens (already unmasked)
            temperature: Sampling temperature
            steering_hook: Optional per-step steering function
            steering_layer: Layer for steering
            collect_activations: Whether to save activations per step
            activation_layers: Which layers to save (if collecting)
            activation_timesteps: Which timesteps to save (if collecting)
        
        Returns:
            Dict with:
                'output_ids': Final denoised sequence [batch, seq_len]
                'activations': Optional dict of {timestep: {layer: tensor}}
                'trajectory': List of intermediate sequences
        """
        batch_size, seq_len = input_ids.shape
        gen_len = seq_len - prefix_len
        
        # Initialize: start with everything masked (after prefix)
        current_ids = input_ids.clone().to(self.device)
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=self.device)
        mask[:, prefix_len:] = True  # Mask all generation positions
        current_ids[:, prefix_len:] = self.mask_token_id
        
        # Results storage
        trajectory = []
        all_activations = {} if collect_activations else None
        
        # Set up activation collection
        if collect_activations and activation_layers:
            self.register_activation_hooks(activation_layers)
        
        if activation_timesteps is None:
            activation_timesteps = list(range(num_steps))
        
        # Denoising schedule: linear unmask schedule
        # At step k, unmask rate = k / num_steps
        for step in range(num_steps):
            # Number of tokens to keep unmasked after this step
            n_unmask = int(gen_len * (step + 1) / num_steps)
            n_unmask = min(n_unmask, gen_len)
            
            # Predict all tokens
            predicted_ids, confidences = self.denoising_step(
                current_ids, mask, temperature,
                steering_hook=steering_hook,
                steering_layer=steering_layer,
            )
            
            # Collect activations at specified timesteps
            if collect_activations and step in activation_timesteps:
                all_activations[step] = {
                    layer: act.clone() 
                    for layer, act in self._activations.items()
                }
            
            # Fill in predictions at masked positions
            filled_ids = current_ids.clone()
            filled_ids[mask] = predicted_ids[mask].to(filled_ids.device)
            
            # Determine which tokens to keep vs. re-mask
            if step < num_steps - 1:
                # Re-mask: keep the n_unmask most confident predictions
                # Only consider generation positions (after prefix)
                gen_confidences = confidences[:, prefix_len:].clone()
                
                # Positions that were already unmasked get high confidence
                already_unmasked = ~mask[:, prefix_len:]
                gen_confidences[already_unmasked] = float('inf')
                
                # Select top-n_unmask confident positions to keep
                _, keep_indices = gen_confidences.topk(n_unmask, dim=-1)
                
                # Create new mask: mask everything, then unmask selected
                new_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=self.device)
                new_mask[:, :prefix_len] = False  # Prefix stays unmasked
                
                for b in range(batch_size):
                    new_mask[b, prefix_len + keep_indices[b]] = False
                
                # Apply new mask
                current_ids = filled_ids.clone()
                current_ids[new_mask] = self.mask_token_id
                mask = new_mask
            else:
                # Last step: keep everything
                current_ids = filled_ids
                mask = torch.zeros_like(mask)
            
            trajectory.append(current_ids.cpu().clone())
        
        # Clean up hooks
        if collect_activations:
            self.clear_hooks()
        
        return {
            'output_ids': current_ids.cpu(),
            'trajectory': trajectory,
            'activations': all_activations,
        }
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        num_steps: int = 30,
        temperature: float = 1.0,
        steering_hook: Optional[Callable] = None,
        steering_layer: Optional[int] = None,
    ) -> str:
        """
        Generate text from a prompt using the denoising loop.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Number of tokens to generate
            num_steps: Denoising steps
            temperature: Sampling temperature
            steering_hook: Optional steering function
            steering_layer: Layer for steering
        
        Returns:
            Generated text (prompt + completion)
        """
        # Tokenize prompt
        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        prefix_len = prompt_ids.shape[1]
        
        # Create sequence with masked generation positions
        gen_ids = torch.full(
            (1, max_new_tokens), 
            self.mask_token_id,
            dtype=torch.long,
        )
        input_ids = torch.cat([prompt_ids, gen_ids], dim=1)
        
        # Run denoising
        result = self.denoising_loop(
            input_ids=input_ids,
            num_steps=num_steps,
            prefix_len=prefix_len,
            temperature=temperature,
            steering_hook=steering_hook,
            steering_layer=steering_layer,
        )
        
        # Decode
        output_text = self.tokenizer.decode(
            result['output_ids'][0], 
            skip_special_tokens=True
        )
        
        return output_text
    
    def get_model_info(self) -> Dict:
        """Get model architecture information."""
        return {
            'model_name': self.model_name,
            'd_model': self.d_model,
            'n_layers': self.n_layers,
            'device': str(self.device),
            'quantized': self.quantize_4bit,
            'mask_token_id': self.mask_token_id,
            'vocab_size': self.config.vocab_size,
        }
