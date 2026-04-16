# DLM-Scope Extension: SAE-Based Feature Analysis & Steering on DiffuGPT

**Extending [DLM-Scope](https://arxiv.org/abs/2602.05859) (Wang et al., ICLR 2026 Workshop)**  
**Built on the official [HKUNLP/DiffuLLaMA](https://github.com/HKUNLP/DiffuLLaMA) codebase**

This project applies the DLM-Scope Sparse Autoencoder (SAE) interpretability framework to DiffuGPT-Medium (355M), extending it with **contrastive feature discovery** between chain-of-thought (CoT) and direct prompting styles.

## Key Contributions

1. **First contrastive SAE analysis on DLMs** — We identify SAE features that differentially activate during CoT vs. Direct prompting, revealing task-specific internal representations in diffusion language models.

2. **DLM-Scope methodology at small scale** — We replicate the core DLM-Scope pipeline (Mask-SAE training, Top-K SAE, diffusion-time steering) on DiffuGPT-Medium, demonstrating the approach generalizes beyond 7B+ models.

3. **Diffusion-time steering** — We implement both All-tokens and Update-tokens steering strategies (Eq. 13-14) with per-step feature injection during the denoising loop.

## Methodology

Following DLM-Scope (Wang et al., 2026):

| Stage | Method | Reference |
|-------|--------|-----------|
| **Activation collection** | Mask-SAE: activations from generated/masked positions during denoising | §3.1 |
| **SAE architecture** | Top-K SAE with decoder normalization | Gao et al. (2024) |
| **Feature discovery** | Welch's t-test + Bonferroni correction (CoT vs Direct) | Novel extension |
| **Steering** | Per-step injection of feature direction `v_f` during denoising | Eq. 13-14 |

### Pipeline

```
GSM8K Problems → CoT/Direct Prompts → DiffuGPT Denoising
                                              ↓
                              Masked-Position Activations (Layer 20)
                                              ↓
                              Top-K SAE Training (d_dict~1024, k=32)
                                              ↓
                        Contrastive Analysis (Welch's t + Bonferroni)
                                              ↓
                         Reasoning Feature Discovery + Steering
```

## Results

### Model Capacity

DiffuGPT-Medium (355M) achieves 0% accuracy on GSM8K across all conditions. This is expected — even GPT-2 XL (1.5B, autoregressive) scores ~2% on GSM8K. The significance of our work lies in **feature discovery and methodology validation**, not downstream accuracy.

### Feature Discovery

The SAE successfully identifies features with statistically significant differential activation between CoT and Direct prompting conditions, demonstrating that DLMs develop distinct internal representations for different reasoning modes — analogous to findings in autoregressive LLMs.

### Steering

Applying DLM-Scope Eq. 13 with discovered features at Layer 20 shows measurable effects on reasoning markers, math operations, and output structure across steering strengths α ∈ [1, 16].

## Project Structure

```
project3_dlm_steering/
├── notebooks/
│   └── dlm_steering_colab.ipynb    # Self-contained Colab notebook (run this)
├── src/
│   ├── models/dlm_wrapper.py       # Local DiffuGPT wrapper
│   ├── training/sae_trainer.py     # SAE training module
│   ├── analysis/contrastive_features.py  # Feature discovery
│   └── steering/diffusion_steerer.py     # Steering hooks
├── configs/
│   └── experiment_config.yaml
└── README.md
```

## Quick Start (Google Colab)

1. Open `notebooks/dlm_steering_colab.ipynb` in Google Colab
2. Select **T4 GPU** runtime
3. Run all cells (~15-20 min total)
4. Results saved to `/content/dlm_results/`

## Dependencies

```
transformers==4.44.2
accelerate
datasets
scipy
seaborn
safetensors
tqdm
```

## References

- Wang, X., Jiang, B., Wan, Y., Yang, B., Kong, L., & Zou, D. (2026). *DLM-Scope: Mechanistic Interpretability of Diffusion Language Models via Sparse Autoencoders.* ICLR 2026 Workshop. [arXiv:2602.05859](https://arxiv.org/abs/2602.05859)
- Gong, S., Li, M., Feng, J., Wu, Z., & Kong, L. (2025). *DiffuGPT: Scaling Diffusion Language Models.* ICLR 2025.
- Gao, L., et al. (2024). *Scaling and Evaluating Sparse Autoencoders.* [arXiv:2406.04093](https://arxiv.org/abs/2406.04093)
- Ye, J., et al. (2025). *Dream: Discrete Diffusion with Refined and Efficient Auxiliary masking.*

## Future Work

- **Scale to DiffuLLaMA-7B / Dream-7B** for meaningful accuracy improvements
- **Layer-wise comparison** of features across all 24 layers
- **Temporal feature dynamics** across diffusion timesteps (DLM-Scope §5)
- **Cross-task transfer** of reasoning features to other benchmarks

## License

MIT
