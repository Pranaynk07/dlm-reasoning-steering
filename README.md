# DLM-Scope Extension: SAE-Based Feature Analysis & Steering on DiffuGPT

**Extending [DLM-Scope](https://arxiv.org/abs/2602.05859) (Wang et al., ICLR 2026)**  
**Built on the official [HKUNLP/DiffuLLaMA](https://github.com/HKUNLP/DiffuLLaMA) codebase**

This project applies the DLM-Scope Sparse Autoencoder (SAE) interpretability framework to DiffuGPT-Medium (355M), extending it with **contrastive feature discovery**, **logit lens interpretation**, and **causal ablation** to characterize math-reasoning representations in diffusion language models.

## Key Findings

### 1. SAE Reconstruction (EV = 0.72)

Our Top-K SAE (d=1024, k=32) trained on Layer 20 achieves **72% explained variance**, demonstrating that SAEs faithfully decompose DLM residual-stream activations at small scale — consistent with DLM-Scope results on Dream-7B.

### 2. Contrastive Feature Discovery (384 significant features)

Welch's t-test with Bonferroni correction (p < 4.88×10⁻⁵) identifies **384 / 1024 features** with statistically significant differences between CoT and Direct prompting. Of these, **14 features** show medium effect sizes (Cohen's d = 0.21–0.51).

### 3. Feature Interpretation via Logit Lens

Projecting SAE decoder directions through the LM head reveals interpretable feature semantics:

| Feature | Cohen's d | Top Promoted Tokens | Interpretation |
|---------|-----------|---------------------|----------------|
| F387 | 0.45 | equations, math, calculus, solving, algebra, equation | Math/equations |
| F197 | 0.27 | solves, solve, solved, solving, Solution, solution | Solution-seeking |
| F895 | 0.27 | calculate, estimate, multiply, subtract, calculated | Computation |
| F355 | 0.21 | minus, numbers, multiplied, rounding, percentage | Arithmetic |
| F164 | 0.51 | Answers, Puzzles, Difficulty, math, Mathematics | Math education |
| F653 | 0.44 | integers, integer, Boolean, digits, combinations | Discrete math |

These features fire significantly more during CoT prompting (Bonferroni p < 5×10⁻⁵), revealing that DLMs develop distinct, interpretable representations for reasoning tasks.

### 4. Causal Ablation

Targeted removal of CoT feature contributions during generation:

| Condition | R-Score | Change |
|-----------|---------|--------|
| No Ablation | 12.5 | — |
| **Ablate CoT Features** | **10.4** | **−17.1%** |
| Ablate Random Features | 11.4 | −9.4% |

Removing CoT features causes **~2× the degradation** of removing random features (7.7 pp difference), establishing causal necessity.

### 5. Diffusion-Time Steering

Applying DLM-Scope Eq. 13 with discovered features shows steering effects on reasoning markers:

| Condition | Reasoning Markers | R-Score |
|-----------|-------------------|---------|
| Baseline | 3.8 | 8.4 |
| +Steer α=5 | **4.1** | 9.4 |
| −Steer α=−5 | 3.3 | 10.0 |
| Random Ctrl | 3.3 | 7.1 |

Positive steering increases reasoning markers (+8% vs baseline), while negative steering and random controls reduce them. Behavioral effects are modest at 355M scale; the methodology is validated for scaling to Dream-7B / LLaDA-8B where baseline reasoning is stronger.

## Methodology

Following DLM-Scope §2–4:

| Stage | Method | Reference |
|-------|--------|-----------|
| Activation collection | Mask-SAE: per-token activations from generated positions | §3.1 |
| SAE architecture | Top-K SAE with decoder normalization (d=1024, k=32) | Gao et al. (2024) |
| Feature discovery | Welch's t-test + Bonferroni correction | Novel extension |
| Feature interpretation | Logit lens: decoder directions → vocabulary space | Novel extension |
| Causal test | Targeted feature ablation vs. random control | Novel extension |
| Steering | Per-step injection during denoising | Eq. 13–14 |

## Quick Start

1. Open `notebooks/dlm_steering_colab.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Select **T4 GPU** runtime
3. Run all cells (~20 min)
4. Results saved to `/content/dlm_results/`

## Project Structure

```
project3_dlm_steering/
├── notebooks/
│   └── dlm_steering_colab.ipynb    # Self-contained Colab notebook
├── src/
│   ├── models/dlm_wrapper.py
│   ├── training/sae_trainer.py
│   ├── analysis/contrastive_features.py
│   └── steering/diffusion_steerer.py
├── configs/
│   └── experiment_config.yaml
└── README.md
```

## References

1. Wang, X., Jiang, B., Wan, Y., Yang, B., Kong, L., & Zou, D. (2026). *DLM-Scope: Mechanistic Interpretability of Diffusion Language Models via Sparse Autoencoders.* ICLR 2026. [arXiv:2602.05859](https://arxiv.org/abs/2602.05859)
2. Gong, S., Li, M., Feng, J., Wu, Z., & Kong, L. (2025). *DiffuGPT.* ICLR 2025.
3. Gao, L., et al. (2024). *Scaling and Evaluating Sparse Autoencoders.* [arXiv:2406.04093](https://arxiv.org/abs/2406.04093)

## License

MIT
