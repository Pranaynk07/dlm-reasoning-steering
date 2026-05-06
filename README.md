# DLM-Scope Extension: SAE Feature Analysis & Steering on DiffuGPT

Extending [DLM-Scope](https://arxiv.org/abs/2602.05859) (Wang et al., ICLR 2026) with contrastive reasoning feature discovery and causal ablation on [DiffuGPT-Medium](https://github.com/HKUNLP/DiffuLLaMA) (355M).

## Results

### SAE Reconstruction
Top-K SAE (d=1024, k=32) trained on Layer 20 residual stream activations. Achieves ~72% explained variance — lower than DLM-Scope's Dream-7B results as expected at this scale, but sufficient for downstream analysis.

### Contrastive Feature Discovery
Welch's t-test with Bonferroni correction (p < 4.88e-5) across CoT vs Direct prompting on GSM8K. Several hundred features show significant differential activation; a subset shows medium effect sizes (Cohen's d 0.21–0.51).

Logit-lens projection of top features:

| Feature | Cohen's d | Top Promoted Tokens | Interpretation |
|---------|-----------|---------------------|----------------|
| F387 | 0.45 | equations, math, calculus, solving | Math/equations |
| F197 | 0.27 | solves, solve, solved, solution | Solution-seeking |
| F895 | 0.27 | calculate, estimate, multiply | Computation |
| F355 | 0.21 | minus, numbers, multiplied, rounding | Arithmetic |
| F164 | 0.51 | Answers, Puzzles, math, Mathematics | Math education |
| F653 | 0.44 | integers, Boolean, digits, combinations | Discrete math |

### Causal Ablation

| Condition | R-Score | Change |
|-----------|---------|--------|
| No ablation | 12.5 | — |
| Ablate CoT features | 10.4 | -17% |
| Ablate random features | 11.4 | -9% |

Removing discovered features causes roughly 2x the degradation of removing random features, suggesting these features are functionally important for reasoning.

### Steering

| Condition | Reasoning Markers | R-Score |
|-----------|-------------------|---------|
| Baseline | 3.8 | 8.4 |
| +Steer (alpha=5) | 4.1 | 9.4 |
| -Steer (alpha=-5) | 3.3 | 10.0 |
| Random control | 3.3 | 7.1 |

Effects are modest at 355M — the methodology validates but results should improve at Dream-7B scale where baseline reasoning is stronger.

## Method

Following DLM-Scope sections 2-4:

1. **Activation collection** — per-token residual stream activations during denoising (Mask-SAE positions)
2. **SAE training** — Top-K with decoder normalization
3. **Contrastive discovery** — Welch's t-test + Bonferroni across CoT/Direct conditions
4. **Feature interpretation** — logit lens (decoder directions projected through LM head)
5. **Causal test** — targeted feature ablation with random controls
6. **Steering** — per-step injection during denoising (Eq. 13-14)

## Usage

Run the Colab notebook:
1. Open `notebooks/dlm_steering_colab.ipynb` in Colab
2. Select T4 GPU runtime
3. Run all cells (~20 min)

Or run the pipeline script:
```bash
python scripts/full_pipeline.py          # full run
python scripts/full_pipeline.py --phase 3  # resume from SAE training
```

## Structure

```
src/
  models/       # DLM wrapper, Top-K SAE
  training/     # SAE trainer
  data/         # GSM8K loader, activation collector
  analysis/     # Contrastive features, evaluation, visualization
  steering/     # Diffusion-time steering
scripts/        # Full pipeline
notebooks/      # Colab notebooks
configs/        # Experiment config
results/        # Outputs (figures, checkpoints, tables)
```

## References

1. Wang et al. (2026). DLM-Scope: Mechanistic Interpretability of DLMs via SAEs. ICLR 2026.
2. Gong et al. (2025). DiffuGPT. ICLR 2025.
3. Gao et al. (2024). Scaling and Evaluating Sparse Autoencoders.
