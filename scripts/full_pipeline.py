#!/usr/bin/env python3
"""
Full experiment pipeline for DLM-Scope extension.
Runs: model loading -> activation collection -> SAE training ->
      contrastive discovery -> steering experiments -> visualization.

Designed for Colab (T4 GPU) with Drive checkpointing.

Usage:
    %run scripts/full_pipeline.py
    %run scripts/full_pipeline.py --phase 3  # resume from SAE training
"""

import os, sys, gc, json, time, argparse, logging
from pathlib import Path


def phase0_setup():
    """Environment setup and dependency check."""
    print("=" * 60)
    print("Phase 0: Setup")
    print("=" * 60)

    IN_COLAB = False
    try:
        import google.colab
        IN_COLAB = True
    except ImportError:
        pass

    if IN_COLAB:
        print("Running on Colab")
        os.system("pip install -q transformers accelerate datasets scipy seaborn einops tqdm pyyaml")
        from google.colab import drive
        drive.mount('/content/drive')
        SAVE_DIR = "/content/drive/MyDrive/dlm_steering"
    else:
        print("Running locally")
        SAVE_DIR = "results/checkpoints"

    os.makedirs(SAVE_DIR, exist_ok=True)

    import torch
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"GPU: {gpu} ({mem:.1f} GB)")
    else:
        print("No GPU -- will be slow")

    return SAVE_DIR, IN_COLAB


def phase1_load_model(save_dir):
    """Load and verify DiffuGPT-Medium."""
    print(f"\n{'='*60}\nPhase 1: Load Model\n{'='*60}")
    import torch
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.models.dlm_wrapper import DLMWrapper
    dlm = DLMWrapper(
        model_name="diffusionfamily/diffugpt-m",
        base_model_name="gpt2-medium",
        cache_dir=os.path.join(save_dir, "model_cache"),
    )
    print(f"Model: {dlm.get_model_info()}")

    # Quick sanity check
    out = dlm.generate("The capital of France is", max_new_tokens=32, num_steps=16, temperature=0.8)
    print(f"Test generation: {out[:100]}")

    # Verify hooks work
    dlm.register_activation_hooks([4, 12, 20])
    test_ids = dlm.tokenizer.encode("test", return_tensors="pt")
    dlm.forward_pass(test_ids)
    for idx, act in dlm.get_activations().items():
        print(f"  Layer {idx}: {act.shape}")
    dlm.clear_hooks()

    with open(os.path.join(save_dir, "phase1_complete.json"), 'w') as f:
        json.dump(dlm.get_model_info(), f, indent=2)
    return dlm


def phase2_collect_activations(dlm, save_dir):
    print(f"\n{'='*60}\nPhase 2: Collect Activations\n{'='*60}")
    from src.data.gsm8k_loader import GSM8KLoader
    from src.data.activation_collector import ActivationCollector

    gsm8k = GSM8KLoader(split="test", n_problems=200, cache_dir=save_dir)
    discovery_set, eval_set = gsm8k.split_discovery_eval(discovery_frac=0.5)
    print(f"Discovery: {len(discovery_set)}, Eval: {len(eval_set)}")

    target_layers = [4, 8, 12, 16, 20]
    collector = ActivationCollector(
        dlm, target_layers,
        save_dir=os.path.join(save_dir, "activations"),
        batch_size=2, denoising_steps=30,
        timestep_samples=[0, 7, 14, 21, 29],
    )

    print("Collecting CoT activations...")
    cot_stats = collector.collect_from_prompts(discovery_set.get_cot_prompts(), max_gen_tokens=128,
                                                collection_name="cot_activations", resume=True)
    print("Collecting Direct activations...")
    direct_stats = collector.collect_from_prompts(discovery_set.get_direct_prompts(), max_gen_tokens=128,
                                                   collection_name="direct_activations", resume=True)

    with open(os.path.join(save_dir, "phase2_complete.json"), 'w') as f:
        json.dump({'n_discovery': len(discovery_set), 'n_eval': len(eval_set),
                   'target_layers': target_layers}, f, indent=2, default=str)
    return collector, discovery_set, eval_set


def phase3_train_sae(collector, save_dir, d_model=1024):
    print(f"\n{'='*60}\nPhase 3: Train SAE\n{'='*60}")
    import torch
    from src.training.sae_trainer import train_sae_for_layer

    primary_layer = 16
    try:
        train_data = collector.get_training_data("cot_activations", primary_layer, max_tokens=500_000)
    except (ValueError, FileNotFoundError) as e:
        print(f"Could not load activations: {e}. Using synthetic data for demo.")
        train_data = torch.randn(10000, d_model)

    print(f"Training data: {train_data.shape}")
    sae = train_sae_for_layer(train_data, d_model, primary_layer,
                               os.path.join(save_dir, "saes"), expansion_factor=4, k=64, n_epochs=5)

    from src.training.sae_trainer import SAETrainer
    trainer = SAETrainer(d_model=d_model, expansion_factor=4, k=64)
    trainer.sae = sae
    try:
        eval_data = collector.get_training_data("direct_activations", primary_layer, max_tokens=50_000)
        ev = trainer.evaluate(eval_data)
        print(f"Eval: EV={ev['explained_variance']:.3f}, L0={ev['l0']:.1f}")
    except (ValueError, FileNotFoundError):
        ev = {}

    with open(os.path.join(save_dir, "phase3_complete.json"), 'w') as f:
        json.dump({'primary_layer': primary_layer, 'eval': ev}, f, indent=2, default=str)
    return sae, primary_layer


def phase4_discover_features(sae, collector, primary_layer, save_dir):
    print(f"\n{'='*60}\nPhase 4: Contrastive Feature Discovery\n{'='*60}")
    import torch
    from src.analysis.contrastive_features import ContrastiveFeatureDiscovery

    discovery = ContrastiveFeatureDiscovery(sae, significance_alpha=0.05,
                                            correction="bonferroni", min_effect_size=0.2)
    try:
        cot_acts = collector.load_activations("cot_activations", primary_layer, 14)
        direct_acts = collector.load_activations("direct_activations", primary_layer, 14)
    except FileNotFoundError:
        print("Using synthetic data for demo")
        cot_acts = torch.randn(5000, sae.d_model) + 0.1
        direct_acts = torch.randn(5000, sae.d_model)

    results = discovery.analyze(cot_acts, direct_acts)
    top_features = discovery.get_top_reasoning_features(n=50)
    print(f"Found {results['n_reasoning']} reasoning features")
    print(f"Top 10: {top_features[:10]}")

    for f in top_features[:5]:
        s = discovery.get_feature_summary(f)
        print(f"  F{f}: d={s['effect_size']:.3f}, p={s['p_value']:.2e}")

    discovery.save(os.path.join(save_dir, "feature_discovery"))
    with open(os.path.join(save_dir, "phase4_complete.json"), 'w') as f:
        json.dump({'n_reasoning': results['n_reasoning'], 'top_50': top_features}, f, indent=2)
    return top_features, results


def phase5_steering_experiments(dlm, sae, reasoning_features, primary_layer, eval_set, save_dir):
    print(f"\n{'='*60}\nPhase 5: Steering Experiments\n{'='*60}")
    from src.steering.diffusion_steerer import SteeringExperiment
    from src.analysis.steering_evaluator import evaluate_experiment, evaluate_alpha_sweep, print_evaluation_summary, save_evaluation

    experiment = SteeringExperiment(dlm, sae, reasoning_features[:50], primary_layer)
    prompts = eval_set.get_cot_prompts()[:50]
    gen_kw = {'max_new_tokens': 128, 'num_steps': 30, 'temperature': 0.8}

    evals = {}

    print("[E1] Baseline...")
    baseline = experiment.run_baseline(prompts, **gen_kw)

    print("[E2] Positive steering (alpha=2)...")
    steered = experiment.run_steered(prompts, alpha=2.0, condition_name="positive_2.0", **gen_kw)
    evals["positive_2.0"] = evaluate_experiment(baseline, steered, "Positive (alpha=2.0)")
    print_evaluation_summary(evals["positive_2.0"])

    print("[E3] Negative steering (alpha=-2)...")
    neg = experiment.run_steered(prompts, alpha=-2.0, condition_name="negative_-2.0", **gen_kw)
    evals["negative_-2.0"] = evaluate_experiment(baseline, neg, "Negative (alpha=-2.0)")

    print("[E4] Random control...")
    rand = experiment.run_random_control(prompts[:20], n_features=50, n_random_sets=3, alpha=2.0, **gen_kw)
    evals["random_control"] = evaluate_experiment(baseline[:20], rand[:20], "Random Control")

    print("[E5] Alpha sweep...")
    alpha_res = experiment.run_alpha_sweep(prompts[:30], alpha_values=[0.5, 1.0, 2.0, 5.0, 10.0], **gen_kw)
    alpha_evals = evaluate_alpha_sweep(baseline[:30], alpha_res)
    for ae in alpha_evals:
        evals[f"alpha_{ae['alpha']}"] = ae

    results_dir = os.path.join(save_dir, "experiment_results")
    os.makedirs(results_dir, exist_ok=True)
    save_evaluation(evals, results_dir)

    with open(os.path.join(results_dir, "baseline_generations.json"), 'w') as f:
        json.dump(baseline, f, indent=2, default=str)
    with open(os.path.join(results_dir, "steered_generations.json"), 'w') as f:
        json.dump(steered, f, indent=2, default=str)

    return evals, baseline, steered, alpha_evals


def phase6_visualize(contrastive_results, evals, baseline, steered, alpha_evals, reasoning_features, save_dir):
    print(f"\n{'='*60}\nPhase 6: Visualization\n{'='*60}")
    from src.analysis.feature_visualizer import (
        plot_differential_heatmap, plot_accuracy_vs_alpha,
        plot_generation_examples, plot_results_summary, setup_style,
    )

    fig_dir = os.path.join(save_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    plot_differential_heatmap(contrastive_results, top_n=25, save_dir=fig_dir)
    if alpha_evals:
        baseline_acc = alpha_evals[0].get('baseline_accuracy', {}).get('accuracy', 0)
        plot_accuracy_vs_alpha(alpha_evals, baseline_acc, save_dir=fig_dir)
    if baseline and steered:
        n = min(4, len(baseline), len(steered))
        plot_generation_examples(
            [r.get('generated', r.get('text', '')) for r in baseline[:n]],
            [r.get('generated', r.get('text', '')) for r in steered[:n]],
            [r.get('prompt', '')[:100] for r in baseline[:n]],
            n_examples=n, save_dir=fig_dir,
        )
    plot_results_summary(evals, save_dir=fig_dir)
    print(f"Figures saved to {fig_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=int, default=0, help='Start from phase N')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    t0 = time.time()

    save_dir, _ = phase0_setup()

    if args.phase <= 1:
        dlm = phase1_load_model(save_dir)
    else:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.models.dlm_wrapper import DLMWrapper
        dlm = DLMWrapper(model_name="diffusionfamily/diffugpt-m", base_model_name="gpt2-medium",
                         cache_dir=os.path.join(save_dir, "model_cache"))

    if args.phase <= 2:
        collector, discovery_set, eval_set = phase2_collect_activations(dlm, save_dir)
    if args.phase <= 3:
        sae, primary_layer = phase3_train_sae(collector if args.phase <= 2 else None, save_dir, d_model=dlm.d_model)
    if args.phase <= 4:
        reasoning_features, contrastive_results = phase4_discover_features(
            sae, collector if args.phase <= 2 else None, primary_layer, save_dir)
    if args.phase <= 5:
        evals, baseline, steered, alpha_evals = phase5_steering_experiments(
            dlm, sae, reasoning_features, primary_layer, eval_set, save_dir)
    if args.phase <= 6:
        phase6_visualize(contrastive_results, evals, baseline, steered, alpha_evals, reasoning_features, save_dir)

    print(f"\nDone in {(time.time()-t0)/60:.1f} min. Results: {save_dir}")


if __name__ == "__main__":
    main()
