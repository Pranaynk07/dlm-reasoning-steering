"""Visualization for DLM steering experiments. Dark theme, publication-quality."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

COLORS = {
    'primary': '#4FC3F7', 'secondary': '#FF8A65',
    'positive': '#81C784', 'negative': '#E57373',
    'neutral': '#90A4AE', 'accent': '#CE93D8',
    'bg': '#1A1A2E', 'fg': '#EAEAEA', 'grid': '#2D2D44',
}


def setup_style():
    plt.style.use('dark_background')
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 12,
        'axes.titlesize': 16, 'axes.labelsize': 13,
        'figure.dpi': 300,
        'figure.facecolor': COLORS['bg'], 'axes.facecolor': COLORS['bg'],
        'text.color': COLORS['fg'], 'axes.labelcolor': COLORS['fg'],
        'xtick.color': COLORS['fg'], 'ytick.color': COLORS['fg'],
        'axes.grid': True, 'grid.alpha': 0.3, 'grid.color': COLORS['grid'],
        'savefig.bbox': 'tight', 'savefig.facecolor': COLORS['bg'],
    })


def save_figure(fig, name, save_dir="results/figures"):
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / f"{name}.png", dpi=300, bbox_inches='tight')
    fig.savefig(path / f"{name}.pdf", bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {name}")


def plot_differential_heatmap(contrastive_results, top_n=30, save_dir="results/figures"):
    setup_style()
    reasoning_features = contrastive_results['reasoning_features'][:top_n]
    if not reasoning_features:
        return

    cot_means = np.array([contrastive_results['cot_means'][f] for f in reasoning_features])
    direct_means = np.array([contrastive_results['direct_means'][f] for f in reasoning_features])
    effects = np.array([contrastive_results['effect_sizes'][f] for f in reasoning_features])

    fig, axes = plt.subplots(1, 3, figsize=(16, 8), gridspec_kw={'width_ratios': [2, 2, 1]})
    labels = [f"F{f}" for f in reasoning_features]

    axes[0].barh(range(len(reasoning_features)), cot_means, color=COLORS['primary'], alpha=0.8)
    axes[0].set_yticks(range(len(reasoning_features)))
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].set_xlabel("Mean Activation")
    axes[0].set_title("CoT Activation", fontsize=14)
    axes[0].invert_yaxis()

    axes[1].barh(range(len(reasoning_features)), direct_means, color=COLORS['secondary'], alpha=0.8)
    axes[1].set_yticks(range(len(reasoning_features)))
    axes[1].set_yticklabels([])
    axes[1].set_xlabel("Mean Activation")
    axes[1].set_title("Direct Activation", fontsize=14)
    axes[1].invert_yaxis()

    colors = [COLORS['positive'] if e > 0.5 else COLORS['primary'] for e in effects]
    axes[2].barh(range(len(reasoning_features)), effects, color=colors, alpha=0.8)
    axes[2].set_yticks(range(len(reasoning_features)))
    axes[2].set_yticklabels([])
    axes[2].set_xlabel("Cohen's d")
    axes[2].set_title("Effect Size", fontsize=14)
    axes[2].axvline(x=0.5, color=COLORS['neutral'], linestyle='--', alpha=0.5, label='Medium')
    axes[2].legend(fontsize=9)
    axes[2].invert_yaxis()

    fig.suptitle("Reasoning-Associated SAE Features", fontsize=16, y=1.02)
    plt.tight_layout()
    save_figure(fig, "fig1_differential_heatmap", save_dir)


def plot_accuracy_vs_alpha(alpha_evaluations, baseline_accuracy, save_dir="results/figures"):
    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    alphas = [e['alpha'] for e in alpha_evaluations]
    accuracies = [e['steered_accuracy']['accuracy'] for e in alpha_evaluations]
    reasoning_scores = [e['mean_reasoning_score_steered'] for e in alpha_evaluations]

    ax1.axhline(y=baseline_accuracy, color=COLORS['neutral'], linestyle='--', alpha=0.7, label=f'Baseline ({baseline_accuracy:.1%})')
    ax1.plot(alphas, accuracies, 'o-', color=COLORS['primary'], linewidth=2, markersize=8, label='Steered')
    ax1.fill_between(alphas, baseline_accuracy, accuracies, alpha=0.1, color=COLORS['primary'])
    ax1.set_xlabel("Steering Strength (alpha)")
    ax1.set_ylabel("GSM8K Accuracy")
    ax1.set_title("Accuracy vs Steering Strength")
    ax1.legend()
    ax1.set_xscale('log')

    baseline_rs = alpha_evaluations[0].get('mean_reasoning_score_baseline', 0)
    ax2.axhline(y=baseline_rs, color=COLORS['neutral'], linestyle='--', alpha=0.7, label='Baseline')
    ax2.plot(alphas, reasoning_scores, 's-', color=COLORS['accent'], linewidth=2, markersize=8, label='Steered')
    ax2.set_xlabel("Steering Strength (alpha)")
    ax2.set_ylabel("Reasoning Score")
    ax2.set_title("Reasoning Quality vs Steering Strength")
    ax2.legend()
    ax2.set_xscale('log')

    plt.tight_layout()
    save_figure(fig, "fig2_accuracy_vs_alpha", save_dir)


def plot_layer_comparison(layer_evaluations, save_dir="results/figures"):
    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    layers = sorted(layer_evaluations.keys())
    acc_deltas = [layer_evaluations[l]['accuracy_delta'] for l in layers]
    steering_scores = [layer_evaluations[l]['steering_score'] for l in layers]

    colors = [COLORS['primary'] if d > 0 else COLORS['negative'] for d in acc_deltas]
    ax1.bar([f"L{l}" for l in layers], acc_deltas, color=colors, alpha=0.8)
    ax1.axhline(y=0, color=COLORS['fg'], linewidth=0.5)
    ax1.set_ylabel("Accuracy delta")
    ax1.set_title("Accuracy Change by Layer")

    ax2.bar([f"L{l}" for l in layers], steering_scores, color=COLORS['accent'], alpha=0.8)
    ax2.axhline(y=0, color=COLORS['fg'], linewidth=0.5)
    ax2.set_ylabel("Steering Score")
    ax2.set_title("Steering Score by Layer")

    plt.tight_layout()
    save_figure(fig, "fig3_layer_comparison", save_dir)


def plot_generation_examples(baseline_outputs, steered_outputs, questions, n_examples=4, save_dir="results/figures"):
    setup_style()
    n = min(n_examples, len(baseline_outputs))
    fig, axes = plt.subplots(n, 2, figsize=(16, 4 * n))
    if n == 1:
        axes = axes.reshape(1, 2)

    for i in range(n):
        for j, (outputs, title, color) in enumerate([
            (baseline_outputs, "Baseline", COLORS['neutral']),
            (steered_outputs, "Steered (alpha=2.0)", COLORS['primary']),
        ]):
            ax = axes[i, j]
            ax.text(0.05, 0.95, f"Q: {questions[i][:80]}...",
                    transform=ax.transAxes, fontsize=9, verticalalignment='top',
                    fontweight='bold', color=COLORS['fg'])
            ax.text(0.05, 0.75, outputs[i][:200],
                    transform=ax.transAxes, fontsize=8, verticalalignment='top',
                    color=color, wrap=True)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
            if i == 0:
                ax.set_title(title, fontsize=13, pad=10)

    fig.suptitle("Baseline vs Steered Generation", fontsize=16, y=1.01)
    plt.tight_layout()
    save_figure(fig, "fig4_generation_examples", save_dir)


def plot_feature_trajectories(timestep_activations, reasoning_features, top_n=10, save_dir="results/figures"):
    setup_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    timesteps = sorted(timestep_activations.keys())
    cmap = plt.cm.viridis(np.linspace(0, 0.9, min(top_n, len(reasoning_features))))

    for i, f_idx in enumerate(reasoning_features[:top_n]):
        acts = [
            float(timestep_activations[t][f_idx])
            if f_idx < len(timestep_activations[t]) else 0
            for t in timesteps
        ]
        ax.plot(timesteps, acts, '-o', color=cmap[i], linewidth=1.5, markersize=4, alpha=0.8, label=f"F{f_idx}")

    ax.set_xlabel("Denoising Step")
    ax.set_ylabel("Mean Feature Activation")
    ax.set_title("Reasoning Features Across Denoising")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    save_figure(fig, "fig5_feature_trajectories", save_dir)


def plot_results_summary(evaluations, save_dir="results/figures"):
    setup_style()
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')

    headers = ['Condition', 'Accuracy', 'Delta', 'Reasoning Score', 'Steering Score']
    rows = []
    for name, data in evaluations.items():
        acc = data.get('steered_accuracy', data.get('baseline_accuracy', {}))
        rows.append([
            name, f"{acc.get('accuracy', 0):.1%}",
            f"{data.get('accuracy_delta', 0):+.1%}",
            f"{data.get('mean_reasoning_score_steered', data.get('mean_reasoning_score_baseline', 0)):.3f}",
            f"{data.get('steering_score', 0):.3f}",
        ])

    table = ax.table(cellText=rows, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    for j in range(len(headers)):
        table[0, j].set_facecolor(COLORS['primary'])
        table[0, j].set_text_props(color='white', fontweight='bold')
    for i in range(1, len(rows) + 1):
        bg = '#2A2A3E' if i % 2 == 0 else '#1E1E32'
        for j in range(len(headers)):
            table[i, j].set_facecolor(bg)
            table[i, j].set_text_props(color=COLORS['fg'])

    ax.set_title("Results Summary", fontsize=16, pad=20)
    save_figure(fig, "fig6_results_summary", save_dir)


def generate_all_figures(results_dir="results"):
    setup_style()
    figures_dir = Path(results_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Figures dir: {figures_dir}")
