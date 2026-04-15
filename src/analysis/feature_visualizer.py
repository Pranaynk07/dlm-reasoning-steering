"""
Publication-Quality Visualization for DLM Steering Experiments.

Generates 6 key figures for the technical report and portfolio:
1. Differential activation heatmap (features × timesteps)
2. GSM8K accuracy curves (accuracy vs steering strength α)
3. Layer-wise steering comparison
4. Before/after generation examples (qualitative)
5. Feature activation trajectories across denoising
6. Reasoning feature clustering (t-SNE)

Style: Dark theme, high-DPI, serif fonts — professional research aesthetic.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Colab
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# ============================================================
# Global Style Configuration
# ============================================================

STYLE_CONFIG = {
    'dark_mode': True,
    'dpi': 300,
    'figsize_default': (10, 6),
    'figsize_wide': (14, 6),
    'figsize_tall': (10, 10),
    'font_family': 'serif',
    'font_size': 12,
    'title_size': 16,
    'label_size': 13,
    'colors': {
        'primary': '#4FC3F7',      # Light blue
        'secondary': '#FF8A65',    # Light orange
        'positive': '#81C784',     # Green
        'negative': '#E57373',     # Red
        'neutral': '#90A4AE',      # Gray
        'accent': '#CE93D8',       # Purple
        'bg': '#1A1A2E',           # Dark background
        'fg': '#EAEAEA',           # Light foreground
        'grid': '#2D2D44',         # Grid color
    },
}


def setup_style():
    """Apply global matplotlib style."""
    cfg = STYLE_CONFIG
    
    if cfg['dark_mode']:
        plt.style.use('dark_background')
    
    plt.rcParams.update({
        'font.family': cfg['font_family'],
        'font.size': cfg['font_size'],
        'axes.titlesize': cfg['title_size'],
        'axes.labelsize': cfg['label_size'],
        'figure.dpi': cfg['dpi'],
        'figure.facecolor': cfg['colors']['bg'],
        'axes.facecolor': cfg['colors']['bg'],
        'text.color': cfg['colors']['fg'],
        'axes.labelcolor': cfg['colors']['fg'],
        'xtick.color': cfg['colors']['fg'],
        'ytick.color': cfg['colors']['fg'],
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.color': cfg['colors']['grid'],
        'savefig.bbox': 'tight',
        'savefig.facecolor': cfg['colors']['bg'],
    })


def save_figure(fig, name: str, save_dir: str = "results/figures"):
    """Save figure in multiple formats."""
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(path / f"{name}.png", dpi=300, bbox_inches='tight')
    fig.savefig(path / f"{name}.pdf", bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved figure: {name}")


# ============================================================
# Figure 1: Differential Activation Heatmap
# ============================================================

def plot_differential_heatmap(
    contrastive_results: Dict,
    top_n: int = 30,
    save_dir: str = "results/figures",
):
    """
    Heatmap of differential activation (CoT - Direct) for top features.
    
    Shows which SAE features are most differentially activated 
    during reasoning vs. direct answering.
    """
    setup_style()
    
    effect_sizes = np.array(contrastive_results['effect_sizes'])
    reasoning_features = contrastive_results['reasoning_features'][:top_n]
    
    if not reasoning_features:
        logger.warning("No reasoning features to plot")
        return
    
    # Get data for top features
    cot_means = np.array([contrastive_results['cot_means'][f] for f in reasoning_features])
    direct_means = np.array([contrastive_results['direct_means'][f] for f in reasoning_features])
    effects = np.array([contrastive_results['effect_sizes'][f] for f in reasoning_features])
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 8), 
                              gridspec_kw={'width_ratios': [2, 2, 1]})
    
    # Panel 1: CoT activation
    ax1 = axes[0]
    feature_labels = [f"F{f}" for f in reasoning_features]
    bars1 = ax1.barh(range(len(reasoning_features)), cot_means, 
                      color=STYLE_CONFIG['colors']['primary'], alpha=0.8)
    ax1.set_yticks(range(len(reasoning_features)))
    ax1.set_yticklabels(feature_labels, fontsize=8)
    ax1.set_xlabel("Mean Activation")
    ax1.set_title("CoT Prompt Activation", fontsize=14)
    ax1.invert_yaxis()
    
    # Panel 2: Direct activation
    ax2 = axes[1]
    bars2 = ax2.barh(range(len(reasoning_features)), direct_means,
                      color=STYLE_CONFIG['colors']['secondary'], alpha=0.8)
    ax2.set_yticks(range(len(reasoning_features)))
    ax2.set_yticklabels([])
    ax2.set_xlabel("Mean Activation")
    ax2.set_title("Direct Prompt Activation", fontsize=14)
    ax2.invert_yaxis()
    
    # Panel 3: Effect size
    ax3 = axes[2]
    colors = [STYLE_CONFIG['colors']['positive'] if e > 0.5 else 
              STYLE_CONFIG['colors']['primary'] for e in effects]
    ax3.barh(range(len(reasoning_features)), effects, color=colors, alpha=0.8)
    ax3.set_yticks(range(len(reasoning_features)))
    ax3.set_yticklabels([])
    ax3.set_xlabel("Cohen's d")
    ax3.set_title("Effect Size", fontsize=14)
    ax3.axvline(x=0.5, color=STYLE_CONFIG['colors']['neutral'], 
                linestyle='--', alpha=0.5, label='Medium effect')
    ax3.legend(fontsize=9)
    ax3.invert_yaxis()
    
    fig.suptitle("Reasoning-Associated SAE Features: Contrastive Analysis", 
                 fontsize=16, y=1.02)
    plt.tight_layout()
    
    save_figure(fig, "fig1_differential_heatmap", save_dir)


# ============================================================
# Figure 2: Accuracy vs Steering Strength
# ============================================================

def plot_accuracy_vs_alpha(
    alpha_evaluations: List[Dict],
    baseline_accuracy: float,
    save_dir: str = "results/figures",
):
    """
    Line plot: GSM8K accuracy at different steering strengths (α).
    Shows optimal α and diminishing returns.
    """
    setup_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    alphas = [e['alpha'] for e in alpha_evaluations]
    accuracies = [e['steered_accuracy']['accuracy'] for e in alpha_evaluations]
    reasoning_scores = [e['mean_reasoning_score_steered'] for e in alpha_evaluations]
    
    # Panel 1: Accuracy
    ax1.axhline(y=baseline_accuracy, color=STYLE_CONFIG['colors']['neutral'],
                linestyle='--', alpha=0.7, label=f'Baseline ({baseline_accuracy:.1%})')
    ax1.plot(alphas, accuracies, 'o-', color=STYLE_CONFIG['colors']['primary'],
             linewidth=2, markersize=8, label='Steered')
    ax1.fill_between(alphas, baseline_accuracy, accuracies, 
                      alpha=0.1, color=STYLE_CONFIG['colors']['primary'])
    ax1.set_xlabel("Steering Strength (α)")
    ax1.set_ylabel("GSM8K Accuracy")
    ax1.set_title("Accuracy vs Steering Strength")
    ax1.legend()
    ax1.set_xscale('log')
    
    # Panel 2: Reasoning quality
    baseline_rs = alpha_evaluations[0].get('mean_reasoning_score_baseline', 0)
    ax2.axhline(y=baseline_rs, color=STYLE_CONFIG['colors']['neutral'],
                linestyle='--', alpha=0.7, label='Baseline')
    ax2.plot(alphas, reasoning_scores, 's-', color=STYLE_CONFIG['colors']['accent'],
             linewidth=2, markersize=8, label='Steered')
    ax2.set_xlabel("Steering Strength (α)")
    ax2.set_ylabel("Reasoning Score")
    ax2.set_title("Reasoning Quality vs Steering Strength")
    ax2.legend()
    ax2.set_xscale('log')
    
    fig.suptitle("Effect of Steering Strength on Mathematical Reasoning", 
                 fontsize=16, y=1.02)
    plt.tight_layout()
    
    save_figure(fig, "fig2_accuracy_vs_alpha", save_dir)


# ============================================================
# Figure 3: Layer-wise Comparison
# ============================================================

def plot_layer_comparison(
    layer_evaluations: Dict[int, Dict],
    save_dir: str = "results/figures",
):
    """
    Bar chart comparing steering effectiveness across layers.
    Tests the DLM-Scope finding that deep layers are most effective.
    """
    setup_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    layers = sorted(layer_evaluations.keys())
    acc_deltas = [layer_evaluations[l]['accuracy_delta'] for l in layers]
    steering_scores = [layer_evaluations[l]['steering_score'] for l in layers]
    
    colors = [STYLE_CONFIG['colors']['primary'] if d > 0 
              else STYLE_CONFIG['colors']['negative'] for d in acc_deltas]
    
    # Accuracy delta
    ax1.bar([f"L{l}" for l in layers], acc_deltas, color=colors, alpha=0.8)
    ax1.axhline(y=0, color=STYLE_CONFIG['colors']['fg'], linewidth=0.5)
    ax1.set_ylabel("Accuracy Δ")
    ax1.set_title("Accuracy Change by Layer")
    
    # Steering score
    ax2.bar([f"L{l}" for l in layers], steering_scores, 
            color=STYLE_CONFIG['colors']['accent'], alpha=0.8)
    ax2.axhline(y=0, color=STYLE_CONFIG['colors']['fg'], linewidth=0.5)
    ax2.set_ylabel("Steering Score S(f)")
    ax2.set_title("Steering Score by Layer")
    
    fig.suptitle("Layer-wise Steering Effectiveness", fontsize=16, y=1.02)
    plt.tight_layout()
    
    save_figure(fig, "fig3_layer_comparison", save_dir)


# ============================================================
# Figure 4: Qualitative Examples
# ============================================================

def plot_generation_examples(
    baseline_outputs: List[str],
    steered_outputs: List[str],
    questions: List[str],
    n_examples: int = 4,
    save_dir: str = "results/figures",
):
    """
    Side-by-side comparison of baseline vs steered generations.
    Visual qualitative evidence.
    """
    setup_style()
    
    n = min(n_examples, len(baseline_outputs))
    
    fig, axes = plt.subplots(n, 2, figsize=(16, 4 * n))
    if n == 1:
        axes = axes.reshape(1, 2)
    
    for i in range(n):
        # Baseline
        ax_b = axes[i, 0]
        ax_b.text(0.05, 0.95, f"Q: {questions[i][:80]}...", 
                  transform=ax_b.transAxes, fontsize=9, verticalalignment='top',
                  fontweight='bold', color=STYLE_CONFIG['colors']['fg'])
        ax_b.text(0.05, 0.75, baseline_outputs[i][:200],
                  transform=ax_b.transAxes, fontsize=8, verticalalignment='top',
                  color=STYLE_CONFIG['colors']['neutral'], wrap=True)
        ax_b.set_xlim(0, 1)
        ax_b.set_ylim(0, 1)
        ax_b.axis('off')
        if i == 0:
            ax_b.set_title("Baseline (No Steering)", fontsize=13, pad=10)
        
        # Steered
        ax_s = axes[i, 1]
        ax_s.text(0.05, 0.95, f"Q: {questions[i][:80]}...",
                  transform=ax_s.transAxes, fontsize=9, verticalalignment='top',
                  fontweight='bold', color=STYLE_CONFIG['colors']['fg'])
        ax_s.text(0.05, 0.75, steered_outputs[i][:200],
                  transform=ax_s.transAxes, fontsize=8, verticalalignment='top',
                  color=STYLE_CONFIG['colors']['primary'], wrap=True)
        ax_s.set_xlim(0, 1)
        ax_s.set_ylim(0, 1)
        ax_s.axis('off')
        if i == 0:
            ax_s.set_title("Steered (Reasoning Features α=2.0)", fontsize=13, pad=10)
    
    fig.suptitle("Qualitative Comparison: Baseline vs. Steered Generation", 
                 fontsize=16, y=1.01)
    plt.tight_layout()
    
    save_figure(fig, "fig4_generation_examples", save_dir)


# ============================================================
# Figure 5: Feature Activation Trajectories
# ============================================================

def plot_feature_trajectories(
    timestep_activations: Dict[int, np.ndarray],
    reasoning_features: List[int],
    top_n: int = 10,
    save_dir: str = "results/figures",
):
    """
    Line plot tracking top reasoning feature activations across denoising.
    Shows when reasoning features become active during generation.
    """
    setup_style()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    timesteps = sorted(timestep_activations.keys())
    cmap = plt.cm.viridis(np.linspace(0, 0.9, min(top_n, len(reasoning_features))))
    
    for i, f_idx in enumerate(reasoning_features[:top_n]):
        activations = [
            float(timestep_activations[t][f_idx]) 
            if f_idx < len(timestep_activations[t]) else 0
            for t in timesteps
        ]
        ax.plot(timesteps, activations, '-o', color=cmap[i], 
                linewidth=1.5, markersize=4, alpha=0.8, label=f"F{f_idx}")
    
    ax.set_xlabel("Denoising Timestep")
    ax.set_ylabel("Mean Feature Activation")
    ax.set_title("Reasoning Feature Activation Across Denoising Steps")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    save_figure(fig, "fig5_feature_trajectories", save_dir)


# ============================================================
# Figure 6: Results Summary Table
# ============================================================

def plot_results_summary(
    evaluations: Dict[str, Dict],
    save_dir: str = "results/figures",
):
    """
    Summary table figure with all experiment results.
    """
    setup_style()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')
    
    # Build table data
    headers = ['Condition', 'Accuracy', 'Δ Accuracy', 'Reasoning Score', 'Steering Score S(f)']
    rows = []
    
    for name, eval_data in evaluations.items():
        rows.append([
            name,
            f"{eval_data.get('steered_accuracy', eval_data.get('baseline_accuracy', {})).get('accuracy', 0):.1%}",
            f"{eval_data.get('accuracy_delta', 0):+.1%}",
            f"{eval_data.get('mean_reasoning_score_steered', eval_data.get('mean_reasoning_score_baseline', 0)):.3f}",
            f"{eval_data.get('steering_score', 0):.3f}",
        ])
    
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc='center',
        loc='center',
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Color headers
    for j in range(len(headers)):
        table[0, j].set_facecolor(STYLE_CONFIG['colors']['primary'])
        table[0, j].set_text_props(color='white', fontweight='bold')
    
    # Alternate row colors
    for i in range(1, len(rows) + 1):
        color = '#2A2A3E' if i % 2 == 0 else '#1E1E32'
        for j in range(len(headers)):
            table[i, j].set_facecolor(color)
            table[i, j].set_text_props(color=STYLE_CONFIG['colors']['fg'])
    
    ax.set_title("Experiment Results Summary", fontsize=16, pad=20)
    
    save_figure(fig, "fig6_results_summary", save_dir)


def generate_all_figures(results_dir: str = "results"):
    """Generate all figures from saved results."""
    setup_style()
    logger.info("Generating all publication figures...")
    
    figures_dir = Path(results_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Figures will be saved to: {figures_dir}")
    # Individual figure functions should be called with actual data
    # This is a convenience entry point
