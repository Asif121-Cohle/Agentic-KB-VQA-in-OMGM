#!/usr/bin/env python3
"""
generate_paper_figures.py
=============================================================================
Generates publication-quality data figures for the IEEE paper:
"Adaptive Agentic Retrieval: Breaking the Rigid Pipeline Bottleneck in
Knowledge-Based Visual Question Answering"

Outputs saved to 'my paper track/paper works/':
1. fig1_failure_taxonomy_and_score_margin.pdf / .png
2. fig2_verifier_diagnostic_and_debiasing.pdf / .png
3. fig3_scaling_matrix_double_dissociation.pdf / .png
=============================================================================
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Matplotlib Publication Aesthetics (IEEE Two-Column Formatting)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 8.5,
    'figure.titlesize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.35,
    'grid.linestyle': '--',
    'grid.linewidth': 0.6,
})

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Data Loading & Preparation
# ---------------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(OUTPUT_DIR, "../.."))
PHASE0_FILE = os.path.join(BASE_DIR, "pre helper/res/phase0_failure_analysis.json")
STEP2_FILE = os.path.join(BASE_DIR, "pre helper/res/result_step2.json")
STEP3_FILE = os.path.join(BASE_DIR, "pre helper/res/step3_predictions.json")

with open(PHASE0_FILE) as f:
    phase0_data = json.load(f)

with open(STEP2_FILE) as f:
    step2_data = json.load(f)

with open(STEP3_FILE) as f:
    step3_data = json.load(f)

# Extract Category-Specific Score Margins Delta = S1 - S2
cat_map = {item['data_id']: item['category'] for item in phase0_data['failures']}
cat_b_qids = [item['data_id'] for item in phase0_data['failures'] if item['category'] == 'Category B']
success_qids = [qid for qid in step2_data if qid not in cat_map]

cat_b_deltas = [step2_data[q]['retrieval_similarities'][0] - step2_data[q]['retrieval_similarities'][1] 
                for q in cat_b_qids if q in step2_data]
success_deltas = [step2_data[q]['retrieval_similarities'][0] - step2_data[q]['retrieval_similarities'][1] 
                  for q in success_qids if q in step2_data]


FIGS_DIR = os.path.join(OUTPUT_DIR, "figs")
os.makedirs(FIGS_DIR, exist_ok=True)


def save_figure(fig, base_filename):
    pdf_path = os.path.join(FIGS_DIR, f"{base_filename}.pdf")
    png_path = os.path.join(FIGS_DIR, f"{base_filename}.png")
    fig.savefig(pdf_path, bbox_inches='tight')
    fig.savefig(png_path, bbox_inches='tight')
    print(f"Saved: {pdf_path} and {png_path}")


# ===========================================================================
# FIGURE 1: Failure Taxonomy Donut & Score-Margin Confidence Signal
# ===========================================================================
def plot_figure_1():
    print("Generating Figure 1: Failure Taxonomy and Score-Margin Distribution...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.2), gridspec_kw={'width_ratios': [1.0, 1.25]})

    # --- Panel A: Donut Chart of Failure Taxonomy ---
    labels = [
        'Baseline Success\n(34.9%, 349)',
        'Cat A: Retrieval Ceiling\n(15.3%, 153)',
        'Cat B: Entity Rerank Confusion\n(20.5%, 205)',
        'Cat C: Section Selection Error\n(20.3%, 203)',
        'Cat D: Generator Reasoning\n(13.7%, 137)'
    ]
    sizes = [34.9, 15.3, 20.5, 20.3, 13.7]
    colors = ['#2E7D32', '#C62828', '#1565C0', '#6A1B9A', '#E65100']
    explode = (0.02, 0.02, 0.06, 0.06, 0.02)  # Explode Cat B and Cat C

    wedges, texts, autotexts = ax1.pie(
        sizes, 
        explode=explode, 
        colors=colors, 
        autopct='%1.1f%%',
        pctdistance=0.77,
        startangle=140,
        wedgeprops=dict(width=0.42, edgecolor='white', linewidth=1.5)
    )
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(7.5)
        autotext.set_weight('bold')

    # Central Callout Text
    ax1.text(0, 0, 'InfoSeek\n1,000 Queries\n\nCat B+C = 40.8%\n(Rigid Bottleneck)',
             ha='center', va='center', fontsize=6.8, weight='bold', color='#263238', linespacing=1.15)

    ax1.set_title('(a) Phase 0 Failure Taxonomy (OMGM)', pad=10)

    # --- Panel B: Score Margin Distribution KDE ---
    sns.kdeplot(success_deltas, ax=ax2, color='#2E7D32', fill=True, alpha=0.35, 
               linewidth=2.0, label=f'Correct Predictions (μ = {np.mean(success_deltas):.4f})')
    sns.kdeplot(cat_b_deltas, ax=ax2, color='#1565C0', fill=True, alpha=0.35, 
               linewidth=2.0, label=f'Cat B Failures (μ = {np.mean(cat_b_deltas):.4f})')

    # Threshold Line tau = 0.025
    tau = 0.025
    ax2.axvline(tau, color='#D32F2F', linestyle='--', linewidth=1.8, label=r'Threshold $\tau = 0.025$')

    # Annotate Ambiguity vs Confident Zone
    ax2.annotate('Ambiguity Zone\n(50.8%, N=508)\nActivates Controller', 
                 xy=(0.012, 18), xytext=(0.012, 23),
                 ha='center', fontsize=7.5, weight='bold', color='#B71C1C',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFEBEE', edgecolor='#EF9A9A', alpha=0.9))

    ax2.annotate('Confident Fast-Path\n(49.2%, N=492)\n0 Compute Overhead', 
                 xy=(0.055, 12), xytext=(0.060, 16),
                 ha='center', fontsize=7.5, weight='bold', color='#1B5E20',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='#E8F5E9', edgecolor='#A5D6A7', alpha=0.9))

    ax2.set_xlabel(r'Score Margin $\Delta = S_1 - S_2$ (Step 2 Blended)', fontsize=9)
    ax2.set_ylabel('Probability Density', fontsize=9)
    ax2.set_xlim(-0.005, 0.12)
    ax2.set_title(r'(b) Score Margin Collapse ($\Delta_{Correct} \to 2.9\times$ drop)', pad=10)
    ax2.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#CFD8DC', framealpha=0.9)

    plt.tight_layout()
    save_figure(fig, "fig1_failure_taxonomy_and_score_margin")
    plt.close()


# ===========================================================================
# FIGURE 2: Zero-Shot Diagnostic vs. Calibrated Symmetric LoRA Verifier
# ===========================================================================
def plot_figure_2():
    print("Generating Figure 2: Verifier Diagnostic & Debiasing...")
    fig, ax = plt.subplots(figsize=(7.6, 3.8))

    metrics = [
        'Pairwise Accuracy\n(Target: >85%)',
        'Positional Primacy Bias\n(|Acc_A - Acc_B|, Target: <2%)',
        'Unparseable / Drift\n(Instruction Collapse)',
        'Symmetric Consistency\n(Order Invariance)'
    ]
    zero_shot_vals = [32.50, 17.50, 47.50, 10.00]
    lora_vals = [90.33, 1.33, 0.00, 88.33]

    x = np.arange(len(metrics))
    width = 0.35

    rects1 = ax.bar(x - width/2, zero_shot_vals, width, label='Zero-Shot LLaVA-1.5-7B (Off-the-shelf)', 
                    color='#78909C', edgecolor='#37474F', linewidth=1)
    rects2 = ax.bar(x + width/2, lora_vals, width, label='Ours: Calibrated Symmetric LoRA Verifier (r=16)', 
                    color='#1E88E5', edgecolor='#0D47A1', linewidth=1)

    # Add values on top of bars
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 2.5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, weight='bold', color='#37474F')

    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 2.5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, weight='bold', color='#0D47A1')

    # Draw 50% random chance line specifically for binary Pairwise Accuracy (Metric 0)
    ax.hlines(50.0, -0.42, 0.42, color='#D32F2F', linestyle=':', linewidth=1.5, alpha=0.85)
    ax.text(0.0, 52.0, 'Random Chance (50.0%)', color='#D32F2F', fontsize=7.2, fontstyle='italic', ha='center')

    # Annotate key jumps with clean badge boxes and non-overlapping arrow targets
    ax.annotate('+57.8% Gain\n(Visual Calibration)', 
                xy=(width/2, 96.5), xytext=(width/2, 107.5),
                ha='center', fontsize=7.2, weight='bold', color='#0D47A1',
                bbox=dict(boxstyle='round,pad=0.22', facecolor='#E3F2FD', edgecolor='#90CAF9', alpha=0.9),
                arrowprops=dict(arrowstyle='->', color='#0D47A1', lw=1.2))

    ax.annotate('13.2x Bias Drop\n(17.5% -> 1.33%)', 
                xy=(1 + width/2, 8.0), xytext=(1.0, 31.0),
                ha='center', fontsize=7.2, weight='bold', color='#0D47A1',
                bbox=dict(boxstyle='round,pad=0.22', facecolor='#E3F2FD', edgecolor='#90CAF9', alpha=0.9),
                arrowprops=dict(arrowstyle='->', color='#0D47A1', lw=1.2))

    ax.annotate('100% Strict Token\nProtocol (0.0% Drift)', 
                xy=(2 + width/2, 8.0), xytext=(2.0, 66.0),
                ha='center', fontsize=7.2, weight='bold', color='#0D47A1',
                bbox=dict(boxstyle='round,pad=0.22', facecolor='#E3F2FD', edgecolor='#90CAF9', alpha=0.9),
                arrowprops=dict(arrowstyle='->', color='#0D47A1', lw=1.2))

    ax.set_ylabel('Percentage (%)', fontsize=9.5)
    ax.set_title('Diagnostic Validation: Zero-Shot Foundation VLM vs. Calibrated PEFT Verifier (Held-out 600 Pairs)', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=8, rotation=20, ha='right', rotation_mode='anchor')
    ax.set_ylim(0, 122)
    ax.legend(loc='upper center', bbox_to_anchor=(0.58, 0.98), ncol=1, frameon=True, facecolor='white', edgecolor='#CFD8DC', framealpha=0.95, fontsize=7.8)

    plt.tight_layout()
    save_figure(fig, "fig2_verifier_diagnostic_and_debiasing")
    plt.close()


# ===========================================================================
# FIGURE 3: The 2x2 Capacity & Verification Matrix (Double Dissociation)
# ===========================================================================
def plot_figure_3():
    print("Generating Figure 3: The 2x2 Capacity & Verification Matrix...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.6), gridspec_kw={'width_ratios': [1.0, 1.25]})

    # --- Panel A: VQA Accuracy Across 4 Quadrants ---
    configs = [
        'Baseline (OMGM)',
        '7B LoRA (Cat B)',
        '13B LoRA (Cat B)',
        '7B LoRA (Cat B+C)',
        '13B LoRA (Cat B+C)'
    ]
    accuracies = [34.90, 36.00, 35.60, 35.10, 36.30]
    bar_colors = ['#455A64', '#00897B', '#5C6BC0', '#80CBC4', '#283593']
    hatches = ['', '', '', '//', '//']

    x = np.arange(len(configs))
    bars = ax1.bar(x, accuracies, color=bar_colors, edgecolor='#263238', linewidth=1)
    for b, h in zip(bars, hatches):
        b.set_hatch(h)

    # Annotate Champions
    ax1.annotate('Pareto Optimal\n(+1.10%)', xy=(1, 36.05), xytext=(1, 36.55),
                 ha='center', fontsize=7.5, weight='bold', color='#00695C',
                 arrowprops=dict(arrowstyle='->', color='#00695C', lw=1.2))

    ax1.annotate('Capacity Champ\n(+1.40%)', xy=(4, 36.35), xytext=(4, 36.75),
                 ha='center', fontsize=7.5, weight='bold', color='#1A237E',
                 arrowprops=dict(arrowstyle='->', color='#1A237E', lw=1.2))

    ax1.annotate('Context\nCrowding\n(+0.20%)', xy=(3, 35.15), xytext=(3, 35.65),
                 ha='center', fontsize=7, color='#D84315',
                 arrowprops=dict(arrowstyle='->', color='#D84315', lw=1.0))

    # Baseline horizontal dotted line
    ax1.axhline(34.90, color='#C62828', linestyle='--', linewidth=1.2, alpha=0.8, label='Baseline (34.90%)')

    ax1.set_ylabel('InfoSeek VQA Accuracy (%)', fontsize=9.5)
    ax1.set_ylim(34.0, 37.2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, fontsize=8, rotation=25, ha='right', rotation_mode='anchor')
    ax1.set_title('(a) VQA Accuracy Scaling', pad=10)
    ax1.legend(loc='lower left', fontsize=7.5, frameon=True, facecolor='white', edgecolor='#CFD8DC', framealpha=0.9)

    # --- Panel B: Recovery vs Degradation Dynamics ---
    models = ['7B (Cat B)', '13B (Cat B)', '7B (Cat B+C)', '13B (Cat B+C)']
    recoveries = [27, 27, 33, 44]
    degradations = [16, 20, 31, 30]
    net_changes = [11, 7, 2, 14]
    win_rates = [62.79, 57.45, 51.56, 59.46]

    x2 = np.arange(len(models))
    w = 0.28

    r1 = ax2.bar(x2 - w, recoveries, w, label='Gross Recoveries (Wrong -> Right)', color='#2E7D32', edgecolor='#1B5E20')
    r2 = ax2.bar(x2, degradations, w, label='Degradations (Right -> Wrong)', color='#D32F2F', edgecolor='#B71C1C')
    r3 = ax2.bar(x2 + w, net_changes, w, label='Net Gain (Recoveries - Degradations)', color='#1976D2', edgecolor='#0D47A1')

    # Add Net numbers and win-rates on top
    for i in range(len(models)):
        ax2.annotate(f'+{net_changes[i]}\n({win_rates[i]:.1f}%)',
                     xy=(x2[i] + w, net_changes[i]),
                     xytext=(0, 2), textcoords="offset points",
                     ha='center', va='bottom', fontsize=7.5, weight='bold', color='#0D47A1')

    # Separation line between Cat B and Cat B+C
    ax2.axvline(1.5, color='#B0BEC5', linestyle=':', linewidth=1.5)
    ax2.text(0.5, 53.5, 'Category B Only\n(Focused Reranking)', ha='center', va='center', fontsize=7.5, weight='bold', color='#263238',
             bbox=dict(boxstyle='round,pad=0.28', facecolor='#ECEFF1', edgecolor='#CFD8DC', alpha=0.95))
    ax2.text(2.5, 53.5, 'Category B + C\n(Multi-Section Context)', ha='center', va='center', fontsize=7.5, weight='bold', color='#263238',
             bbox=dict(boxstyle='round,pad=0.28', facecolor='#ECEFF1', edgecolor='#CFD8DC', alpha=0.95))

    ax2.set_ylabel('Number of Queries (out of 1,000)', fontsize=9.5)
    ax2.set_ylim(0, 58)
    ax2.set_yticks([0, 10, 20, 30, 40, 50])
    ax2.set_xticks(x2)
    ax2.set_xticklabels(models, fontsize=8, rotation=25, ha='right', rotation_mode='anchor')
    ax2.set_title('(b) Query Recovery vs. Degradation Dynamics', pad=10)
    ax2.legend(loc='upper left', bbox_to_anchor=(0.01, 0.85), fontsize=7.2, frameon=True, facecolor='white', edgecolor='#CFD8DC', framealpha=0.95)

    plt.tight_layout()
    save_figure(fig, "fig3_scaling_matrix_double_dissociation")
    plt.close()


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_figure_1()
    plot_figure_2()
    plot_figure_3()
    print("\nAll publication data figures generated successfully in 'my paper track/paper works/'!")
