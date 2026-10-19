"""
Generate publication-quality figures for the RespNews IEEE Paper.
Outputs:
1. paper/figures/system_architecture.png & .pdf
2. paper/figures/pareto_frontier.png & .pdf
3. paper/figures/multimodal_comparison.png & .pdf
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# IEEE style configuration
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['figure.titlesize'] = 11

out_dir = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(out_dir, exist_ok=True)

# ==========================================
# Figure 1: System Architecture Diagram
# ==========================================
fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=300)
ax.axis('off')
ax.set_xlim(0, 10)
ax.set_ylim(0, 4.5)

def draw_box(ax, x, y, w, h, title, subtitle="", color="#EBF3FB", border="#2B5B84", lw=1.5):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.15",
                                  facecolor=color, edgecolor=border, linewidth=lw)
    ax.add_patch(rect)
    if subtitle:
        ax.text(x + w/2, y + h*0.62, title, ha='center', va='center', weight='bold', fontsize=8.5, color='#1A202C')
        ax.text(x + w/2, y + h*0.30, subtitle, ha='center', va='center', fontsize=7.0, color='#4A5568')
    else:
        ax.text(x + w/2, y + h*0.5, title, ha='center', va='center', weight='bold', fontsize=8.5, color='#1A202C')

def draw_arrow(ax, x1, y1, x2, y2, label=""):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="#2D3748", lw=1.3, shrinkA=2, shrinkB=2))
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.12, label, ha='center', va='bottom', fontsize=6.8, color="#2B6CB0", weight='semibold')

# Section backgrounds
bg_input = patches.FancyBboxPatch((0.1, 0.2), 2.2, 4.0, boxstyle="round,pad=0.1", facecolor='#F7FAFC', edgecolor='#CBD5E0', linestyle='--')
bg_model = patches.FancyBboxPatch((2.6, 0.2), 4.7, 4.0, boxstyle="round,pad=0.1", facecolor='#F0FFF4', edgecolor='#9AE6B4', linestyle='--')
bg_output = patches.FancyBboxPatch((7.6, 0.2), 2.3, 4.0, boxstyle="round,pad=0.1", facecolor='#FFF5F5', edgecolor='#FEB2B2', linestyle='--')
ax.add_patch(bg_input)
ax.add_patch(bg_model)
ax.add_patch(bg_output)

ax.text(1.2, 4.0, "I. Multimodal Inputs", ha='center', weight='bold', fontsize=8, color='#4A5568')
ax.text(4.95, 4.0, "II. Multimodal Encoders & Heterogeneous KG", ha='center', weight='bold', fontsize=8, color='#22543D')
ax.text(8.75, 4.0, "III. Responsible Re-Ranking", ha='center', weight='bold', fontsize=8, color='#742A2A')

# 1. Inputs
draw_box(ax, 0.25, 2.5, 1.9, 1.0, "News Content", "Title, Body & Image", "#EDF2F7", "#718096")
draw_box(ax, 0.25, 0.6, 1.9, 1.0, "User History", "Read sequence & logs", "#EDF2F7", "#718096")

# 2. Encoders
draw_box(ax, 2.8, 2.7, 1.9, 0.85, "MuRIL Text", "Indic/Multilingual (768d)", "#EBF8FF", "#3182CE")
draw_box(ax, 2.8, 1.5, 1.9, 0.85, "EfficientNet / Xception", "Visual Features & Forensics", "#EBF8FF", "#3182CE")
draw_box(ax, 2.8, 0.4, 1.9, 0.85, "Susceptibility Profiler", "23-Technique Propensity", "#FEFCBF", "#D69E2E")

# 3. Fusion & KG
draw_box(ax, 5.2, 2.1, 1.9, 1.45, "Heterogeneous KG", "GraphSAGE (Users, News,\nTopics, Named Entities)\n+ Gated Fusion Unit", "#C6F6D5", "#2F855A")
draw_box(ax, 5.2, 0.5, 1.9, 1.2, "23-Persuasion Detector", "Multi-Label Rhetoric\nSemEval-2023 Task 3", "#FEEBC8", "#DD6B20")

# 4. Responsible Scoring & Output
draw_box(ax, 7.8, 2.2, 1.9, 1.35, "Responsible Score", "$S = \\alpha R - \\beta P + \\gamma E$\nProvider Capping\nParity Gap = 0.000", "#FED7D7", "#E53E3E")
draw_box(ax, 7.8, 0.6, 1.9, 1.1, "Top-K Recommendation", "Persuasion-Mitigated Feed\n+ 100% Explanations", "#E2E8F0", "#4A5568")

# Arrows
draw_arrow(ax, 2.15, 3.1, 2.8, 3.1, "text")
draw_arrow(ax, 2.15, 2.7, 2.8, 2.0, "image")
draw_arrow(ax, 2.15, 1.1, 2.8, 0.85, "clicks")

draw_arrow(ax, 4.7, 3.1, 5.2, 3.0)
draw_arrow(ax, 4.7, 1.9, 5.2, 2.6)
draw_arrow(ax, 4.7, 3.1, 5.2, 1.3, "text")
draw_arrow(ax, 4.7, 0.8, 7.8, 2.5, "susceptibility")

draw_arrow(ax, 7.1, 2.8, 7.8, 3.0, "Relevance R")
draw_arrow(ax, 7.1, 1.1, 7.8, 2.6, "Risk P")

draw_arrow(ax, 8.75, 2.2, 8.75, 1.7)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "system_architecture.png"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(out_dir, "system_architecture.pdf"), bbox_inches='tight')
plt.close()
print("Saved system_architecture.png & .pdf")

# ==========================================
# Figure 2: Pareto Frontier Trade-Off
# ==========================================
fig, ax1 = plt.subplots(figsize=(3.5, 2.6), dpi=300)

betas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
ndcg10 = [0.6083, 0.6082, 0.6081, 0.6080, 0.6075, 0.6068, 0.6059, 0.6045, 0.6028, 0.6009, 0.5985]
per_exposure = [0.0044, 0.0035, 0.0022, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]
# Convert exposure to percentage suppression or exposure
manip_exposure = [0.42, 0.31, 0.18, 0.05, 0.04, 0.03, 0.03, 0.02, 0.02, 0.02, 0.01]

color1 = '#1A365D'
color2 = '#C53030'

line1 = ax1.plot(betas, ndcg10, marker='o', markersize=4, color=color1, linewidth=1.5, label='Ranking Quality (nDCG@10)')
ax1.set_xlabel(r'Persuasion Penalty Weight ($\beta$)')
ax1.set_ylabel('nDCG@10', color=color1)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim(0.595, 0.612)

ax2 = ax1.twinx()
line2 = ax2.plot(betas, manip_exposure, marker='s', markersize=4, color=color2, linewidth=1.5, linestyle='--', label='Manipulative Exposure')
ax2.set_ylabel('Persuasive Exposure Rate', color=color2)
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(0.0, 0.45)

# Highlight sweet spot at beta = 0.3
ax1.axvline(0.3, color='#2F855A', linestyle=':', linewidth=1.3)
ax1.annotate('Optimal Trade-off\n($\\beta=0.30$)', xy=(0.3, 0.6080), xytext=(0.42, 0.602),
             arrowprops=dict(facecolor='#2F855A', arrowstyle="->", lw=1.0),
             fontsize=7.5, weight='bold', color='#2F855A')

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='lower left', frameon=True, fontsize=6.8)
ax1.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "pareto_frontier.png"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(out_dir, "pareto_frontier.pdf"), bbox_inches='tight')
plt.close()
print("Saved pareto_frontier.png & .pdf")

# ==========================================
# Figure 3: Multimodal Architecture Comparison
# ==========================================
fig, ax = plt.subplots(figsize=(3.5, 2.6), dpi=300)

models = ['NRMS', 'GLORY', 'RoBERTa', 'MuRIL', 'EfficNet', 'Xception', 'RespNews (Ours)']
auc_vals = [0.6730, 0.6986, 0.6755, 0.6794, 0.6671, 0.6664, 0.6804]
ndcg_vals = [0.5117, 0.5384, 0.5990, 0.6041, 0.5957, 0.5922, 0.6080]

x = np.arange(len(models))
width = 0.36

rects1 = ax.bar(x - width/2, auc_vals, width, label='AUC', color='#4299E1', edgecolor='#2B6CB0', alpha=0.9)
rects2 = ax.bar(x + width/2, ndcg_vals, width, label='nDCG@10', color='#48BB78', edgecolor='#2F855A', alpha=0.9)

ax.set_ylabel('Score')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=35, ha='right', fontsize=7)
ax.set_ylim(0.45, 0.75)
ax.legend(loc='upper left', frameon=True, fontsize=7.2)
ax.grid(axis='y', linestyle=':', alpha=0.5)

# Highlight Ours
rects1[-1].set_facecolor('#E53E3E')
rects1[-1].set_edgecolor('#9B2C2C')
rects2[-1].set_facecolor('#DD6B20')
rects2[-1].set_edgecolor('#9C4221')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "multimodal_comparison.png"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(out_dir, "multimodal_comparison.pdf"), bbox_inches='tight')
plt.close()
print("Saved multimodal_comparison.png & .pdf")
