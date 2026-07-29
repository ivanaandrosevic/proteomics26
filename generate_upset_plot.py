import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

file_name = 'complete_proteomics_TwoWayANOVA_stats.csv'
df_master = pd.read_csv(file_name)

fc_threshold = 0.5
p_threshold = 0.05

groups = {
    'IF (Fasting)': (df_master['log2FC_IF_vs_AL'].abs() >= fc_threshold) & (df_master['pvalue_IF_vs_AL'] <= p_threshold),
    'L-Carnitine': (df_master['log2FC_L-Carnitine_vs_AL'].abs() >= fc_threshold) & (df_master['pvalue_L-Carnitine_vs_AL'] <= p_threshold),
    'LPE 18:1': (df_master['log2FC_LPE_vs_AL'].abs() >= fc_threshold) & (df_master['pvalue_LPE_vs_AL'] <= p_threshold),
    'LPC 17:0': (df_master['log2FC_LPC_vs_AL'].abs() >= fc_threshold) & (df_master['pvalue_LPC_vs_AL'] <= p_threshold)
}
labels = list(groups.keys())
df_bits = pd.DataFrame({k: v.astype(int) for k, v in groups.items()})
all_combos = list(product([1, 0], repeat=len(labels)))[:-1]

combo_counts = []
for combo in all_combos:
    mask = True
    for idx, label in enumerate(labels):
        mask = mask & (df_bits[label] == combo[idx])
    count = mask.sum()
    combo_counts.append({'combo': combo, 'count': count})

df_combos = pd.DataFrame(combo_counts)
df_combos = df_combos.sort_values(by='count', ascending=False).reset_index(drop=True)

n_combos = len(df_combos)
counts = df_combos['count'].tolist()
combos_matrix = np.array(df_combos['combo'].tolist())

fig = plt.figure(figsize=(11, 7))
gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.8], hspace=0.08)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax1)

primary_color = '#2B5C8F'
grid_bg_color = '#F5F5F5'
circle_dark = '#1C3A5E'
circle_light = '#D0D7DE'

bars = ax1.bar(range(n_combos), counts, color=primary_color, width=0.6, edgecolor='black', linewidth=0.7)
ax1.set_ylabel('Intersection Size\n(# of Overlapping Proteins)', fontsize=11, fontweight='bold')
ax1.grid(axis='y', linestyle='--', alpha=0.5)

for bar in bars:
    height = bar.get_height()
    ax1.annotate(f'{height}',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3),  
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

ax1.set_title('Proteomics Shared Target Landscape (UpSet Intersection Matrix)\nEvaluating Common and Unique Molecular Signatures Across Four Treatment Groups', 
             fontsize=13, pad=15, fontweight='bold', color='#111111')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

n_sets = len(labels)

for i in range(n_sets):
    ax2.axhline(i, color=grid_bg_color, lw=12, zorder=0)

for col in range(n_combos):
    active_rows = np.where(combos_matrix[col] == 1)[0]

    if len(active_rows) > 1:
        ax2.plot([col, col], [active_rows.min(), active_rows.max()], color=circle_dark, lw=2, zorder=1)

    for row in range(n_sets):
        if combos_matrix[col, row] == 1:
            ax2.plot(col, row, marker='o', markersize=10, color=circle_dark, zorder=2)
        else:
            ax2.plot(col, row, marker='o', markersize=8, color=circle_light, zorder=2)

ax2.set_yticks(range(n_sets))
ax2.set_yticklabels(labels, fontsize=11, fontweight='bold')
ax2.set_xlabel('Specific Treatment Intersection Configurations', fontsize=11, fontweight='bold', labelpad=10)

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
ax2.set_xlim(-0.5, n_combos - 0.5)

plt.tight_layout()
output_filename = 'proteomics_upset_plot.svg'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"The UpSet plot has been generated")