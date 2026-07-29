import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib import rcParams

FILE_A = "GO_UP_in_IF.txt"
LABEL_A = "up in IF"
COLOR_A = "#0f6478"

FILE_B = "GP_UP_in_AL.txt"
LABEL_B = "up in ad lib."
COLOR_B = "#1a1a1a"

OUTPUT_PREFIX = "Proteomics_IFvsAL_Enrichr_Plot"

PADJ_THRESHOLD = 0.05
N_TERMS_PER_SIDE = 10
SORT_BY = "Adjusted P-value"

rcParams['font.family'] = "Arial"
rcParams['svg.fonttype'] = 'none'
rcParams['font.size'] = 12

def load_and_prep(path, direction_label, is_top=True):
    df = pd.read_csv(path, sep='\t')
    df = df[df['Adjusted P-value'] < PADJ_THRESHOLD].copy()
    df['neglog10padj'] = -np.log10(df['Adjusted P-value'].replace(0, 1e-300))
    
    if SORT_BY == "Combined Score":
        df = df.sort_values('Combined Score', ascending=False)
    else:
        df = df.sort_values('Adjusted P-value', ascending=True)
        
    df = df.head(N_TERMS_PER_SIDE).reset_index(drop=True)
    
    # Strip GO/Reactome IDs
    df['Term_clean'] = df['Term'].str.replace(r'\s*\(GO:\d+\)', '', regex=True)\
                                 .str.replace(r'\s*R-\w+-\d+', '', regex=True)
    
    metric = df['Combined Score'] if SORT_BY == "Combined Score" else df['neglog10padj']
    df['score'] = metric if is_top else -metric
    df['Direction'] = direction_label
    return df

# 1. Load Data
df_top = load_and_prep(FILE_A, LABEL_A, is_top=True)
df_bottom = load_and_prep(FILE_B, LABEL_B, is_top=False)

df_all = pd.concat([df_top, df_bottom], ignore_index=True)
N_total = len(df_all)
df_all['y_pos'] = np.arange(N_total)[::-1]

cmap = LinearSegmentedColormap.from_list("custom_div", [COLOR_B, "#ffffff", COLOR_A])
vmin, vmax = df_all['score'].min(), df_all['score'].max()
norm = plt.Normalize(vmin=vmin, vmax=vmax)

fig, ax = plt.subplots(figsize=(6.0, 7.5), dpi=300)

X_BUBBLE = 0.0
X_TEXT = 0.25
X_CATEGORY = 2.4

cbar_width = 0.10
gradient = np.linspace(vmax, vmin, 256).reshape(-1, 1)
ax.imshow(gradient, aspect='auto', cmap=cmap, norm=norm,
          extent=[X_BUBBLE - cbar_width/2, X_BUBBLE + cbar_width/2, -0.5, N_total - 0.5],
          zorder=1)

y_min_top, y_max_top = df_all[df_all['Direction'] == LABEL_A]['y_pos'].min() - 0.5, df_all[df_all['Direction'] == LABEL_A]['y_pos'].max() + 0.5
y_min_bot, y_max_bot = df_all[df_all['Direction'] == LABEL_B]['y_pos'].min() - 0.5, df_all[df_all['Direction'] == LABEL_B]['y_pos'].max() + 0.5

ax.fill_betweenx([y_min_top, y_max_top], X_CATEGORY, X_CATEGORY + 0.25, color=COLOR_A, alpha=0.2, zorder=1)
ax.fill_betweenx([y_min_bot, y_max_bot], X_CATEGORY, X_CATEGORY + 0.25, color=COLOR_B, alpha=0.2, zorder=1)

ax.text(X_CATEGORY + 0.125, (y_min_top + y_max_top)/2, LABEL_A, rotation=-90, 
        va='center', ha='center', fontweight='bold', fontsize=7.5, color=COLOR_A)
ax.text(X_CATEGORY + 0.125, (y_min_bot + y_max_bot)/2, LABEL_B, rotation=-90, 
        va='center', ha='center', fontweight='bold', fontsize=7.5, color=COLOR_B)

p_vals = df_all['neglog10padj']
sizes = 30 + (p_vals - p_vals.min()) / (p_vals.max() - p_vals.min() + 1e-6) * 120

bubble_colors = [cmap(norm(val)) for val in df_all['score']]

ax.scatter([X_BUBBLE] * N_total, df_all['y_pos'], s=sizes, c=bubble_colors,
           edgecolors='black', linewidth=0.6, zorder=3)

for _, row in df_all.iterrows():
    ax.text(X_TEXT, row['y_pos'], row['Term_clean'], va='center', ha='left', fontsize=7)

ax.text(X_BUBBLE, N_total + 0.3, f"▲ {LABEL_A}", va='center', ha='center', 
        fontweight='bold', fontsize=8.5, color=COLOR_A)
ax.text(X_BUBBLE, -0.8, f"▼ {LABEL_B}", va='center', ha='center', 
        fontweight='bold', fontsize=8.5, color=COLOR_B)

X_LEGEND = -0.8
y_mid = N_total / 2

leg_p_vals = [int(np.floor(p_vals.min())), 
              int(np.round((p_vals.min() + p_vals.max()) / 2)), 
              int(np.ceil(p_vals.max()))]
leg_p_vals = sorted(list(set(leg_p_vals)))

ax.text(X_LEGEND, y_mid + 1.8, "-log$_{10}$(padj)", ha='center', va='center', fontweight='bold', fontsize=7)

for i, val in enumerate(leg_p_vals):
    y_loc = y_mid + 0.8 - (i * 0.9)
    s_val = 30 + (val - p_vals.min()) / (p_vals.max() - p_vals.min() + 1e-6) * 120
    
    # Draw legend bubble
    ax.scatter(X_LEGEND - 0.15, y_loc, s=s_val, color='gray', edgecolors='black', linewidth=0.5, zorder=3)
    # Draw legend label
    ax.text(X_LEGEND + 0.1, y_loc, f"{val}", va='center', ha='left', fontsize=6.5)

ax.set_xlim(-1.2, X_CATEGORY + 0.4)
ax.set_ylim(-1.5, N_total + 1.0)
ax.axis('off')

plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05)
plt.savefig(f'{OUTPUT_PREFIX}.svg', transparent=True, dpi=300)
plt.close()

print(f"Plot saved successfully: {OUTPUT_PREFIX}")