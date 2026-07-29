import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.rcParams['svg.fonttype'] = 'none'

data_file = 'csv2026-10_report.pg_matrix.csv'
stats_file = 'complete_proteomics_TwoWayANOVA_stats.csv'

df_matrix = pd.read_csv(data_file)
df_stats = pd.read_csv(stats_file)

group_cols = {
    'Naive': ['1.1', '1.2', '1.3', '1.4'],
    'ad lib.': ['4.1', '4.2', '4.3', '4.4'],
    'IF': ['6.1', '6.2', '6.3', '6.4'],
}
all_samples = sum(group_cols.values(), [])

df_clean = df_matrix.dropna(subset=all_samples).copy()
for col in all_samples:
  df_clean[col] = pd.to_numeric(
      df_clean[col].astype(str).str.replace("'", ''), errors='coerce'
  )
df_clean = df_clean.dropna(subset=all_samples).copy()

df_clean['gene_name'] = (
    df_clean['T: Genes']
    .fillna(df_clean['T: Protein.Group'])
    .apply(lambda x: str(x).split(';')[0])
)

merged = pd.merge(
    df_clean,
    df_stats[['gene_name', 'ANOVA_p_InteractionEffect']],
    on='gene_name',
    how='inner',
)
top_features = merged.sort_values(by='ANOVA_p_InteractionEffect').head(30)

heatmap_data = top_features.set_index('gene_name')[all_samples]

heatmap_zscore = heatmap_data.apply(
    lambda row: (row - row.mean()) / row.std(), axis=1
)

col_colors = ['#d4d4d4'] * 4 + ['#323232'] * 4 + ['#034e61'] * 4

g = sns.clustermap(
    heatmap_zscore,
    cmap="RdBu_r",
    vmin=-1.5,
    vmax=1.5,
    col_cluster=False,
    row_cluster=True,
    col_colors=col_colors,
    figsize=(8, 10),
    dendrogram_ratio=(0.15, 0.02),
    cbar_pos=(1.02, 0.35, 0.03, 0.3),
    cbar_kws={'label': 'Z-score'},
)

g.ax_heatmap.set_xticklabels([])
g.ax_heatmap.set_xlabel('')
g.ax_heatmap.set_ylabel('Gene Symbol', fontweight='bold')

legend_patches = [
    mpatches.Patch(color='#d4d4d4', label='Naive'),
    mpatches.Patch(color='#323232', label=r'$\mathit{ad.\ lib}\ +/+$'),
    mpatches.Patch(color='#034e61', label='IF +/+'),
]

g.ax_heatmap.legend(
    handles=legend_patches,
    bbox_to_anchor=(1.25, 1.05),
    loc='upper left',
    frameon=False,
    fontsize=11,
)

output_svg = 'heatmap_naive_al_if.svg'
plt.savefig(output_svg, format='svg', bbox_inches='tight')
plt.close()
