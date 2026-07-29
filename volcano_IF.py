# Differential abundance is computed with a two-sample t-test per protein on
# the log2 intensities, with Benjamini-Hochberg FDR correction
# Install requirements: pip install pandas numpy matplotlib scipy statsmodels adjustText

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # remove this line if you want an interactive window instead
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import stats
from statsmodels.stats.multitest import multipletests
from adjustText import adjust_text

INPUT_FILE = "csv2026-10_report.pg_matrix.csv"
OUTPUT_PREFIX = "Proteomics_Volcano_IFvsAL"

GENE_COL = "T: Genes"
PROTEIN_ID_COL = "T: Protein.Group"

GROUP_A_CODE = '6'
GROUP_A_LABEL = 'IF'
GROUP_B_CODE = '4'
GROUP_B_LABEL = 'ad lib.'

PADJ_THRESHOLD = 0.05
LFC_THRESHOLD = 1
N_LABELS_PER_SIDE = 8
FONT_FAMILY = "Arial"

rcParams['font.family'] = FONT_FAMILY
rcParams['font.size'] = 7
rcParams['pdf.fonttype'] = 42
rcParams['svg.fonttype'] = 'none'

df = pd.read_csv(INPUT_FILE)

cols_A = [c for c in df.columns if '.' in c and c.split('.')[0] == GROUP_A_CODE]
cols_B = [c for c in df.columns if '.' in c and c.split('.')[0] == GROUP_B_CODE]

if not cols_A or not cols_B:
    raise ValueError("Could not find columns for one or both group codes. "
                      "Check GROUP_A_CODE / GROUP_B_CODE against your column-name prefixes.")

print(f"{GROUP_A_LABEL}: {cols_A}")
print(f"{GROUP_B_LABEL}: {cols_B}")

mat_A = df[cols_A].values
mat_B = df[cols_B].values

log2fc = mat_A.mean(axis=1) - mat_B.mean(axis=1)

tstat, pval = stats.ttest_ind(mat_A, mat_B, axis=1, equal_var=False)
pval = np.nan_to_num(pval, nan=1.0)
_, padj, _, _ = multipletests(pval, method='fdr_bh')

gene_names = df[GENE_COL].fillna(df[PROTEIN_ID_COL])

res = pd.DataFrame({
    'gene_name': gene_names,
    'log2FC': log2fc,
    'pvalue': pval,
    'padj': padj,
})
res.to_csv(f'{OUTPUT_PREFIX}_results.csv', index=False)

res['neglog10padj'] = -np.log10(res['padj'].replace(0, 1e-300))

sig_up_A = (res['padj'] < PADJ_THRESHOLD) & (res['log2FC'] > LFC_THRESHOLD)
sig_up_B = (res['padj'] < PADJ_THRESHOLD) & (res['log2FC'] < -LFC_THRESHOLD)
ns = ~(sig_up_A | sig_up_B)

fig, ax = plt.subplots(figsize=(3.0, 2.6))
ax.scatter(res.loc[ns, 'log2FC'], res.loc[ns, 'neglog10padj'],
           s=3, c='#c9c9c9', linewidth=0, alpha=0.6, zorder=1)
ax.scatter(res.loc[sig_up_B, 'log2FC'], res.loc[sig_up_B, 'neglog10padj'],
           s=4, c='#1a1a1a', linewidth=0, zorder=2)
ax.scatter(res.loc[sig_up_A, 'log2FC'], res.loc[sig_up_A, 'neglog10padj'],
           s=4, c='#0f6478', linewidth=0, zorder=2)

top_A = res[sig_up_A].sort_values('padj').head(N_LABELS_PER_SIDE)
top_B = res[sig_up_B].sort_values('padj').head(N_LABELS_PER_SIDE)
texts = []
for _, row in pd.concat([top_A, top_B]).iterrows():
    texts.append(ax.text(row['log2FC'], row['neglog10padj'], str(row['gene_name']),
                          fontsize=5.5, style='italic'))
adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', lw=0.3),
            expand_points=(1.2, 1.2))

ax.axhline(-np.log10(PADJ_THRESHOLD), color='#999999', linestyle='--', linewidth=0.5, zorder=0)
ax.axvline(LFC_THRESHOLD, color='#999999', linestyle='--', linewidth=0.5, zorder=0)
ax.axvline(-LFC_THRESHOLD, color='#999999', linestyle='--', linewidth=0.5, zorder=0)

ax.set_xlabel(f"log$_2$ fold change ({GROUP_A_LABEL} / {GROUP_B_LABEL})", fontsize=7)
ax.set_ylabel('-log$_{10}$ (adj. p-value)', fontsize=7)
ax.tick_params(labelsize=6, width=0.6, length=2.5)
for spine in ax.spines.values():
    spine.set_linewidth(0.6)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.text(0.98, 0.98, f'up in {GROUP_A_LABEL}', transform=ax.transAxes, ha='right', va='top',
        fontsize=6.5, color='#0f6478', fontweight='bold')
ax.text(0.02, 0.98, f'up in {GROUP_B_LABEL}', transform=ax.transAxes, ha='left', va='top',
        fontsize=6.5, color='#1a1a1a', fontweight='bold', style='italic')

plt.tight_layout(pad=0.3)
plt.savefig(f'{OUTPUT_PREFIX}.pdf', transparent=True)
plt.savefig(f'{OUTPUT_PREFIX}.svg', transparent=True)
plt.close()

print(f"Saved {OUTPUT_PREFIX}.pdf, .svg, and {OUTPUT_PREFIX}_results.csv")
print(f"Significant up in {GROUP_A_LABEL}: {sig_up_A.sum()}")
print(f"Significant up in {GROUP_B_LABEL}: {sig_up_B.sum()}")