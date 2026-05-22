import dash
from dash import dcc, html, dash_table, Input, Output
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import os
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# DATA LOADING & TWO-WAY ANOVA
# ==============================================================================
print("🚀 Loading Proteomics Matrix...")

data_file = 'Proteomics_data.csv'
if not os.path.exists(data_file):
    raise FileNotFoundError(f"CRITICAL ERROR: Could not find '{data_file}' in this folder. Please verify the filename.")

df = pd.read_csv(data_file)

group_cols = {
    'Naive (Healthy Control)': ['1.1', '1.2', '1.3', '1.4'],
    'AL (Pathologically Primed)': ['4.1', '4.2', '4.3', '4.4'],
    'IF (Intermittent Fasting)': ['6.1', '6.2', '6.3', '6.4'],
    'L-Carnitine': ['7.1', '7.2', '7.3', '7.4'],
    'LPE 18:1': ['8.1', '8.2', '8.3', '8.4'],
    'LPC 17:0': ['9.1', '9.2', '9.3', '9.4']
}
all_samples = sum(group_cols.values(), [])

# --- STEP 1: FORCE ABSOLUTE NUMERIC CLEANING ---
df_clean = df.dropna(subset=all_samples).copy()
for col in all_samples:
    # Coerce errors to NaN to eliminate single quotes or hidden spaces, then drop them
    df_clean[col] = pd.to_numeric(df_clean[col].astype(str).str.replace("'", "").str.strip(), errors='coerce')
df_clean = df_clean.dropna(subset=all_samples).copy()

# Parse tidy gene symbols cleanly
df_clean['gene_name'] = df_clean['T: Genes'].fillna(df_clean['T: Protein.Group']).apply(lambda x: str(x).split(';')[0])

print(f" -> Cleaned Data Matrix successfully! Processing {len(df_clean)} verified protein rows.")

# --- STEP 2: MAP FACTORIAL METADATA ---
sample_design = []
for grp, samples in group_cols.items():
    for s in samples:
        priming = 'Healthy' if 'Naive' in grp else 'Primed'
        if 'Naive' in grp or 'AL' in grp:
            therapy = 'None'
        elif 'IF' in grp:
            therapy = 'Fasting'
        else:
            therapy = 'Lipid_Supplement'
        sample_design.append({'SampleID': s, 'Priming': priming, 'Therapy': therapy})

design_df = pd.DataFrame(sample_design)

# Setup analytics grids
master_stats_df = df_clean[['gene_name', 'T: Protein.Group', 'T: First.Protein.Description']].copy()

# --- STEP 3: PAIRWISE LOG2 FOLD CHANGES VS AL ---
print("Calculating Log2 Fold Changes and T-Tests vs AL baseline...")
al_matrix = df_clean[group_cols['AL (Pathologically Primed)']].values.astype(float)
al_means = np.mean(al_matrix, axis=1)

tx_groups = ['IF (Intermittent Fasting)', 'L-Carnitine', 'LPE 18:1', 'LPC 17:0']
for tx in tx_groups:
    tx_matrix = df_clean[group_cols[tx]].values.astype(float)
    short_name = tx.split(' ')[0]
    master_stats_df[f'log2FC_{short_name}_vs_AL'] = np.mean(tx_matrix, axis=1) - al_means
    
    pvals = []
    for i in range(len(df_clean)):
        # Calculate standard pair t-tests safely
        _, p = stats.ttest_ind(tx_matrix[i], al_matrix[i], equal_var=False)
        pvals.append(p if (not np.isnan(p) and p > 0) else 1.0)
    master_stats_df[f'pvalue_{short_name}_vs_AL'] = pvals

# --- STEP 4: SAFE ROW-BY-ROW TWO-WAY ANOVA ---
print("Running Defensive Ordinary Least Squares (OLS) Two-Way ANOVA Engine...")
p_priming_effect = []
p_therapy_effect = []
p_interaction_effect = []

for idx, row in df_clean.iterrows():
    # Gather expression levels for current protein row configuration
    expr_vals = [float(row[s]) for s in design_df['SampleID']]
    current_model_df = design_df.copy()
    current_model_df['Expression'] = expr_vals
    
    # Verify non-zero variance before optimizing regression matrices
    if np.var(expr_vals) == 0:
        p_priming_effect.append(1.0)
        p_therapy_effect.append(1.0)
        p_interaction_effect.append(1.0)
        continue
        
    try:
        # Fit OLS Model safely
        model = ols('Expression ~ C(Priming) * C(Therapy)', data=current_model_df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        
        p_p = anova_table.loc['C(Priming)', 'PR(>F)']
        p_t = anova_table.loc['C(Therapy)', 'PR(>F)']
        p_i = anova_table.loc['C(Priming):C(Therapy)', 'PR(>F)']
        
        p_priming_effect.append(p_p if not np.isnan(p_p) else 1.0)
        p_therapy_effect.append(p_t if not np.isnan(p_t) else 1.0)
        p_interaction_effect.append(p_i if not np.isnan(p_i) else 1.0)
    except Exception:
        # Fallback values if a singular row matrix is ill-conditioned or mathematically unstable
        p_priming_effect.append(1.0)
        p_therapy_effect.append(1.0)
        p_interaction_effect.append(1.0)

# Inject array outputs back into master data sheets
master_stats_df['ANOVA_p_PrimingEffect'] = p_priming_effect
master_stats_df['ANOVA_p_TherapyEffect'] = p_therapy_effect
master_stats_df['ANOVA_p_InteractionEffect'] = p_interaction_effect
df_clean['anova_pvalue'] = p_interaction_effect

print("Two-Way ANOVA Complete! Exporting compiled master tables...")
os.makedirs('Exported_Plots_2Way', exist_ok=True)
master_stats_df.to_csv('complete_proteomics_TwoWayANOVA_stats.csv', index=False)

print("All systems initialized! Initializing layout dashboard wrapper...")

# ==============================================================================
# 2. DASHBOARD WEB INTERFACE DESIGN
# ==============================================================================
app = dash.Dash(__name__, title="Proteomics Factorial Analyzer", suppress_callback_exceptions=True)
server = app.server

app.layout = html.Div(style={'fontFamily': 'Segoe UI, Arial, sans-serif', 'backgroundColor': '#F4F6F7', 'margin': '0', 'padding': '25px'}, children=[
    
    # Header Control Panel Banner
    html.Div(style={'backgroundColor': '#2A4D69', 'padding': '25px', 'borderRadius': '8px', 'marginBottom': '25px', 'color': 'white', 'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center'}, children=[
        html.Div(children=[
            html.H1("Proteomics Interactive Multi-Arm Explorer", style={'margin': '0', 'fontSize': '28px'}),
            html.P("Factorial Study Design Engine integrated with robust Type-II Two-Way ANOVA modeling", style={'margin': '5px 0 0 0', 'opacity': '0.85'})
        ]),
        html.Button("💾 DOWNLOAD ALL COMPLETED 2-WAY STATS (.CSV)", id="btn-master-download", 
                    style={'backgroundColor': '#E67E22', 'color': 'white', 'border': 'none', 'padding': '12px 20px', 'borderRadius': '6px', 'cursor': 'pointer', 'fontWeight': 'bold'}),
        dcc.Download(id="master-download-tracker")
    ]),
    
    dcc.Tabs(id="dashboard-tabs", value='tab-profile', children=[
        dcc.Tab(label='Single Protein Factor Variance', value='tab-profile', style={'fontWeight': 'bold'}),
        dcc.Tab(label='Treatment vs AL Volcano Plots', value='tab-volcano', style={'fontWeight': 'bold'}),
        dcc.Tab(label='Two-Way ANOVA Signatures Map', value='tab-heatmap', style={'fontWeight': 'bold'}),
        dcc.Tab(label='Multi-Group Overlap Intersect', value='tab-overlap', style={'fontWeight': 'bold'})
    ]),
    
    html.Div(id='tab-window-content', style={'paddingTop': '25px'})
])

# ==============================================================================
# 3. INTERACTIVE CALLBACK BACKEND ARCHITECTURE
# ==============================================================================
@app.callback(Output("master-download-tracker", "data"), Input("btn-master-download", "n_clicks"), prevent_initial_call=True)
def download_complete_statistical_sheet(n_clicks):
    return dcc.send_data_frame(master_stats_df.to_csv, "complete_proteomics_TwoWayANOVA_stats.csv", index=False)

@app.callback(Output('tab-window-content', 'children'), Input('dashboard-tabs', 'value'))
def switch_tabs(active_tab):
    all_genes = sorted(df_clean['gene_name'].unique())
    
    if active_tab == 'tab-profile':
        return html.Div(style={'display': 'flex', 'gap': '20px'}, children=[
            html.Div(style={'flex': '1', 'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '8px'}, children=[
                html.H3("Select Target Protein", style={'marginTop': '0', 'color': '#2A4D69'}),
                dcc.Dropdown(id='profile-dropdown', options=[{'label': g, 'value': g} for g in all_genes], value=all_genes[0], clearable=False),
                html.Br(),
                html.Div(id='profile-stats-card', style={'padding': '15px', 'backgroundColor': '#F8F9FA', 'borderRadius': '6px', 'borderLeft': '4px solid #2A4D69'})
            ]),
            html.Div(style={'flex': '2.5', 'backgroundColor': 'white', 'padding': '15px', 'borderRadius': '8px'}, children=[
                dcc.Graph(id='profile-boxplot')
            ])
        ])
        
    elif active_tab == 'tab-volcano':
        tx_options = [k for k in group_cols.keys() if 'AL' not in k]
        return html.Div(children=[
            html.Div(style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '8px', 'marginBottom': '20px'}, children=[
                html.Label("Select Intervention Group to Compare Against AL Group:", style={'fontWeight': 'bold'}),
                dcc.RadioItems(id='volcano-tx-selector', options=[{'label': f' {tx}', 'value': tx} for tx in tx_options], value=tx_options[1], inline=True, style={'padding': '10px 0'})
            ]),
            dcc.Graph(id='volcano-plot-graph')
        ])
        
    elif active_tab == 'tab-heatmap':
        return html.Div(children=[
            html.Div(style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '8px', 'marginBottom': '20px'}, children=[
                html.Label("Filter Heatmap Layer by Significant Two-Way Feature Constraints:", style={'fontWeight': 'bold', 'color': '#2A4D69'}),
                dcc.RadioItems(id='heatmap-anova-factor-selector', 
                               options=[
                                   {'label': ' Main Effect: Priming Condition (Healthy vs Primed)', 'value': 'ANOVA_p_PrimingEffect'},
                                   {'label': ' Main Effect: Intervention Strategy', 'value': 'ANOVA_p_TherapyEffect'},
                                   {'label': ' Structural Interaction Core (Priming × Therapy)', 'value': 'ANOVA_p_InteractionEffect'}
                               ], value='ANOVA_p_InteractionEffect', style={'padding': '10px 0'}),
                html.Label("Display Density Count Limit:"),
                dcc.Dropdown(id='heatmap-density-dropdown', options=[{'label': f'Top {n} Matches', 'value': n} for n in [15, 30, 50, 100]], value=30, style={'width': '200px'})
            ]),
            dcc.Graph(id='heatmap-graph')
        ])

    elif active_tab == 'tab-overlap':
        return html.Div(children=[
            html.Div(style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '8px', 'marginBottom': '20px', 'display': 'flex', 'gap': '40px'}, children=[
                html.Div(style={'flex': '1'}, children=[
                    html.Label("1. Select Groups to Intersect:", style={'fontWeight': 'bold', 'color': '#E67E22'}),
                    dcc.Checklist(id='overlap-groups-check', options=[{'label': f' {g.split(" ")[0]}', 'value': g} for g in tx_groups], value=tx_groups, inline=True)
                ]),
                html.Div(style={'width': '200px'}, children=[
                    html.Label("2. Regulation Direction:", style={'fontWeight': 'bold'}),
                    dcc.RadioItems(id='overlap-direction-radio', options=[{'label': ' Up-regulated vs AL', 'value': 'up'}, {'label': ' Suppressed vs AL', 'value': 'down'}], value='down')
                ]),
                html.Div(style={'flex': '1'}, children=[
                    html.Label("3. Set P-Value and Fold Thresholds:", style={'fontWeight': 'bold'}),
                    html.Div(style={'display': 'flex', 'gap': '15px', 'paddingTop': '10px'}, children=[
                        html.Div(children=["Min |log2FC|: ", dcc.Input(id='overlap-fc-input', type='number', value=0.5, step=0.1, style={'width': '60px'})]),
                        html.Div(children=["Max P-Value: ", dcc.Input(id='overlap-p-input', type='number', value=0.05, step=0.01, style={'width': '60px'})])
                    ])
                ])
            ]),
            
            html.Div(style={'display': 'flex', 'gap': '20px'}, children=[
                html.Div(style={'flex': '1'}, children=[dcc.Graph(id='overlap-bar-chart')]),
                html.Div(style={'flex': '1.8', 'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '8px'}, children=[
                    html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '10px'}, children=[
                        html.H4(id='overlap-table-header', style={'margin': '0', 'color': '#2A4D69'}),
                        html.Button("📥 Download Sub-Grid", id="btn-download-csv", style={'backgroundColor': '#27AE60', 'color': 'white', 'border': 'none', 'padding': '5px 10px', 'borderRadius': '4px'})
                    ]),
                    dcc.Download(id="download-dataframe-csv"),
                    dash_table.DataTable(
                        id='overlap-data-table',
                        columns=[
                            {"name": "Protein/Gene", "id": "gene_name"},
                            {"name": "IF log2FC", "id": "log2FC_IF (Intermittent Fasting)", "type": "numeric"},
                            {"name": "Lcar log2FC", "id": "log2FC_L-Carnitine", "type": "numeric"},
                            {"name": "LPE log2FC", "id": "log2FC_LPE 18:1", "type": "numeric"},
                            {"name": "LPC log2FC", "id": "log2FC_LPC 17:0", "type": "numeric"},
                            {"name": "Two-Way Intersect p", "id": "anova_pvalue", "type": "numeric"}
                        ],
                        page_size=10, sort_action="native", filter_action="native",
                        style_header={'backgroundColor': '#2A4D69', 'color': 'white', 'fontWeight': 'bold'}
                    )
                ])
            ])
        ])

# --- TAB 1 DETAILED METADATA COMPLIANCE CARD ---
@app.callback(
    [Output('profile-boxplot', 'figure'), Output('profile-stats-card', 'children')],
    Input('profile-dropdown', 'value')
)
def render_protein_profile(selected_gene):
    row_match = df_clean[df_clean['gene_name'] == selected_gene]
    stats_match = master_stats_df[master_stats_df['gene_name'] == selected_gene]
    if row_match.empty or stats_match.empty: return go.Figure(), "Signature trace data absent."
    
    row = row_match.iloc[0]
    st = stats_match.iloc[0]
    
    melt_df = pd.DataFrame([{'Group': k, 'Intensity': float(v)} for k, cols in group_cols.items() for v in row[cols].values])
    fig = px.box(melt_df, x='Group', y='Intensity', color='Group', points="all", title=f"Two-Way Profile Vector: {selected_gene}", color_discrete_sequence=px.colors.qualitative.Safe)
    fig.update_layout(showlegend=False, template='plotly_white')
    
    card_html = [
        html.H4(f"🔍 Feature Tracker: {selected_gene}"),
        html.P(f"Priming Effect p: {st['ANOVA_p_PrimingEffect']:.4e}", style={'margin': '4px 0'}),
        html.P(f"Therapy Effect p: {st['ANOVA_p_TherapyEffect']:.4e}", style={'margin': '4px 0'}),
        html.H5(f"Interaction Term p: {st['ANOVA_p_InteractionEffect']:.4e}", style={'color': '#D35400', 'margin': '6px 0'}),
        html.Small(f"Desc: {str(row.get('T: First.Protein.Description', 'N/A')).split('OS=')[0]}", style={'color': '#7F8C8D'})
    ]
    return fig, card_html

# --- TAB 2 BACKEND: VOLCANO SCATTER MATRIX ---
@app.callback(Output('volcano-plot-graph', 'figure'), Input('volcano-tx-selector', 'value'))
def render_dynamic_volcano(selected_tx):
    short_tx = selected_tx.split(" ")[0]
    fc_col = f'log2FC_{short_tx}_vs_AL'
    p_col = f'pvalue_{short_tx}_vs_AL'
    
    plot_df = master_stats_df.copy()
    plot_df['neg_log10_p'] = -np.log10(plot_df[p_col])
    plot_df['Significance'] = 'Not Significant'
    plot_df.loc[(plot_df[fc_col] >= 0.5) & (plot_df[p_col] <= 0.05), 'Significance'] = f'Up in {short_tx}'
    plot_df.loc[(plot_df[fc_col] <= -0.5) & (plot_df[p_col] <= 0.05), 'Significance'] = 'Up in AL'
    
    fig = px.scatter(plot_df, x=fc_col, y='neg_log10_p', color='Significance', hover_name='gene_name',
                       title=f"Volcano Framework: {short_tx} vs AL",
                       color_discrete_map={f'Up in {short_tx}': '#27AE60', 'Up in AL': '#C0392B', 'Not Significant': '#95A5A6'})
    fig.add_vline(x=0.5, line_dash="dash", line_color="black")
    fig.add_vline(x=-0.5, line_dash="dash", line_color="black")
    fig.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="black")
    fig.update_layout(template='plotly_white')
    return fig

# --- TAB 3 BACKEND: HEATMAP LAYER VIA SELECTABLE TWO-WAY FACTORS ---
@app.callback(
    Output('heatmap-graph', 'figure'),
    [Input('heatmap-anova-factor-selector', 'value'), Input('heatmap-density-dropdown', 'value')]
)
def render_heatmap_matrix(selected_factor, top_n):
    # Sort dataset row indexes based specifically on the selected factor profile (Main Effects vs Interaction)
    sorted_stats = master_stats_df.sort_values(by=selected_factor).head(top_n)
    matching_genes = sorted_stats['gene_name'].tolist()
    
    # Isolate relevant rows in cleaned expression profile array dataframe
    expression_sub = df_clean[df_clean['gene_name'].isin(matching_genes)].set_index('gene_name').reindex(matching_genes)
    z_scores = expression_sub[all_samples].apply(lambda r: (r - r.mean()) / r.std(), axis=1).values
    
    formatted_headers = [f"{g.split(' ')[0]}_{i}" for g, cols in group_cols.items() for i in range(1, len(cols) + 1)]
    
    fig = go.Figure(data=go.Heatmap(z=z_scores, x=formatted_headers, y=matching_genes, colorscale='RdBu_r', zmin=-2, zmax=2))
    fig.update_layout(title=f"Expression Profile Heatmap: Top {top_n} Features Ranked by {selected_factor}", template='plotly_white', height=300+(top_n*14))
    
    # Add grouping boundary dividing vectors lines to visual chart map
    for b in range(1, len(group_cols)):
        fig.add_vline(x=b*4 - 0.5, line_width=2, line_dash="dash", line_color="black")
    return fig

# --- TAB 4 BACKEND: OVERLAP CLUSTERING AND SUBSET SELECTIONS ---
@app.callback(
    [Output('overlap-bar-chart', 'figure'), Output('overlap-data-table', 'data'), Output('overlap-table-header', 'children')],
    [Input('overlap-groups-check', 'value'), Input('overlap-direction-radio', 'value'), Input('overlap-fc-input', 'value'), Input('overlap-p-input', 'value')]
)
def compute_overlaps(selected_tx_groups, direction, fc_cut, p_cut):
    if not selected_tx_groups: return go.Figure(), [], "Please select a subset configuration."
    conditions = []
    bar_data = []
    for tx in tx_groups:
        st = tx.split(' ')[0]
        fc_c, p_c = f'log2FC_{st}_vs_AL', f'pvalue_{st}_vs_AL'
        is_sig = (master_stats_df[fc_c] >= fc_cut) & (master_stats_df[p_c] <= p_cut) if direction == 'up' else (master_stats_df[fc_c] <= -fc_cut) & (master_stats_df[p_c] <= p_cut)
        bar_data.append({'Group': st, 'Type': 'Individual Significant', 'Count': is_sig.sum()})
        if tx in selected_tx_groups: conditions.append(is_sig)
        
    master_mask = np.logical_and.reduce(conditions)
    overlap_df = master_stats_df[master_mask].copy()
    bar_data.append({'Group': 'SHARED INTERSECT', 'Type': 'Intersection Core', 'Count': len(overlap_df)})
    
    fig_bar = px.bar(pd.DataFrame(bar_data), x='Group', y='Count', color='Type', barmode='group', color_discrete_map={'Individual Significant': '#2A4D69', 'Intersection Core': '#E67E22'})
    fig_bar.update_layout(template='plotly_white', showlegend=False)
    
    display_df = overlap_df.copy()
    for c in display_df.columns:
        if c not in ['gene_name', 'T: Protein.Group', 'T: First.Protein.Description']:
            display_df[c] = display_df[c].apply(lambda x: f"{x:.4e}" if x < 0.001 else f"{x:.2f}")
    display_df['anova_pvalue'] = display_df['ANOVA_p_InteractionEffect']
    
    out_df = display_df.rename(columns={'log2FC_IF_vs_AL': 'log2FC_IF (Intermittent Fasting)', 'log2FC_L-Carnitine_vs_AL': 'log2FC_L-Carnitine', 'log2FC_LPE_vs_AL': 'log2FC_LPE 18:1', 'log2FC_LPC_vs_AL': 'log2FC_LPC 17:0'})
    return fig_bar, out_df.to_dict('records'), f"Two-Way Filter Overlap Intersect Matrix Count: {len(overlap_df)} Matching Features"

@app.callback(Output("download-dataframe-csv", "data"), Input("btn-download-csv", "n_clicks"), [dash.dependencies.State('overlap-data-table', 'data')], prevent_initial_call=True)
def export_table_to_csv(n_clicks, table_rows):
    if not table_rows: return dash.no_update
    return dcc.send_data_frame(pd.DataFrame(table_rows).to_csv, "overlapping_filtered_subset.csv", index=False)

# ==============================================================================
# 4. RUN SYSTEM INTERFACE
# ==============================================================================
# ==============================================================================
# 4. RUN SYSTEM INTERFACE
# ==============================================================================
if __name__ == '__main__':
    # Grab the port assigned by Render, or default to 8050 locally
    port = int(os.environ.get("PORT", 8050))
    
    # Force the app to bind to 0.0.0.0 so external cloud traffic can reach it
    app.run(host='0.0.0.0', port=port, debug=False)
