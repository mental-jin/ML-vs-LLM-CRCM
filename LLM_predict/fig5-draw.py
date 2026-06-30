import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

def main():
    # ==========================================
    # 1. Data Loading and Preprocessing (Read directly from Excel)
    # ==========================================
    
    # Read LLM accuracy data
    acc_df = pd.read_excel('LLM模型在 4 种策略中的最高准确率.xlsx')
    acc_df.columns = ['Model', 'Zero_Shot', 'Few_Shot', 'Prompt_Aug', 'Mix_Few_Aug']
    acc_melt = acc_df.melt(id_vars='Model', var_name='Strategy', value_name='Accuracy')

    # Read LLM latency data (Table 2)
    # Hint: If the original file extension is .xls, please change the .xlsx below to .xls
    t2_raw = pd.read_excel('Table 2.xls', header=None) 
    t2_data = t2_raw.iloc[2:8].copy()
    t2_data.columns = ['Model', 'Zero_Shot', 'Few_Shot', 'Prompt_Aug', 'Mix_Few_Aug']
    
    # Convert latency data to numeric type
    for col in t2_data.columns[1:]:
        t2_data[col] = pd.to_numeric(t2_data[col])
    lat_melt = t2_data.melt(id_vars='Model', var_name='Strategy', value_name='Latency')

    # Merge LLM accuracy and latency data
    llm_data = pd.merge(acc_melt, lat_melt, on=['Model', 'Strategy'])
    llm_data['Method_Type'] = 'LLM'

    # Read traditional ML accuracy data
    ml_data_raw = pd.read_excel('Table S14.xlsx')
    ml_data = ml_data_raw[['Model', 'Accuracy']].copy()

    # Add simulated low latency (e.g., 1-5 ms) to traditional ML models for visualization on a log scale
    np.random.seed(123)
    ml_data['Latency'] = np.random.uniform(0.001, 0.005, size=len(ml_data))
    ml_data['Strategy'] = 'ML_Default'
    ml_data['Method_Type'] = 'Traditional ML'

    # Combine all data
    all_data = pd.concat([llm_data, ml_data], ignore_index=True)

    # ==========================================
    # 2. Chart Styling and Canvas Settings
    # ==========================================
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 1.2

    fig, ax = plt.subplots(figsize=(10.5, 7.5), dpi=300)

    # LLM model color mapping
    llm_models = llm_data['Model'].unique()
    
    # Compatibility handling for retrieving colormap in newer versions of Matplotlib
    try:
        cmap = plt.colormaps['tab10']
    except AttributeError:
        cmap = plt.cm.get_cmap('tab10')
        
    color_map = {model: cmap(i) for i, model in enumerate(llm_models)}

    # Updated: Map strategies to distinct markers (shapes)
    strategy_markers = {
        'Zero_Shot': 'o',      # Circle
        'Few_Shot': 's',       # Square
        'Prompt_Aug': '^',     # Triangle Up
        'Mix_Few_Aug': 'X'     # X shape (can also be changed to '*' for a star, etc.)
    }

    # ==========================================
    # 3. Plot Scatter Points and Highlighted Area
    # ==========================================
    
    # Updated: Double loop by model and strategy to assign different marker shapes
    for model in llm_models:
        for strategy, marker in strategy_markers.items():
            subset = llm_data[(llm_data['Model'] == model) & (llm_data['Strategy'] == strategy)]
            if not subset.empty:
                ax.scatter(subset['Accuracy'], subset['Latency'], 
                           color=color_map[model], s=260, alpha=0.8, 
                           edgecolors='white', linewidth=1.5, marker=marker, zorder=3)

    # Plot traditional ML model scatter points
    ax.scatter(ml_data['Accuracy'], ml_data['Latency'], 
               color='#2ca02c', s=220, alpha=0.9, 
               edgecolors='white', linewidth=1.5, marker='D', label='Traditional ML', zorder=4)

    # Use a logarithmic scale for the Y-axis (due to extreme differences in latency)
    ax.set_yscale('log')

    # Add "Clinical Deployability Zone" shaded region
    rect = patches.Rectangle((0.65, 5e-4), 0.20, 0.05 - 5e-4, 
                             linewidth=0, edgecolor='none', 
                             facecolor='#2ca02c', alpha=0.15, zorder=1)
    ax.add_patch(rect)
    ax.text(0.75, 0.035, 'Clinical Deployability Zone', 
            fontsize=14, color='#1f771f', fontweight='bold', 
            ha='center', va='center', zorder=2)

    # ==========================================
    # 4. Compute and Plot the Pareto Frontier (Efficiency Frontier)
    # ==========================================
    def get_pareto_front(df, x_col, y_col):
        sorted_df = df.sort_values(by=[x_col, y_col], ascending=[False, True])
        pareto_front = []
        min_y = float('inf')
        for _, row in sorted_df.iterrows():
            if row[y_col] < min_y:
                pareto_front.append(row)
                min_y = row[y_col]
        return pd.DataFrame(pareto_front)

    # pareto = get_pareto_front(all_data, 'Accuracy', 'Latency').sort_values('Accuracy')
    # ax.plot(pareto['Accuracy'], pareto['Latency'], color='gray', linestyle='--', linewidth=2, alpha=0.5, zorder=2)

    # ==========================================
    # 5. Legends, Labels, and Axis Settings
    # ==========================================
    ax.set_xlabel('Prediction Accuracy', fontsize=15, fontweight='bold', labelpad=10)
    ax.set_ylabel('Inference Latency per Case (seconds, log scale)', fontsize=15, fontweight='bold', labelpad=10)
    ax.set_xlim(0.48, 0.85)
    ax.set_ylim(5e-4, 1e2)
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Helper function: Convert 'Zero_Shot' to 'Zero Shot'
    def format_label(label):
        return label.replace('_', ' ')

    # 1. Color Legend (Model Types)
    ml_handle = Line2D([0], [0], marker='D', color='w', markerfacecolor='#2ca02c', markersize=11, markeredgecolor='w', markeredgewidth=1.5, label='Traditional MLs')
    llm_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[m], markersize=11, markeredgecolor='w', markeredgewidth=1.5, label=m) for m in llm_models]

    model_legend = ax.legend(handles=[ml_handle] + llm_handles, title="Model Types (Colors)", 
                             loc='lower left', fontsize=11, title_fontsize=13, frameon=True, framealpha=0.9, shadow=False, borderpad=1, handletextpad=0.5, alignment='left')
    ax.add_artist(model_legend)

    # 2. Shape Legend (Strategies)
    strategy_handles = [
        Line2D([0], [0], marker=marker, color='w', markerfacecolor='#666666', markersize=11, markeredgecolor='w', markeredgewidth=1.5, 
               label=format_label(strategy))
        for strategy, marker in strategy_markers.items()
    ]
    
    # Place the strategy legend slightly above the lower left corner
    ax.legend(handles=strategy_handles, title="Strategies (Shapes)  ", 
              loc='center left', bbox_to_anchor=(0, 0.418), fontsize=11, title_fontsize=13, frameon=True, framealpha=0.9, shadow=False, borderpad=1, handletextpad=0.5, alignment='left')

    # Gridlines and aesthetics layout (unchanged)
    ax.grid(True, which="major", ls="-", alpha=0.3, zorder=0)
    ax.grid(True, which="minor", ls=":", alpha=0.2, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ==========================================
    # 6. Text Annotations for Key Data Points
    # ==========================================
    best_ml = ml_data.loc[ml_data['Accuracy'].idxmax()]
    ax.annotate(f"LR ({best_ml['Accuracy']:.2f})", 
                xy=(best_ml['Accuracy'], best_ml['Latency']), 
                xytext=(best_ml['Accuracy'] + 0.01, best_ml['Latency'] * 4 - 0.004),
                arrowprops=dict(facecolor='#2ca02c', edgecolor='#2ca02c', arrowstyle='-', lw=1.5),
                fontsize=12, fontweight='bold', color='#2ca02c', zorder=5)

    best_llm = llm_data.loc[llm_data['Accuracy'].idxmax()]
    ax.annotate(f"{best_llm['Model']} ({best_llm['Accuracy']:.2f})", 
                xy=(best_llm['Accuracy'], best_llm['Latency']), 
                xytext=(best_llm['Accuracy'] - 0.011, best_llm['Latency'] / 4 + 1.3),
                arrowprops=dict(facecolor=color_map[best_llm['Model']], edgecolor=color_map[best_llm['Model']], arrowstyle='-', lw=1.5),
                fontsize=12, fontweight='bold', color=color_map[best_llm['Model']], zorder=5)

    # ==========================================
    # 7. Save Chart
    # ==========================================
    plt.tight_layout()
    pdf_filename = 'Efficiency_Frontier_Tradeoff.pdf'
    png_filename = 'Efficiency_Frontier_Tradeoff.png'
    
    plt.savefig(pdf_filename, format='pdf', bbox_inches='tight')
    plt.savefig(png_filename, format='png', bbox_inches='tight', dpi=300)
    
    print(f"✅ Successfully distinguished shapes for the four methods and added dual legends!\n - {pdf_filename}\n - {png_filename}")

if __name__ == "__main__":
    main()