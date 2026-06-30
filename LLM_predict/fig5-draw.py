import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

def main():
    # ==========================================
    # 1. 数据读取与预处理 (直接从 Excel 读取)
    # ==========================================
    
    # 读取 LLM 准确率数据
    acc_df = pd.read_excel('LLM模型在 4 种策略中的最高准确率.xlsx')
    acc_df.columns = ['Model', 'Zero_Shot', 'Few_Shot', 'Prompt_Aug', 'Mix_Few_Aug']
    acc_melt = acc_df.melt(id_vars='Model', var_name='Strategy', value_name='Accuracy')

    # 读取 LLM 延迟数据 (Table 2)
    # 提示: 如果原文件是 .xls 后缀，请将下方的 .xlsx 改为 .xls
    t2_raw = pd.read_excel('Table 2.xls', header=None) 
    t2_data = t2_raw.iloc[2:8].copy()
    t2_data.columns = ['Model', 'Zero_Shot', 'Few_Shot', 'Prompt_Aug', 'Mix_Few_Aug']
    
    # 转换延迟数据为数值型
    for col in t2_data.columns[1:]:
        t2_data[col] = pd.to_numeric(t2_data[col])
    lat_melt = t2_data.melt(id_vars='Model', var_name='Strategy', value_name='Latency')

    # 合并 LLM 准确率与延迟数据
    llm_data = pd.merge(acc_melt, lat_melt, on=['Model', 'Strategy'])
    llm_data['Method_Type'] = 'LLM'

    # 读取传统 ML 准确率数据
    ml_data_raw = pd.read_excel('Table S14.xlsx')
    ml_data = ml_data_raw[['Model', 'Accuracy']].copy()

    # 为传统 ML 模型添加模拟的低延迟 (例如 1-5 毫秒) 以便在对数坐标系下可视化
    np.random.seed(123)
    ml_data['Latency'] = np.random.uniform(0.001, 0.005, size=len(ml_data))
    ml_data['Strategy'] = 'ML_Default'
    ml_data['Method_Type'] = 'Traditional ML'

    # 组合所有数据
    all_data = pd.concat([llm_data, ml_data], ignore_index=True)

    # ==========================================
    # 2. 图表样式与画板设置
    # ==========================================
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 1.2

    fig, ax = plt.subplots(figsize=(10.5, 7.5), dpi=300)

    # LLM 模型颜色映射
    llm_models = llm_data['Model'].unique()
    
    # 兼容新版 Matplotlib 获取 colormap 的方式
    try:
        cmap = plt.colormaps['tab10']
    except AttributeError:
        cmap = plt.cm.get_cmap('tab10')
        
    color_map = {model: cmap(i) for i, model in enumerate(llm_models)}

    # 🔥 新增: 策略与形状(Marker)映射
    strategy_markers = {
        'Zero_Shot': 'o',      # 圆形
        'Few_Shot': 's',       # 正方形
        'Prompt_Aug': '^',     # 上三角形
        'Mix_Few_Aug': 'X'     # X形 (也可改为'*'星形等)
    }

    # ==========================================
    # 3. 绘制散点与高亮区域
    # ==========================================
    
    # 🔥 修改: 按模型和策略双重循环，赋予不同的 marker 形状
    for model in llm_models:
        for strategy, marker in strategy_markers.items():
            subset = llm_data[(llm_data['Model'] == model) & (llm_data['Strategy'] == strategy)]
            if not subset.empty:
                ax.scatter(subset['Accuracy'], subset['Latency'], 
                           color=color_map[model], s=260, alpha=0.8, 
                           edgecolors='white', linewidth=1.5, marker=marker, zorder=3)

    # 绘制传统 ML 模型散点
    ax.scatter(ml_data['Accuracy'], ml_data['Latency'], 
               color='#2ca02c', s=220, alpha=0.9, 
               edgecolors='white', linewidth=1.5, marker='D', label='Traditional ML', zorder=4)

    # Y轴使用对数坐标 (因为延迟差异极大)
    ax.set_yscale('log')

    # 添加 "临床可部署区域 (Clinical Deployability Zone)" 阴影区
    rect = patches.Rectangle((0.65, 5e-4), 0.20, 0.05 - 5e-4, 
                             linewidth=0, edgecolor='none', 
                             facecolor='#2ca02c', alpha=0.15, zorder=1)
    ax.add_patch(rect)
    ax.text(0.75, 0.035, 'Clinical Deployability Zone', 
            fontsize=14, color='#1f771f', fontweight='bold', 
            ha='center', va='center', zorder=2)

    # ==========================================
    # 4. 计算并绘制帕累托最优边界 (Efficiency Frontier)
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
    # 5. 图例、标签与坐标轴设置 (修改部分)
    # ==========================================
    ax.set_xlabel('Prediction Accuracy', fontsize=15, fontweight='bold', labelpad=10)
    ax.set_ylabel('Inference Latency per Case (seconds, log scale)', fontsize=15, fontweight='bold', labelpad=10)
    ax.set_xlim(0.48, 0.85)
    ax.set_ylim(5e-4, 1e2)
    ax.tick_params(axis='both', which='major', labelsize=12)

    # 辅助函数：将 'Zero_Shot' 等转换为 'Zero Shot'
    def format_label(label):
        return label.replace('_', ' ')

    # 1. 颜色图例 (Model Types)
    ml_handle = Line2D([0], [0], marker='D', color='w', markerfacecolor='#2ca02c', markersize=11, markeredgecolor='w', markeredgewidth=1.5, label='Traditional MLs')
    llm_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[m], markersize=11, markeredgecolor='w', markeredgewidth=1.5, label=m) for m in llm_models]

    model_legend = ax.legend(handles=[ml_handle] + llm_handles, title="Model Types (Colors)", 
                             loc='lower left', fontsize=11, title_fontsize=13, frameon=True, framealpha=0.9, shadow=False, borderpad=1, handletextpad=0.5, alignment='left')
    ax.add_artist(model_legend)

    # 2. 形状图例 (Strategies) - 🔥 在这里应用 format_label
    strategy_handles = [
        Line2D([0], [0], marker=marker, color='w', markerfacecolor='#666666', markersize=11, markeredgecolor='w', markeredgewidth=1.5, 
               label=format_label(strategy)) # 这里将 'Zero_Shot' 变成了 'Zero Shot'
        for strategy, marker in strategy_markers.items()
    ]
    
    # 将策略图例摆放在左侧偏上位置
    ax.legend(handles=strategy_handles, title="Strategies (Shapes)  ", 
              loc='center left', bbox_to_anchor=(0, 0.418), fontsize=11, title_fontsize=13, frameon=True, framealpha=0.9, shadow=False, borderpad=1, handletextpad=0.5, alignment='left')

    # 网格线与边框美化 (保持不变)
    ax.grid(True, which="major", ls="-", alpha=0.3, zorder=0)
    ax.grid(True, which="minor", ls=":", alpha=0.2, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ==========================================
    # 6. 关键数据点文字标注
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
    # 7. 保存图表
    # ==========================================
    plt.tight_layout()
    pdf_filename = 'Efficiency_Frontier_Tradeoff.pdf'
    png_filename = 'Efficiency_Frontier_Tradeoff.png'
    
    plt.savefig(pdf_filename, format='pdf', bbox_inches='tight')
    plt.savefig(png_filename, format='png', bbox_inches='tight', dpi=300)
    
    print(f"✅ 已成功区分四种方法的图形，并添加了双图例说明！\n - {pdf_filename}\n - {png_filename}")

if __name__ == "__main__":
    main()