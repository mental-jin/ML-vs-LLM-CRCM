"""
整理并统计分析脚本

功能：
1) 读取 Excel 文件 
2) 按 'Metastasis' 列分组，对指定变量进行组间比较：
   - 连续变量：Shapiro 正态性检验 => t 检验 或 Mann-Whitney U 检验
   - 分类变量：卡方检验或（2x2 且期望频数小）Fisher 精确检验
3) 将统计结果保存为 '统计分析结果.xlsx'

依赖：pandas, numpy, scipy
运行：在包含 Excel 文件的目录中运行 python 整理.py
"""

import re
import argparse
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact, shapiro
import warnings
warnings.filterwarnings('ignore')

from display_names import display_name as _display_name

FILE_PATH = r'tmp/merged_1634.xlsx'
OUTPUT_FILE_PATH = r'tmp/1634_terminal_analysis_results.xlsx'


def safe_shapiro(series):
    """对 series 做 Shapiro 检验：当样本过少或全相同时，返回 False（视为非正态）"""
    if len(series) < 3:
        return False
    if np.all(series == series.iloc[0]):
        return False
    try:
        _, p = shapiro(series)
        return p > 0.05
    except Exception:
        return False


def analyze_continuous(group1, group2, all_data=None):
    """对两个一维数值序列选择合适检验并返回统计字符串与 p 值
    
    参数:
        group1: M1组数据
        group2: M0组数据  
        all_data: 全量数据（可选，用于计算Overall统计）
    """
    # 去除缺失
    g1 = pd.to_numeric(group1, errors='coerce').dropna()
    g2 = pd.to_numeric(group2, errors='coerce').dropna()
    if len(g1) == 0 or len(g2) == 0:
        return None

    # 检查正态性
    normal = safe_shapiro(g1) and safe_shapiro(g2)

    if normal:
        # t 检验（等方差默认不作假设，使用 Welch t）
        stat, p = stats.ttest_ind(g1, g2, equal_var=False, nan_policy='omit')
        desc1 = f"{g1.mean():.2f}±{g1.std(ddof=1):.2f}"
        desc2 = f"{g2.mean():.2f}±{g2.std(ddof=1):.2f}"
        # Overall 描述
        if all_data is not None:
            g_all = pd.to_numeric(all_data, errors='coerce').dropna()
            desc_all = f"{g_all.mean():.2f}±{g_all.std(ddof=1):.2f}" if len(g_all) > 0 else ""
        else:
            desc_all = ""
        method = 't检验(Welch)'
    else:
        # Mann-Whitney U
        try:
            stat, p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
        except Exception:
            p = np.nan
        # 用中位数和第25/75百分位数显示，格式为: 中位数 (第25百分位数, 第75百分位数)
        m1 = g1.median()
        q1_1 = g1.quantile(0.25)
        q3_1 = g1.quantile(0.75)
        m2 = g2.median()
        q1_2 = g2.quantile(0.25)
        q3_2 = g2.quantile(0.75)
        desc1 = f"{m1:.2f} ({q1_1:.2f}, {q3_1:.2f})"
        desc2 = f"{m2:.2f} ({q1_2:.2f}, {q3_2:.2f})"
        # Overall 描述
        if all_data is not None:
            g_all = pd.to_numeric(all_data, errors='coerce').dropna()
            if len(g_all) > 0:
                m_all = g_all.median()
                q1_all = g_all.quantile(0.25)
                q3_all = g_all.quantile(0.75)
                desc_all = f"{m_all:.2f} ({q1_all:.2f}, {q3_all:.2f})"
            else:
                desc_all = ""
        else:
            desc_all = ""
        method = 'Mann-Whitney U'

    return {'group1': desc1, 'group2': desc2, 'overall': desc_all, 'method': method, 'p': p}


def analyze_categorical(group1, group2, varname=None, all_data=None):
    """对分类变量进行卡方或 Fisher 检验，返回各组比例描述、检验方法与 p 值

    如果传入 varname，当 varname 为 'pT' 或 'pN' 时，按数字从高到低排序类别；
    当 varname 为 'Differentiation_grade' 时，按 ['高中','中','中低'] 的顺序优先显示。
    
    参数:
        group1: M1组数据
        group2: M0组数据
        varname: 变量名（用于特殊排序）
        all_data: 全量数据（可选，用于计算Overall统计）
    """
    g1 = group1.dropna().astype(str)
    g2 = group2.dropna().astype(str)
    if len(g1) == 0 or len(g2) == 0:
        return None

    # 统计频数并构建列联表（行=类别，列=组）
    combined = pd.concat([g1, g2], axis=0)
    categories = list(pd.unique(combined))

    # 根据变量名决定展示顺序
    if varname is not None:
        if varname in ('T_stage', 'N_stage'):
            # 提取纯数字类别并按数值从高到低排序
            num_cats = [int(c) for c in categories if re.fullmatch(r"\d+", str(c))]
            num_cats = sorted(set(num_cats), reverse=True)
            ordered = [str(x) for x in num_cats]
            # 把非数字类别放到后面（保持原顺序）
            other = [c for c in categories if not re.fullmatch(r"\d+", str(c))]
            categories = [c for c in ordered if c in categories] + [c for c in other if c not in ordered]
        elif varname == 'Differentiation_grade':
            preferred = ['G1', 'G2', 'G3']
            ordered = [p for p in preferred if p in categories]
            categories = ordered + [c for c in categories if c not in ordered]

    table = pd.DataFrame(index=categories, columns=['g1', 'g2']).fillna(0)
    for cat in categories:
        table.loc[cat, 'g1'] = int((g1 == cat).sum())
        table.loc[cat, 'g2'] = int((g2 == cat).sum())

    # 计算期望频数
    try:
        chi2, p_chi, dof, expected = chi2_contingency(table.values)
    except Exception:
        chi2, p_chi, dof, expected = (np.nan, np.nan, np.nan, np.zeros_like(table.values))

    use_fisher = False
    if table.shape == (2, 2):
        # 对 2x2 表且期望频数小于5时考虑 Fisher
        if (expected < 5).any():
            use_fisher = True

    if use_fisher:
        try:
            # fisher_exact 需要 2x2 的 ndarray
            _, p = fisher_exact(table.values)
            method = 'Fisher精确检验'
        except Exception:
            p = p_chi
            method = '卡方检验(近似)'
    else:
        p = p_chi
        method = '卡方检验'

    # 辅助：格式化类别标签（若为数字且为整数形式则显示为整数而不是 1.0）
    def format_label(cat):
        try:
            f = float(cat)
            if f.is_integer():
                return str(int(f))
            else:
                s = str(f)
                # 去掉多余的零
                return s.rstrip('0').rstrip('.')
        except Exception:
            return str(cat)

    # 各类计数与比例（count(percentage)），确保显示每组的人数
    total1 = len(g1)
    total2 = len(g2)
    desc1_parts = []
    desc2_parts = []
    for cat in categories:
        cnt1 = int(table.loc[cat, 'g1'])
        cnt2 = int(table.loc[cat, 'g2'])
        pct1 = (cnt1 / total1 * 100) if total1 > 0 else 0.0
        pct2 = (cnt2 / total2 * 100) if total2 > 0 else 0.0
        label = format_label(cat)
        desc1_parts.append(f"{label}:{cnt1}({pct1:.1f}%)")
        desc2_parts.append(f"{label}:{cnt2}({pct2:.1f}%)")

    desc1 = ', '.join(desc1_parts)
    desc2 = ', '.join(desc2_parts)

    # Overall 描述（全量数据）
    if all_data is not None:
        g_all = all_data.dropna().astype(str)
        total_all = len(g_all)
        desc_all_parts = []
        for cat in categories:
            cnt_all = int((g_all == cat).sum())
            pct_all = (cnt_all / total_all * 100) if total_all > 0 else 0.0
            label = format_label(cat)
            desc_all_parts.append(f"{label}:{cnt_all}({pct_all:.1f}%)")
        desc_all = ', '.join(desc_all_parts)
    else:
        desc_all = ""

    return {'group1': desc1, 'group2': desc2, 'overall': desc_all, 'method': method, 'p': p}


def analyze_dataframe(df: pd.DataFrame, group_col: str = 'Metastasis') -> pd.DataFrame:
    # 保险起见：将形如"/"（两侧可能有空白）的内容统一视为缺失
    df = df.copy()
    df.replace(r'^\s*/\s*$', np.nan, regex=True, inplace=True)

    # 分组
    if group_col not in df.columns:
        raise KeyError(f"未找到列 '{group_col}'，无法分组")

    # 强制将分组列标准化为 0/1（兼容 Excel 里为 '0'/'1'、'0.0'/'1.0'、0/1、0.0/1.0，及前后空白）
    group_raw = df[group_col]
    group_str = group_raw.astype(str).str.strip()
    # 将字符串形式的缺失统一为 NaN
    group_str = group_str.replace({'': np.nan, 'nan': np.nan, 'NaN': np.nan, 'None': np.nan, 'NONE': np.nan})
    group_num = pd.to_numeric(group_str, errors='coerce')

    # 校验：非缺失值只允许 0 或 1
    unique_vals = pd.unique(group_num.dropna())
    invalid_vals = [v for v in unique_vals if v not in (0, 1)]
    if invalid_vals:
        raise ValueError(
            f"分组列 '{group_col}' 存在非0/1的取值: {invalid_vals}。"
            "请先清洗该列，确保仅包含 0 或 1（缺失可为空）。"
        )

    group1 = df[group_num == 1]
    group2 = df[group_num == 0]
    print(
        f"分组完成({group_col}): M1(=1)={len(group1)} 行, M0(=0)={len(group2)} 行, 缺失={int(group_num.isna().sum())} 行"
    )

    # 3) 变量列表（按用户要求）
    continuous_vars = [
       "IgA","IgG",	"IgM",	"C3","C4","C1Q","IgE","CH50","Creatinine","Albumin","ALP","ALT",
       "AST","Direct bilirubin","GGT","LDH","Prealbumin","Glucose","AST/ALT ratio",
       "Total bile acids","Total bilirubin","Triglycerides","Total protein","UA","Urea",
       "mAST","GLDH","A/G ratio","Globulin","Total cholesterol","Indirect bilirubin",
       "eGFR","IL10","IL4","IL5","IL8","IL12-p70","IFN-γ","IFN-α","IL-1β","IL6","IL17","TNF-α",
       "IL2",
       "CD3+ T cells %","CD4+ T cells %","CD8+ T cells %","CD19+ B cells %",
       "NK cells %",
       "Treg cells %",
       "CD4+ count","CD8+ count","CD19+ count","NK count",
       "CD3+ count","CD3+HLA-DR+ T cells %",
       
       "VitA","VitB1","VitB2","VitB6","VitC","VitE","Calcium",
       "AFP-L3 %","SCCA","CA724","NSE","CYFRA21-1","AFPL3","CA242","CA50","CA199","CEA","AFP",
       "CA153","CA125","WBC","RBC","Hemoglobin","Neutrophil count","Lymphocyte count","Platelet count",
       "Monocyte count","CRP","Iron","Reticulocyte  %","NLR","PLR","LMR","SII","PNI", "ALBI Score",

        "Age", "BMI", "Tumor size", "Tumor volume", "Ki67","TNLE","PLN"
    ]

    categorical_vars = [
        "Sex", "T stage", "N stage", "Differentiation grade", "Vascular invasion", "Perineural invasion",
        "Carcinoma nodule", "CDX2", "MLH1", "MSH2", "PMS2", "MSH6", "HER2", "Family history",
        "Colonic obstruction", "Hypertension", "Diabetes", "Coronary artery disease", "Hyperlipidemia",
        "Chemotherapy", "BRAF mutant", "KRAS mutant", "NRAS mutant", "HER2 mutant",
        "NTRK1 mutant", "NTRK2 mutant", "NTRK3 mutant", "RET mutant", "MSI-H",
        "mGPS"
    ]

    # 兼容性：只保留实际存在于 df 的列
    continuous_vars = [v for v in continuous_vars if v in df.columns]
    categorical_vars = [v for v in categorical_vars if v in df.columns]

    results = []

    # 分析连续变量
    for var in continuous_vars:
        try:
            res = analyze_continuous(group1[var], group2[var], all_data=df[var])
            if res is None:
                continue
            results.append({
				'变量': _display_name(var, mode="plot") or var,
                'Overall': res['overall'],
                'M0': res['group2'],
                'M1': res['group1'],
                '检验方法': res['method'],
                'P值': ('' if pd.isna(res['p']) else f"{res['p']:.4f}"),
                '显著性': '*' if (not pd.isna(res['p']) and res['p'] < 0.05) else ''
            })
        except Exception as e:
            print(f"连续变量 {var} 处理出错: {e}")

    # 分析分类变量
    for var in categorical_vars:
        try:
            res = analyze_categorical(group1[var], group2[var], varname=var, all_data=df[var])
            if res is None:
                continue
            results.append({
				'变量': _display_name(var, mode="plot") or var,
                'Overall': res['overall'],
                'M0': res['group2'],
                'M1': res['group1'],
                '检验方法': res['method'],
                'P值': ('' if pd.isna(res['p']) else f"{res['p']:.4f}"),
                '显著性': '*' if (not pd.isna(res['p']) and res['p'] < 0.05) else ''
            })
        except Exception as e:
            print(f"分类变量 {var} 处理出错: {e}")

    return pd.DataFrame(results)


def analyze_excel(file_path: str, output_file_path: str, group_col: str = 'Metastasis') -> None:
    print(f"读取文件: {file_path}")
    df = pd.read_excel(file_path)
    results_df = analyze_dataframe(df, group_col=group_col)
    results_df.to_excel(output_file_path, index=False)
    print(f"统计分析完成，结果已保存到: {output_file_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分组统计分析（连续变量+分类变量）")
    parser.add_argument("--input", default=FILE_PATH, help="输入Excel路径")
    parser.add_argument("--output", default=OUTPUT_FILE_PATH, help="输出结果Excel路径")
    parser.add_argument("--group_col", default='Metastasis', help="分组列名（1 vs 0）")
    return parser.parse_args()


def main():
    args = _parse_args()
    analyze_excel(args.input, args.output, group_col=args.group_col)


if __name__ == '__main__':
    main()
