# -*- coding: utf-8 -*-
"""
Sorting and Statistical Analysis Script

Functions:
1) Read Excel file 
2) Group by 'Metastasis' column, perform inter-group comparisons for specified variables:
   - Continuous variables: Shapiro normality test => t-test or Mann-Whitney U test
   - Categorical variables: Chi-square test or Fisher's exact test (if 2x2 and expected count is small)
3) Save statistical results as '统计分析结果.xlsx' (Statistical_Analysis_Results.xlsx)

Dependencies: pandas, numpy, scipy
Run: python 整理.py in the directory containing the Excel file
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
    """Perform Shapiro test on series: returns False (skewed) if samples are too few or identical."""
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
    """Select appropriate test for two 1D numeric sequences and return statistical description and p-value.
    
    Parameters:
        group1: M1 group data
        group2: M0 group data  
        all_data: All data (optional, used to compute Overall statistics)
    """
    g1 = pd.to_numeric(group1, errors='coerce').dropna()
    g2 = pd.to_numeric(group2, errors='coerce').dropna()
    if len(g1) == 0 or len(g2) == 0:
        return None

    normal = safe_shapiro(g1) and safe_shapiro(g2)

    if normal:
        # t-test (Welch t-test is used by default without assuming equal variance)
        stat, p = stats.ttest_ind(g1, g2, equal_var=False, nan_policy='omit')
        desc1 = f"{g1.mean():.2f}±{g1.std(ddof=1):.2f}"
        desc2 = f"{g2.mean():.2f}±{g2.std(ddof=1):.2f}"
        if all_data is not None:
            g_all = pd.to_numeric(all_data, errors='coerce').dropna()
            desc_all = f"{g_all.mean():.2f}±{g_all.std(ddof=1):.2f}" if len(g_all) > 0 else ""
        else:
            desc_all = ""
        method = 't-test (Welch)'
    else:
        # Mann-Whitney U
        try:
            stat, p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
        except Exception:
            p = np.nan
        # Display as: Median (25th percentile, 75th percentile)
        m1 = g1.median()
        q1_1 = g1.quantile(0.25)
        q3_1 = g1.quantile(0.75)
        m2 = g2.median()
        q1_2 = g2.quantile(0.25)
        q3_2 = g2.quantile(0.75)
        desc1 = f"{m1:.2f} ({q1_1:.2f}, {q3_1:.2f})"
        desc2 = f"{m2:.2f} ({q1_2:.2f}, {q3_2:.2f})"
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
    """Perform Chi-square or Fisher test on categorical variables, return proportions, methods, and p-values.

    If varname is passed, categories are sorted from high to low when varname is 'pT' or 'pN'.
    When varname is 'Differentiation_grade', they are prioritized as ['G1', 'G2', 'G3'].
    
    Parameters:
        group1: M1 group data
        group2: M0 group data
        varname: Variable name (used for custom sorting)
        all_data: All data (optional, used to compute Overall statistics)
    """
    g1 = group1.dropna().astype(str)
    g2 = group2.dropna().astype(str)
    if len(g1) == 0 or len(g2) == 0:
        return None

    combined = pd.concat([g1, g2], axis=0)
    categories = list(pd.unique(combined))

    if varname is not None:
        if varname in ('T_stage', 'N_stage'):
            num_cats = [int(c) for c in categories if re.fullmatch(r"\d+", str(c))]
            num_cats = sorted(set(num_cats), reverse=True)
            ordered = [str(x) for x in num_cats]
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

    try:
        chi2, p_chi, dof, expected = chi2_contingency(table.values)
    except Exception:
        chi2, p_chi, dof, expected = (np.nan, np.nan, np.nan, np.zeros_like(table.values))

    use_fisher = False
    if table.shape == (2, 2):
        if (expected < 5).any():
            use_fisher = True

    if use_fisher:
        try:
            _, p = fisher_exact(table.values)
            method = 'Fisher exact test'
        except Exception:
            p = p_chi
            method = 'Chi-square test (approximate)'
    else:
        p = p_chi
        method = 'Chi-square test'

    def format_label(cat):
        try:
            f = float(cat)
            if f.is_integer():
                return str(int(f))
            else:
                s = str(f)
                return s.rstrip('0').rstrip('.')
        except Exception:
            return str(cat)

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
    df = df.copy()
    df.replace(r'^\s*/\s*$', np.nan, regex=True, inplace=True)

    if group_col not in df.columns:
        raise KeyError(f"Column '{group_col}' not found, unable to group data.")

    group_raw = df[group_col]
    group_str = group_raw.astype(str).str.strip()
    group_str = group_str.replace({'': np.nan, 'nan': np.nan, 'NaN': np.nan, 'None': np.nan, 'NONE': np.nan})
    group_num = pd.to_numeric(group_str, errors='coerce')

    unique_vals = pd.unique(group_num.dropna())
    invalid_vals = [v for v in unique_vals if v not in (0, 1)]
    if invalid_vals:
        raise ValueError(
            f"Grouping column '{group_col}' contains non-0/1 values: {invalid_vals}. "
            "Please clean this column first to ensure it contains only 0 or 1 (missing values can be empty)."
        )

    group1 = df[group_num == 1]
    group2 = df[group_num == 0]
    print(
        f"Grouping completed({group_col}): M1(=1)={len(group1)} rows, M0(=0)={len(group2)} rows, Missing={int(group_num.isna().sum())} rows"
    )

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

    continuous_vars = [v for v in continuous_vars if v in df.columns]
    categorical_vars = [v for v in categorical_vars if v in df.columns]

    results = []

    # Analyze continuous variables
    for var in continuous_vars:
        try:
            res = analyze_continuous(group1[var], group2[var], all_data=df[var])
            if res is None:
                continue
            results.append({
				'Variable': _display_name(var, mode="plot") or var,
                'Overall': res['overall'],
                'M0': res['group2'],
                'M1': res['group1'],
                'Test Method': res['method'],
                'P-value': ('' if pd.isna(res['p']) else f"{res['p']:.4f}"),
                'Significance': '*' if (not pd.isna(res['p']) and res['p'] < 0.05) else ''
            })
        except Exception as e:
            print(f"Error processing continuous variable {var}: {e}")

    # Analyze categorical variables
    for var in categorical_vars:
        try:
            res = analyze_categorical(group1[var], group2[var], varname=var, all_data=df[var])
            if res is None:
                continue
            results.append({
				'Variable': _display_name(var, mode="plot") or var,
                'Overall': res['overall'],
                'M0': res['group2'],
                'M1': res['group1'],
                'Test Method': res['method'],
                'P-value': ('' if pd.isna(res['p']) else f"{res['p']:.4f}"),
                'Significance': '*' if (not pd.isna(res['p']) and res['p'] < 0.05) else ''
            })
        except Exception as e:
            print(f"Error processing categorical variable {var}: {e}")

    return pd.DataFrame(results)


def analyze_excel(file_path: str, output_file_path: str, group_col: str = 'Metastasis') -> None:
    print(f"Reading file: {file_path}")
    df = pd.read_excel(file_path)
    results_df = analyze_dataframe(df, group_col=group_col)
    results_df.to_excel(output_file_path, index=False)
    print(f"Statistical analysis completed. Results saved to: {output_file_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Group Statistical Analysis (Continuous + Categorical)")
    parser.add_argument("--input", default=FILE_PATH, help="Input Excel path")
    parser.add_argument("--output", default=OUTPUT_FILE_PATH, help="Output results Excel path")
    parser.add_argument("--group_col", default='Metastasis', help="Grouping column name (1 vs 0)")
    return parser.parse_args()


def main():
    args = _parse_args()
    analyze_excel(args.input, args.output, group_col=args.group_col)


if __name__ == '__main__':
    main()