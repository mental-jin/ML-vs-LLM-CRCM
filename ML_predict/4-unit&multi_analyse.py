import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning, PerfectSeparationError, HessianInversionWarning
from scipy.stats import chi2
import scipy.linalg
from patsy import dmatrices
from typing import Any, Dict, List, Optional, Tuple, cast
from datetime import datetime
import warnings
import re

try:
    from display_names import display_name as _display_name, display_level as _display_level
except Exception:
    def _display_name(name: str, mode: str = "plot") -> str:  # type: ignore
        return str(name)

    def _display_level(var: str, level: str, mode: str = "plot") -> str:  # type: ignore
        return f"{var}({level})"

# ==========================================
# 1. 配置区域 (请根据你的实际数据修改这里)
# ==========================================
file_path = r'tmp\merged_1634.xlsx' # Excel 文件路径
target_col = 'Metastasis'          # 因变量列名 (Excel里的表头)
exclude_cols = ['Name', 'ReportTime', 'VisitNumber', 'Number']    # 需要剔除的自变量列名（与 Excel 表头一致，留空则不限）

# 多因素分析使用插补后的数据（用于减少缺失导致的掉样本）
multivar_file_path = r'tmp\merged_1634.xlsx'

# 可由用户强制指定的变量类型（优先级高于自动识别）
# 使用原始列名（与 Excel 表头一致）
FORCE_CATEGORICAL: List[str] = [ "Sex", "T stage", "N stage", "Differentiation grade", "Vascular invasion", "Perineural invasion",
        "Carcinoma nodule", "MLH1", "MSH2", "PMS2", "MSH6",  "Family history",
        "Colonic obstruction", "Hypertension", "Diabetes", "Coronary artery disease", "Hyperlipidemia",
        "BRAF mutant", "KRAS mutant", "NRAS mutant", 
        "MSI-H",
        "mGPS","Ki67"
        ]
    
FORCE_CONTINUOUS: List[str] = ["C1Q","Creatinine","Albumin","ALP","ALT",
       "AST","GGT","LDH","Prealbumin",
       "Total bile acids","Total bilirubin","Triglycerides","Total protein",
       "GLDH","A/G ratio","Globulin","Total cholesterol",
       "IL10","IL4","IL5","IL8","IL12-p70","IFN-γ","IFN-α","IL-1β","IL6","IL17","TNF-α",
       "IL2",
      "CD3+ T cells %","CD4+ T cells %","CD8+ T cells %","CD19+ B cells %",
       "NK cells %",
       "Treg cells %",
       "CD4+ count","CD8+ count","CD19+ count","NK count",
       "CD3+ count","CD3+HLA-DR+ T cells %",

       "VitA","VitB1","VitB2","VitB6","VitC","VitE","Calcium",
       "SCCA","CA724","NSE","CYFRA21-1","CA242","CA50","CA199","CEA","AFP",
       "CA153","CA125","WBC","RBC","Hemoglobin","Neutrophil count","Lymphocyte count","Platelet count",
       "Monocyte count","CRP","Iron","Reticulocyte  %","NLR","PLR","LMR","SII","PNI", "ALBI Score",

        "Age", "BMI", "Tumor size", "Tumor volume", "TNLE","PLN"
        ]

# 多因素（论文式“独立影响因素”）专用：强制变量类型
# - 优先级最高，仅影响“多因素手工清单模型”的类型判定
# - 便于你在缩减后的入模变量清单里，逐个明确指定哪些是分类/连续
# - 留空则沿用上面的 FORCE_* 与自动识别
MV_FORCE_CATEGORICAL: List[str] = ["Sex", "T stage", "N stage", "Differentiation grade", "Vascular invasion", "Perineural invasion",
        "Carcinoma nodule", "MLH1", "MSH2", "PMS2", "MSH6",  "Family history",
        "Colonic obstruction", "Hypertension", "Diabetes", "Coronary artery disease", "Hyperlipidemia",
        "BRAF mutant", "KRAS mutant", "NRAS mutant", 
        "MSI-H",
        "mGPS","Ki67"
]
MV_FORCE_CONTINUOUS: List[str] = ["C1Q","Creatinine","Albumin","ALP","ALT",
       "AST","GGT","LDH","Prealbumin",
       "Total bile acids","Total bilirubin","Triglycerides","Total protein",
       "GLDH","A/G ratio","Globulin","Total cholesterol",
       "IL10","IL4","IL5","IL8","IL12-p70","IFN-γ","IFN-α","IL-1β","IL6","IL17","TNF-α",
       "IL2",
      "CD3+ T cells %","CD4+ T cells %","CD8+ T cells %","CD19+ B cells %",
       "NK cells %",
       "Treg cells %",
       "CD4+ count","CD8+ count","CD19+ count","NK count",
       "CD3+ count","CD3+HLA-DR+ T cells %",
       "VitA","VitB1","VitB2","VitB6","VitC","VitE","Calcium",
       "SCCA","CA724","NSE","CYFRA21-1","CA242","CA50","CA199","CEA","AFP",
       "CA153","CA125","WBC","RBC","Hemoglobin","Neutrophil count","Lymphocyte count","Platelet count",
       "Monocyte count","CRP","Iron","Reticulocyte  %","NLR","PLR","LMR","SII","PNI", "ALBI Score",
        "Age", "BMI", "Tumor size", "Tumor volume", "TNLE","PLN"
]

# 用户可指定每个分类变量的参照组，格式: {'原始列名': '参照值'}
# 例如: REFERENCE_MAP = {'Sex': 'Male'}
REFERENCE_MAP: Dict[str, str] = {"Sex":'Female', "T stage":'T1-T2', "N stage":'0', "Differentiation grade":'G1', "Vascular invasion":'0', "Perineural invasion":'0',
    "Carcinoma nodule":'0', "MLH1":'1', "MSH2":'1', "PMS2":'1', "MSH6":'1', "Family history":'0',
    "Colonic obstruction":'0', "Hypertension":'0', "Diabetes":'0', "Coronary artery disease":'0', "Hyperlipidemia":'0',
    "BRAF mutant":'0', "KRAS mutant":'0', "NRAS mutant":'0', "mGPS":'0',"Ki67":'0',
    "MSI-H":'0'}

# 判定为分类变量的阈值 (如果某列只有不到 10 个唯一值，视为分类变量)
cat_threshold = 10 
logit_method = None            # 优化算法，None 表示使用 statsmodels 默认 (Newton-Raphson)
logit_maxiter = 1000           # 最大迭代次数，避免提前停止
exp_clip_value = 50            # 取指数前的截断，防止 exp 溢出

# 多因素模型数值稳定性配置
MV_STANDARDIZE_CONTINUOUS = True   # 是否对连续变量做 z-score 标准化
MV_Z_CLIP = 8.0                    # 标准化后截断范围（避免极端值导致 overflow）
MV_REG_ALPHA = 0.5                 # 正则化强度（fit_regularized 的 alpha）
MV_REG_L1_WT = 0.0                 # L1权重：0=纯L2，1=纯L1

# 为提升多因素模型的稳定性/效能（减少共线性、减少稀疏类别导致的分离）
MV_PRUNE_CONTINUOUS_BY_CORR = True
MV_CORR_THRESHOLD = 0.90           # 连续变量两两相关（Spearman）高于该阈值时，按列表顺序保留前者、剔除后者
MV_COLLAPSE_RARE_LEVELS = True
MV_MIN_LEVEL_COUNT = 10            # 分类变量某水平计数 < 该值时合并为 "Other"

# 将分期/分级按“有序数值趋势项”建模（减少自由度、提高稳定性）
# 当前按用户要求：T stage / N stage / Differentiation grade 作为“分类变量”处理
ORDINAL_TREND_VARS: List[str] = []

# 多因素逻辑回归：仅挑选以下变量进行分析（使用“原始列名/Excel表头”）
# 说明：用于论文式“独立影响因素”时，请把这里改成你预先指定的少量临床变量清单；
# 本脚本不会再做 LASSO 等数据驱动筛选。
MULTIVAR_FEATURES_ORIG: List[str] = [
    "LDH",
    "Prealbumin",
    "RBC",
    "Hemoglobin",
    "Lymphocyte count",
    "Tumor size",
    "IL5",
    "IFN-α",
    "CD3+HLA-DR+ T cells %",
    "MSH6",
    "BMI",
    "IL4",
    "IL6",
    "Ki67",
    "IL12-p70",
    "IL2",
    "NK count",
    "Treg cells %",
    "CD4+ count",
    "CD19+ count",
    "CD3+ count",
    "CA724",
    "CA242",
    "CA199",
    "CEA",
    "Tumor size",
    "PLN",
    "NLR",
    "PLR",
    "LMR",
    "SII",
    "T stage",
    "N stage",
    "Differentiation grade",
    "Vascular invasion",
    "Perineural invasion",
    "Carcinoma nodule",
    "PMS2",
    "Colonic obstruction",
    "KRAS mutant",
    "Tumor volume",
    "MLH1",
    "CD8+ count",
    "NK cells %",
    "IL8",
    "CD19+ B cells %",
    "CD4+ T cells %"

]

# ==========================================
# 2. 数据读取与预处理
# ==========================================
def clean_column_name(name):
    """
    清洗列名，移除特殊字符，便于 statsmodels 公式使用
    例如: "BMI (kg/m2)" -> "BMI_kg_m2"
    """
    new_name = re.sub(r'[^\w]', '_', name) # 将非字母数字下划线的字符替换为 _
    return new_name


def make_unique_safe_names(original_names: List[str]) -> Tuple[List[str], Dict[str, str], Dict[str, str]]:
    """为 statsmodels 公式生成安全且唯一的列名。

    返回:
      - safe_names: 与 original_names 等长、去重后的安全列名
      - safe_to_orig: {safe: orig}
      - orig_to_safe: {orig: safe}
    """
    used: Dict[str, int] = {}
    safe_names: List[str] = []
    safe_to_orig: Dict[str, str] = {}
    orig_to_safe: Dict[str, str] = {}

    for orig in original_names:
        base = clean_column_name(str(orig))
        # 避免空串
        if not base:
            base = "col"

        if base not in used:
            used[base] = 0
            safe = base
        else:
            used[base] += 1
            safe = f"{base}__{used[base]}"

        safe_names.append(safe)
        safe_to_orig[safe] = orig
        orig_to_safe[orig] = safe

    return safe_names, safe_to_orig, orig_to_safe


def _escape_patsy_str(value: str) -> str:
    """用于在 patsy/statsmodels 公式字符串里安全地包裹单引号字符串。"""
    return value.replace("\\", "\\\\").replace("'", "\\'")


def coerce_binary_target(series: pd.Series) -> Tuple[pd.Series, Optional[Dict[str, int]]]:
    """将二分类因变量稳健转换为 0/1。

    - 若已是 {0,1} / {False,True}，直接转换
    - 若是字符串形式的 0/1，转换
    - 若是常见中文/英文二分类，转换
    - 若仅有两个唯一值，给出映射（并返回 mapping 供打印提示）
    """
    s = series.copy()
    # 先处理布尔
    if s.dropna().map(type).isin([bool]).all():
        return s.astype(int), None

    # 尝试数值化（保留 NaN）
    s_num = pd.to_numeric(s, errors='coerce')
    uniq_num = sorted(set(s_num.dropna().unique().tolist()))
    if set(uniq_num).issubset({0.0, 1.0}) and len(uniq_num) in (1, 2):
        return s_num.astype('Int64').astype(float).astype('float64'), None

    # 尝试字符串映射
    s_str = s.astype(str).str.strip()
    # 把 "nan" 这种伪字符串恢复为缺失
    s_str = s_str.replace({'nan': np.nan, 'NaN': np.nan, 'None': np.nan, '': np.nan})

    mapping_known = {
        '0': 0, '1': 1,
        'false': 0, 'true': 1,
        '否': 0, '是': 1,
        '无': 0, '有': 1,
        '阴性': 0, '阳性': 1,
        'negative': 0, 'positive': 1,
        'no': 0, 'yes': 1,
    }
    s_lower = s_str.str.lower()
    mapped = s_lower.map(mapping_known)
    if mapped.dropna().nunique() in (1, 2) and mapped.notna().sum() > 0:
        return mapped.astype('float64'), None

    # 最后兜底：若确实只有两个非空取值，给出映射并提示
    uniq = [x for x in pd.unique(s_str.dropna()) if x is not None]
    if len(uniq) == 2:
        auto_map = {str(uniq[0]): 0, str(uniq[1]): 1}
        return s_str.map(auto_map).astype('float64'), auto_map

    raise ValueError(
        f"因变量 '{target_col}' 不是明显的二分类(0/1)。请先清洗或在代码里补充映射。"
    )


def dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def prune_continuous_by_spearman_corr(df_in: pd.DataFrame, cols_in_order: List[str], threshold: float) -> List[str]:
    """按列顺序对连续变量做相关性去冗余：若 |rho|>=threshold，则保留更靠前的变量。"""
    cols = [c for c in cols_in_order if c in df_in.columns]
    if len(cols) < 2:
        return cols
    # Spearman 对非正态/离群更稳
    corr = df_in[cols].corr(method='spearman').abs()
    keep: List[str] = []
    dropped: List[str] = []
    for c in cols:
        drop = False
        for k in keep:
            val = corr.at[c, k]
            if pd.notna(val):
                rho = float(cast(float, val))
                if rho >= threshold:
                    drop = True
                    break
        if drop:
            dropped.append(c)
        else:
            keep.append(c)
    if dropped:
        print(f"提示：多因素连续变量相关性去冗余，剔除 {len(dropped)} 列: {dropped}")
    return keep


def parse_ordinal_trend(series: pd.Series) -> pd.Series:
    """将分期/分级等有序变量解析为数值等级（尽量从字符串中提取数字）。

    支持形如：1, "1", "T3", "3a", "2-3" 等。
    解析失败则返回 NaN。
    """
    s = series.copy()
    # 避免 replace 的 downcast FutureWarning
    s = s.mask(s == '/', np.nan)

    # 先尝试直接数值化（对纯数值/字符串数值最快）
    as_num = pd.to_numeric(s, errors='coerce')
    if int(as_num.notna().sum()) == int(s.notna().sum()):
        return as_num

    # 再从字符串里提取第一个整数（用于 T3、N1、3a、2-3 等）
    s_str = s.astype(str).str.strip()
    s_str = s_str.replace({'nan': np.nan, 'NaN': np.nan, 'None': np.nan, '': np.nan})
    extracted = pd.to_numeric(s_str.str.extract(r'(\d+)')[0], errors='coerce')
    return as_num.fillna(extracted)

def safe_exp(value):
    """稳定的指数函数，避免出现无限大。"""
    clipped = np.clip(value, -exp_clip_value, exp_clip_value)
    return float(np.exp(clipped))


def infer_is_categorical(orig_name: str, series: pd.Series) -> bool:
    """统一的变量类型判定：优先 FORCE_*，否则按 dtype/唯一值阈值。"""
    if orig_name in ORDINAL_TREND_VARS:
        return False
    if orig_name in FORCE_CATEGORICAL:
        return True
    if orig_name in FORCE_CONTINUOUS:
        return False
    unique_count = int(series.nunique(dropna=True))
    if series.dtype == 'object' or series.dtype.name == 'category':
        return True
    if unique_count < cat_threshold:
        return True
    return False


def to_numeric_series(s: pd.Series) -> pd.Series:
    s2 = s.copy()
    s2 = s2.mask(s2 == '/', np.nan)
    return pd.to_numeric(s2, errors='coerce')


def clean_categorical_series(s: pd.Series) -> pd.Series:
    return normalize_categorical_series(s)


def normalize_categorical_series(s: pd.Series) -> pd.Series:
    """统一规范化分类变量的取值。

    目的：避免同一类别被写成不同形式（如 0 vs 0.0 vs '0 '）导致重复水平、参照组匹配失败、汇总表出现“同名类别+全0”。

    规则：
    - 去首尾空白
    - 将 '/', 空串, 'nan' 等视为缺失
    - 对可解析为数值的类别：整数 -> '0'/'1'；非整数 -> 使用 %.10g 规范化
    """
    s2 = s.copy()
    s2 = s2.mask(s2 == '/', np.nan)

    s_str = s2.astype(str).str.strip()
    s_str = s_str.replace({'nan': np.nan, 'NaN': np.nan, 'None': np.nan, 'NONE': np.nan, '': np.nan})
    # 先保留非缺失为字符串；再对数值型类别做归一
    out = s_str.copy()
    num = pd.to_numeric(s_str, errors='coerce')
    is_num = num.notna() & np.isfinite(num)
    if is_num.any():
        is_int = is_num & np.isclose(num, np.round(num))
        if is_int.any():
            out.loc[is_int] = num.loc[is_int].round().astype('Int64').astype(str)
        is_float = is_num & ~is_int
        if is_float.any():
            out.loc[is_float] = num.loc[is_float].map(lambda x: f"{x:.10g}")
    return out


def binarize_ki67_inplace(df_in: pd.DataFrame, orig_to_safe: Dict[str, str]) -> None:
    """Ki67：按用户要求转为二分类（>50 -> 1，<=50 -> 0）。"""
    if 'Ki67' not in orig_to_safe:
        return
    safe_ki67 = orig_to_safe['Ki67']
    ki67_numeric = pd.to_numeric(df_in[safe_ki67].replace('/', np.nan), errors='coerce')
    ki67_bin = pd.Series(np.nan, index=df_in.index, dtype='float64')
    ki67_bin[ki67_numeric > 50] = 1.0
    ki67_bin[ki67_numeric <= 50] = 0.0
    df_in[safe_ki67] = ki67_bin

try:
    # 读取 Excel
    df = pd.read_excel(file_path)
    print(f"成功读取数据，共 {len(df)} 行，{len(df.columns)} 列。")
    
    # 检查因变量是否存在
    if target_col not in df.columns:
        raise ValueError(f"错误：在 Excel 中找不到列名 '{target_col}'")

    
    # 删除因变量缺失的行
    df = df.dropna(subset=[target_col])
    
    # 创建唯一且安全的列名映射（避免清洗后重名导致列覆盖/映射错误）
    safe_names, safe_to_orig, orig_to_safe = make_unique_safe_names(df.columns.tolist())
    df.columns = safe_names
    safe_target = orig_to_safe[target_col]

    # 将因变量转换为二分类 0/1
    df[safe_target], target_mapping = coerce_binary_target(df[safe_target])
    if target_mapping is not None:
        print(f"提示：检测到二分类因变量非 0/1，自动映射为 0/1: {target_mapping}")

    # Ki67：按用户要求转为二分类（>50 -> 1，<50 -> 0；=50 视为 0）
    if 'Ki67' in orig_to_safe:
        safe_ki67 = orig_to_safe['Ki67']
        ki67_numeric = pd.to_numeric(df[safe_ki67].replace('/', np.nan), errors='coerce')
        ki67_bin = pd.Series(np.nan, index=df.index, dtype='float64')
        ki67_bin[ki67_numeric > 50] = 1.0
        ki67_bin[ki67_numeric <= 50] = 0.0
        df[safe_ki67] = ki67_bin
        print(
            "提示：Ki67 已转换为二分类（>50=1，<=50=0）。"
            f" 可用值计数: 0={int((df[safe_ki67]==0).sum())}, 1={int((df[safe_ki67]==1).sum())}, NA={int(df[safe_ki67].isna().sum())}"
        )

    # 排除列：按原始列名剔除
    excluded_safe = {orig_to_safe[col] for col in exclude_cols if col in orig_to_safe}

    missing_exclusions = [col for col in exclude_cols if col not in orig_to_safe]
    if missing_exclusions:
        print(f"提示：以下列未在数据中找到，因此无法剔除: {missing_exclusions}")
    excluded_present = [safe_to_orig[col] for col in excluded_safe if col in safe_to_orig]
    if excluded_present:
        print(f"将从分析中剔除这些列: {excluded_present}")
    
except Exception as e:
    print(f"数据加载出错: {e}")
    exit()

# ==========================================
# 3. 自动化单因素分析逻辑
# ==========================================
results_list = []

# 获取所有自变量 (排除因变量)
features = [col for col in df.columns if col != safe_target and col not in excluded_safe]

print("\n正在进行单因素分析...")

for safe_col in features:
    original_name = safe_to_orig[safe_col]

    # 如果变量没有足够的有效取值，则跳过
    non_na_unique = df[safe_col].dropna().nunique()
    if non_na_unique < 2:
        print(f"Warning: 变量 '{original_name}' 有效取值不足，跳过分析。")
        continue
    
    # --- A. 变量类型判断（支持用户强制指定） ---
    unique_count = df[safe_col].nunique(dropna=True)
    is_categorical = False
    is_ordinal_trend = original_name in ORDINAL_TREND_VARS
    # 优先使用用户指定的类型（使用原始列名）
    if is_ordinal_trend:
        is_categorical = False
    elif original_name in FORCE_CATEGORICAL:
        is_categorical = True
    elif original_name in FORCE_CONTINUOUS:
        is_categorical = False
    else:
        # 自动判断：object 或 category 类型，或唯一值小于阈值
        if df[safe_col].dtype == 'object' or df[safe_col].dtype.name == 'category':
            is_categorical = True
        elif unique_count < cat_threshold:
            is_categorical = True
    

    
    # --- B. 构建本变量的建模数据（避免连续变量里混入字符串导致 patsy 崩溃） ---
    tmp = df[[safe_target, safe_col]].copy()
    if is_ordinal_trend:
        tmp[safe_col] = parse_ordinal_trend(tmp[safe_col])
    else:
        # 统一将 '/' 等非数值字符视为缺失（与 9-mice copy.py 的处理一致）
        # 避免 pandas replace 的 downcast FutureWarning
        tmp[safe_col] = tmp[safe_col].mask(tmp[safe_col] == '/', np.nan)
        if is_categorical:
            tmp[safe_col] = normalize_categorical_series(tmp[safe_col])
        else:
            tmp[safe_col] = pd.to_numeric(tmp[safe_col], errors='coerce')

    # 记录该变量实际参与拟合的样本量（因 statsmodels 会丢弃 NaN）
    tmp = tmp.dropna(subset=[safe_target, safe_col])
    n_used = int(len(tmp))
    n_event = int((tmp[safe_target] == 1).sum())
    n_nonevent = int((tmp[safe_target] == 0).sum())

    # 因变量在该变量可用样本上必须同时包含 0/1
    if n_event == 0 or n_nonevent == 0:
        print(
            f"Warning: 变量 '{original_name}' 可用样本中因变量无变异(1={n_event},0={n_nonevent})，跳过分析。"
        )
        continue

    if n_used < 10:
        print(f"Warning: 变量 '{original_name}' 可用样本量过小(n={n_used})，跳过分析。")
        continue

    # --- C. 构建公式（分类变量显式指定参照组，避免事后猜测） ---
    ref_group = "-"
    if is_categorical:
        all_levels = tmp[safe_col].dropna().unique().tolist()
        all_levels = [str(x) for x in all_levels]
        ref = None
        if original_name in REFERENCE_MAP:
            candidate = str(REFERENCE_MAP[original_name])
            if candidate in all_levels:
                ref = candidate
        if ref is None and all_levels:
            # 默认用排序后的第一个（更可复现）
            ref = sorted(all_levels)[0]
        if ref is not None:
            ref_group = ref
            ref_escaped = _escape_patsy_str(ref)
            formula_term = f"C({safe_col}, Treatment(reference='{ref_escaped}'))"
        else:
            formula_term = f"C({safe_col})"
    else:
        formula_term = safe_col

    formula = f"{safe_target} ~ {formula_term}"
    
    # --- D. 拟合模型 ---
    def _fit_once(method_override: Optional[str] = None):
        with warnings.catch_warnings():
            warnings.filterwarnings('error', category=ConvergenceWarning)
            fit_kwargs: Dict[str, Any] = {'disp': 0, 'maxiter': logit_maxiter}
            chosen_method = method_override or logit_method
            if chosen_method:
                fit_kwargs['method'] = chosen_method
            return smf.logit(formula=formula, data=tmp).fit(**fit_kwargs)

    try:
        model = _fit_once()

        # --- E. 变量整体显著性（LR test：full vs null）---
        overall_p = np.nan
        try:
            null_model = smf.logit(formula=f"{safe_target} ~ 1", data=tmp).fit(disp=0, maxiter=logit_maxiter)
            lr_stat = 2.0 * (float(model.llf) - float(null_model.llf))
            df_diff = int(model.df_model - null_model.df_model)
            if df_diff > 0 and np.isfinite(lr_stat):
                overall_p = float(chi2.sf(lr_stat, df_diff))
        except Exception:
            overall_p = np.nan
        
        # --- E. 提取参数 ---
        params = model.params
        conf = model.conf_int()
        pvalues = model.pvalues

        # 收集分类变量在 params 中出现的水平，用于推断参考组
        cat_levels_in_params = []

        # 先将所有系数写入结果列表
        for term in params.index:
            if term == 'Intercept':
                continue

            # 计算 OR 和 CI
            or_val = safe_exp(params[term])
            ci_lower = safe_exp(conf.loc[term][0])
            ci_upper = safe_exp(conf.loc[term][1])
            p_val = pvalues[term]

            display_name = original_name
            comp_group = "-"
            if is_ordinal_trend:
                var_type_label = '有序(趋势)'
            else:
                var_type_label = '分类' if is_categorical else '连续'

            if is_categorical:
                match = re.search(r'\[T\.(.*?)\]', term)
                if match:
                    level = match.group(1)
                    comp_group = level
                    cat_levels_in_params.append(level)

            results_list.append({
                '原始变量名': display_name,
                '比较组': comp_group,
                '参照组': ref_group,
                '变量类型': var_type_label,
                '样本量n': n_used,
                '事件数(=1)': n_event,
                '非事件数(=0)': n_nonevent,
                '整体P值(LR)': overall_p,
                'OR值': or_val,
                '95% CI 下限': ci_lower,
                '95% CI 上限': ci_upper,
                'P值': p_val,
                '_term': term
            })

    except ConvergenceWarning:
        # 自动重试一次（常见于默认 NR 不稳定的情形）
        if logit_method is None:
            try:
                model = _fit_once(method_override='lbfgs')

                overall_p = np.nan
                try:
                    null_model = smf.logit(formula=f"{safe_target} ~ 1", data=tmp).fit(
                        disp=0, maxiter=logit_maxiter
                    )
                    lr_stat = 2.0 * (float(model.llf) - float(null_model.llf))
                    df_diff = int(model.df_model - null_model.df_model)
                    if df_diff > 0 and np.isfinite(lr_stat):
                        overall_p = float(chi2.sf(lr_stat, df_diff))
                except Exception:
                    overall_p = np.nan

                params = model.params
                conf = model.conf_int()
                pvalues = model.pvalues

                for term in params.index:
                    if term == 'Intercept':
                        continue

                    or_val = safe_exp(params[term])
                    ci_lower = safe_exp(conf.loc[term][0])
                    ci_upper = safe_exp(conf.loc[term][1])
                    p_val = pvalues[term]

                    display_name = original_name
                    comp_group = "-"
                    if is_ordinal_trend:
                        var_type_label = '有序(趋势)'
                    else:
                        var_type_label = '分类' if is_categorical else '连续'

                    if is_categorical:
                        match = re.search(r'\[T\.(.*?)\]', term)
                        if match:
                            comp_group = match.group(1)

                    results_list.append({
                        '原始变量名': display_name,
                        '比较组': comp_group,
                        '参照组': ref_group,
                        '变量类型': var_type_label,
                        '样本量n': n_used,
                        '事件数(=1)': n_event,
                        '非事件数(=0)': n_nonevent,
                        '整体P值(LR)': overall_p,
                        'OR值': or_val,
                        '95% CI 下限': ci_lower,
                        '95% CI 上限': ci_upper,
                        'P值': p_val,
                        '_term': term
                    })

                continue
            except Exception:
                pass

        print(f"Warning: 变量 '{original_name}' 因模型不收敛被跳过，请检查数据。")
    except PerfectSeparationError:
        print(f"Warning: 变量 '{original_name}' 出现完全分离(Perfect Separation)，跳过。")
    except Exception as e:
        print(f"Warning: 变量 '{original_name}' 分析失败。原因: {str(e)[:100]}")

# ==========================================
# 4. 结果整理与输出
# ==========================================
if results_list:
    res_df = pd.DataFrame(results_list)
    # 供后续“单+多因素汇总表”使用
    uni_res_df = res_df.copy()
    
    # 清理临时列
    if '_term' in res_df.columns:
        res_df = res_df.drop(columns=['_term'])

    # 格式化保留3位小数
    cols_to_round = ['OR值', '95% CI 下限', '95% CI 上限', 'P值', '整体P值(LR)']
    res_df[cols_to_round] = res_df[cols_to_round].round(3)
    
    # 合并 CI 列
    res_df['95% CI'] = res_df.apply(lambda x: f"{x['95% CI 下限']} - {x['95% CI 上限']}", axis=1)

    def _item_display_row(row: pd.Series) -> str:
        var = str(row.get('原始变量名', ''))
        var_type = str(row.get('变量类型', ''))
        comp = str(row.get('比较组', '-'))
        if var_type == '分类' and comp not in ('-', '', 'nan', 'None'):
            return _display_level(var, comp, mode='plot')
        return _display_name(var, mode='plot')

    def _ref_display_row(row: pd.Series) -> str:
        var = str(row.get('原始变量名', ''))
        var_type = str(row.get('变量类型', ''))
        ref = str(row.get('参照组', '-'))
        if var_type == '分类' and ref not in ('-', '', 'nan', 'None'):
            return _display_level(var, ref, mode='plot')
        return '-'

    # 新增展示列：连续变量带单位；分类变量显示为 var(level)
    res_df['项(展示)'] = res_df.apply(_item_display_row, axis=1)
    res_df['参照(展示)'] = res_df.apply(_ref_display_row, axis=1)
    
    # 整理最终显示列（保留原始列，额外增加展示列，避免影响下游脚本）
    final_df = res_df[
        ['项(展示)', '参照(展示)', '原始变量名', '比较组', '参照组', '变量类型', '样本量n', '事件数(=1)', '非事件数(=0)', '整体P值(LR)', 'OR值', '95% CI', 'P值']
    ]
    
    # 打印预览
    print("\n" + "="*50)
    print("分析完成！结果预览：")
    print("="*50)
    print(final_df.head(10).to_markdown(index=False))
    
    # 保存结果到 Excel
    output_filename = '单因素分析结果.xlsx'
    try:
        final_df.to_excel(output_filename, index=False)
        print(f"\n完整结果已保存至: {output_filename}")
    except PermissionError:
        # 通常是 Excel 正在打开占用导致无法覆盖
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        alt_filename = f"单因素分析结果_{ts}.xlsx"
        final_df.to_excel(alt_filename, index=False)
        print(
            f"\n提示：无法写入 '{output_filename}'（可能正被 Excel 打开）。"
            f"已改为保存到: {alt_filename}"
        )
    
else:
    print("没有生成任何结果，请检查数据格式。")
    uni_res_df = None


# ==========================================
# 5. 多因素逻辑回归分析（指定变量集合）
# ==========================================
print("\n=== 多因素逻辑回归分析（指定变量集合） ===")

# 多因素：从插补后的文件单独读取（与单因素可不同）
try:
    mv_df_raw = pd.read_excel(multivar_file_path)
    print(f"多因素数据读取成功: {multivar_file_path}，共 {len(mv_df_raw)} 行，{len(mv_df_raw.columns)} 列。")
except Exception as e:
    print(f"多因素数据加载出错: {e}")
    mv_df_raw = None

if mv_df_raw is None:
    print("多因素分析跳过（无法读取插补后数据文件）。")
else:
    # 为多因素数据创建唯一安全列名映射
    mv_safe_names, mv_safe_to_orig, mv_orig_to_safe = make_unique_safe_names(mv_df_raw.columns.tolist())
    mv_df_raw.columns = mv_safe_names

    if target_col not in mv_orig_to_safe:
        print(f"多因素分析跳过：在 {multivar_file_path} 中找不到因变量列 '{target_col}'")
    else:
        mv_safe_target = mv_orig_to_safe[target_col]

        # 二分类因变量转换
        try:
            mv_df_raw[mv_safe_target], mv_target_mapping = coerce_binary_target(mv_df_raw[mv_safe_target])
            if mv_target_mapping is not None:
                print(f"提示：多因素因变量自动映射为 0/1: {mv_target_mapping}")
        except Exception as e:
            print(f"多因素分析跳过：因变量无法转为二分类 0/1。原因: {e}")
            mv_safe_target = None

        # 多因素也需要与单因素一致：Ki67 二分类
        if mv_safe_target is not None:
            binarize_ki67_inplace(mv_df_raw, mv_orig_to_safe)

        # 排除列（若存在于多因素数据中）
        mv_excluded_safe = {mv_orig_to_safe[col] for col in exclude_cols if col in mv_orig_to_safe}

        if mv_safe_target is not None:
            mv_orig_list = dedupe_preserve_order(MULTIVAR_FEATURES_ORIG)
            mv_missing = [v for v in mv_orig_list if v not in mv_orig_to_safe]
            if mv_missing:
                print(f"提示：以下多因素变量在插补后数据中不存在，将跳过: {mv_missing}")

            mv_present_orig = [v for v in mv_orig_list if v in mv_orig_to_safe]
            mv_present_safe = [mv_orig_to_safe[v] for v in mv_present_orig]

            # 剔除被排除列/因变量
            mv_filtered: List[Tuple[str, str]] = []  # (orig, safe)
            for orig, safe in zip(mv_present_orig, mv_present_safe):
                if safe == mv_safe_target:
                    continue
                if safe in mv_excluded_safe:
                    continue
                mv_filtered.append((orig, safe))

            if not mv_filtered:
                print("没有可用于多因素分析的变量（可能都缺失或被排除）。")
            else:
                mv_data = mv_df_raw[[mv_safe_target] + [safe for _, safe in mv_filtered]].copy()

                # 逐列清洗/类型转换
                mv_terms: List[str] = []
                mv_ref_map_used: Dict[str, str] = {}
                mv_continuous_safe_cols: List[str] = []
                mv_continuous_protected: List[str] = []
                mv_cat_vars_orig: set = set()
                for orig, safe in mv_filtered:
                    is_ordinal_trend = orig in ORDINAL_TREND_VARS

                    if is_ordinal_trend:
                        mv_data[safe] = parse_ordinal_trend(mv_data[safe])
                        mv_terms.append(safe)
                        mv_continuous_safe_cols.append(safe)
                        mv_continuous_protected.append(safe)
                        continue

                    mv_data[safe] = mv_data[safe].mask(mv_data[safe] == '/', np.nan)

                    # 类型判断：优先 MV_FORCE_*（仅多因素用），其次 FORCE_*，否则自动
                    unique_count = mv_data[safe].nunique(dropna=True)
                    is_categorical = False
                    if orig in MV_FORCE_CATEGORICAL:
                        is_categorical = True
                    elif orig in MV_FORCE_CONTINUOUS:
                        is_categorical = False
                    elif orig in FORCE_CATEGORICAL:
                        is_categorical = True
                    elif orig in FORCE_CONTINUOUS:
                        is_categorical = False
                    else:
                        if mv_data[safe].dtype == 'object' or mv_data[safe].dtype.name == 'category':
                            is_categorical = True
                        elif unique_count < cat_threshold:
                            is_categorical = True

                    if is_categorical:
                        mv_cat_vars_orig.add(orig)
                        mv_data[safe] = normalize_categorical_series(mv_data[safe])

                        # 合并稀有类别，降低分离/不收敛风险
                        if MV_COLLAPSE_RARE_LEVELS:
                            vc = mv_data[safe].value_counts(dropna=True)
                            rare_levels = vc[vc < MV_MIN_LEVEL_COUNT].index.tolist()
                            if rare_levels:
                                mv_data[safe] = mv_data[safe].where(~mv_data[safe].isin(rare_levels), other='Other')

                        # 参照组
                        ref = None
                        all_levels = mv_data[safe].dropna().unique().tolist()
                        all_levels = [str(x) for x in all_levels]
                        if orig in REFERENCE_MAP:
                            candidate = str(REFERENCE_MAP[orig])
                            if candidate in all_levels:
                                ref = candidate
                        if ref is None and all_levels:
                            ref = sorted(all_levels)[0]

                        if ref is not None:
                            mv_ref_map_used[orig] = ref
                            ref_escaped = _escape_patsy_str(ref)
                            mv_terms.append(f"C({safe}, Treatment(reference='{ref_escaped}'))")
                        else:
                            mv_terms.append(f"C({safe})")
                    else:
                        mv_data[safe] = pd.to_numeric(mv_data[safe], errors='coerce')
                        mv_terms.append(safe)
                        mv_continuous_safe_cols.append(safe)

                # 删除缺失：多因素默认最简单做法是完整案例分析
                mv_data = mv_data.dropna(subset=[mv_safe_target] + [safe for _, safe in mv_filtered])
                mv_n_used = int(len(mv_data))
                mv_n_event = int((mv_data[mv_safe_target] == 1).sum())
                mv_n_nonevent = int((mv_data[mv_safe_target] == 0).sum())
                print(f"多因素分析可用样本量: n={mv_n_used}, 事件(=1)={mv_n_event}, 非事件(=0)={mv_n_nonevent}")

                # 连续变量标准化 + 截断，降低 exp/log 溢出风险
                if mv_continuous_safe_cols and MV_PRUNE_CONTINUOUS_BY_CORR:
                    protected_set = set(mv_continuous_protected)
                    prune_candidates = [c for c in mv_continuous_safe_cols if c not in protected_set]
                    pruned = prune_continuous_by_spearman_corr(
                        mv_data, prune_candidates, threshold=MV_CORR_THRESHOLD
                    )
                    dropped = [c for c in prune_candidates if c not in pruned]
                    if dropped:
                        mv_continuous_safe_cols = mv_continuous_protected + pruned
                        # 从公式中移除被剔除的连续变量（只在 term 恰好等于 safe 名时移除）
                        mv_terms = [t for t in mv_terms if t not in dropped]
                        # 从数据中删除被剔除列
                        mv_data = mv_data.drop(columns=dropped, errors='ignore')
                    else:
                        mv_continuous_safe_cols = mv_continuous_protected + pruned

                if MV_STANDARDIZE_CONTINUOUS and mv_continuous_safe_cols:
                    for c in mv_continuous_safe_cols:
                        col = mv_data[c].astype('float64')
                        mean = float(col.mean())
                        std = float(col.std(ddof=0))
                        if not np.isfinite(std) or std == 0.0:
                            mv_data[c] = 0.0
                        else:
                            z = (col - mean) / std
                            mv_data[c] = z.clip(-MV_Z_CLIP, MV_Z_CLIP)
                    print(
                        f"提示：多因素连续变量已标准化并截断到 [-{MV_Z_CLIP}, {MV_Z_CLIP}]（共 {len(mv_continuous_safe_cols)} 列）。"
                    )

                if mv_n_used < 20 or mv_n_event == 0 or mv_n_nonevent == 0:
                    print("Warning: 多因素分析可用样本不足或因变量无变异，跳过多因素回归。")
                else:
                    mv_formula = f"{mv_safe_target} ~ " + " + ".join(mv_terms)

                    def _fit_mv(method_override: Optional[str] = None):
                        """多因素拟合：使用 patsy 生成设计矩阵，并通过 QR 列主元选取满秩列，避免奇异矩阵。"""
                        y, X = dmatrices(mv_formula, mv_data, return_type='dataframe')

                        # statsmodels.Logit 更偏好 1D endog
                        y = y.iloc[:, 0].astype('float64')
                        X = X.astype('float64')

                        # 删除完全常量列（含全0列）
                        # 注意：patsy 生成的 'Intercept' 是常量列，但应保留
                        const_cols: List[str] = []
                        for c in X.columns:
                            if c == 'Intercept':
                                continue
                            # nunique==1 代表常量列（包含全0/全1/单一类别等）
                            if int(X[c].nunique(dropna=False)) <= 1:
                                const_cols.append(c)
                        if const_cols:
                            print(f"提示：多因素设计矩阵中删除 {len(const_cols)} 个常量列。")
                            X = X.drop(columns=const_cols, errors='ignore')

                        # QR with pivoting 选取满秩列
                        qr_res = scipy.linalg.qr(
                            np.asarray(X.values, dtype='float64'),
                            mode='economic',
                            pivoting=True,
                        )
                        R = qr_res[1]
                        piv = qr_res[2]
                        diag = np.abs(np.diag(R))
                        if diag.size == 0:
                            raise ValueError(
                                '多因素设计矩阵为空或无法分解（无可用自变量列）。'
                            )
                        eps = np.finfo(diag.dtype).eps
                        tol = float(eps * max(X.shape) * diag.max())
                        rank = int((diag > tol).sum())
                        if rank <= 0:
                            raise ValueError(
                                '多因素设计矩阵秩为 0（可能所有自变量为常量/完全共线）。'
                            )
                        keep_idx = sorted(piv[:rank])
                        if rank < X.shape[1]:
                            dropped_cols = [X.columns[i] for i in piv[rank:]]
                            print(
                                f"Warning: 多因素设计矩阵非满秩，"
                                f"已自动丢弃 {len(dropped_cols)} 个共线列以继续拟合。"
                            )
                        X = X.iloc[:, keep_idx]

                        # 经验性诊断：事件数/参数数（EPV），帮助判断过拟合风险
                        n_params = max(1, int(X.shape[1]) - 1)  # 不计截距
                        epv_event = float(mv_n_event) / float(n_params)
                        epv_nonevent = float(mv_n_nonevent) / float(n_params)
                        if epv_event < 10.0 or epv_nonevent < 10.0:
                            print(
                                f"Warning: 多因素模型 EPV 偏低（事件EPV={epv_event:.2f}, 非事件EPV={epv_nonevent:.2f}, 参数数={n_params}）。"
                                " 结果可能不稳定/存在过拟合风险。"
                            )

                        fit_kwargs: Dict[str, Any] = {'disp': 0, 'maxiter': logit_maxiter}
                        chosen = method_override or logit_method
                        # 多因素默认优先用 lbfgs（相比 NR 更稳）
                        if not chosen:
                            chosen = 'lbfgs'
                        fit_kwargs['method'] = chosen

                        # 把 RuntimeWarning（overflow/divide by zero 等）当作异常捕获，便于自动回退
                        try:
                            with warnings.catch_warnings():
                                warnings.filterwarnings('error', category=ConvergenceWarning)
                                warnings.filterwarnings('error', category=RuntimeWarning)
                                warnings.filterwarnings('ignore', category=HessianInversionWarning)
                                return sm.Logit(y, X).fit(**fit_kwargs)
                        except RuntimeWarning:
                            print(
                                "Warning: 多因素 MLE 拟合触发数值溢出/无穷警告，"
                                "改用正则化(Logit.fit_regularized)回退。"
                            )
                            return sm.Logit(y, X).fit_regularized(
                                alpha=MV_REG_ALPHA, L1_wt=MV_REG_L1_WT, disp=0, maxiter=logit_maxiter
                            )  # type: ignore[call-arg]
                        except Exception as e:
                            # 若仍出现奇异矩阵/极端分离，回退到正则化拟合以获得可用系数
                            msg = str(e)
                            if 'Singular matrix' in msg or 'singular' in msg.lower():
                                print(
                                    "Warning: 多因素 MLE 拟合出现奇异矩阵，"
                                    "改用正则化(Logit.fit_regularized)回退。"
                                )
                                return sm.Logit(y, X).fit_regularized(
                                    alpha=MV_REG_ALPHA, L1_wt=MV_REG_L1_WT, disp=0, maxiter=logit_maxiter
                                )  # type: ignore[call-arg]
                            raise

                    mv_model = None
                    try:
                        mv_model = _fit_mv()
                    except ConvergenceWarning:
                        if logit_method is None:
                            try:
                                mv_model = _fit_mv(method_override='lbfgs')
                            except Exception as e:
                                print(f"Warning: 多因素模型重试仍失败: {str(e)[:120]}")
                                mv_model = None
                    except PerfectSeparationError:
                        print("Warning: 多因素模型出现完全分离(Perfect Separation)，请检查变量/分组。")
                        mv_model = None
                    except Exception as e:
                        print(f"Warning: 多因素模型拟合失败: {str(e)[:120]}")
                        mv_model = None

                    if mv_model is not None:
                        mv_params = mv_model.params
                        try:
                            mv_conf = mv_model.conf_int()
                        except Exception:
                            mv_conf = None
                        try:
                            mv_pvalues = mv_model.pvalues
                        except Exception:
                            mv_pvalues = None

                        mv_rows: List[Dict[str, object]] = []
                        for term in mv_params.index:
                            if term == 'Intercept':
                                continue

                            # 反推原始变量名
                            base_safe = term
                            comp_group = "-"
                            ref_group = "-"

                            if term.startswith('C('):
                                # 形如 C(var, Treatment(...))[T.level]
                                m_var = re.match(r"C\(([^,\)]+)", term)
                                if m_var:
                                    base_safe = m_var.group(1)
                                m_level = re.search(r"\[T\.(.*?)\]", term)
                                if m_level:
                                    comp_group = m_level.group(1)

                            orig_name = mv_safe_to_orig.get(base_safe, base_safe)
                            if orig_name in ORDINAL_TREND_VARS:
                                var_type_label = '有序(趋势)'
                            else:
                                var_type_label = '分类' if orig_name in mv_cat_vars_orig else '连续'
                            if orig_name in mv_ref_map_used:
                                ref_group = mv_ref_map_used[orig_name]

                            or_val = safe_exp(mv_params[term])
                            if mv_conf is not None and term in mv_conf.index:
                                ci_lower = safe_exp(mv_conf.loc[term][0])
                                ci_upper = safe_exp(mv_conf.loc[term][1])
                            else:
                                ci_lower = np.nan
                                ci_upper = np.nan
                            if mv_pvalues is not None and term in mv_pvalues.index:
                                p_val = float(mv_pvalues[term])
                            else:
                                p_val = np.nan

                            mv_rows.append({
                                '原始变量名': orig_name,
                                '比较组': comp_group,
                                '参照组': ref_group,
                                '变量类型': var_type_label,
                                '样本量n': mv_n_used,
                                '事件数(=1)': mv_n_event,
                                '非事件数(=0)': mv_n_nonevent,
                                'OR值': or_val,
                                '95% CI 下限': ci_lower,
                                '95% CI 上限': ci_upper,
                                'P值': p_val,
                            })

                        if mv_rows:
                            mv_df = pd.DataFrame(mv_rows)
                            # 供后续“单+多因素汇总表”使用
                            mv_res_df = mv_df.copy()
                            mv_df[['OR值', '95% CI 下限', '95% CI 上限', 'P值']] = mv_df[
                                ['OR值', '95% CI 下限', '95% CI 上限', 'P值']
                            ].round(3)
                            mv_df['95% CI'] = mv_df.apply(
                                lambda x: f"{x['95% CI 下限']} - {x['95% CI 上限']}", axis=1
                            )

                            # 新增展示列：连续变量带单位；分类变量显示为 var(level)
                            mv_df['项(展示)'] = mv_df.apply(
                                lambda r: _display_level(str(r['原始变量名']), str(r['比较组']), mode='plot')
                                if str(r['变量类型']) == '分类' and str(r['比较组']) not in ('-', '', 'nan', 'None')
                                else _display_name(str(r['原始变量名']), mode='plot'),
                                axis=1,
                            )
                            mv_df['参照(展示)'] = mv_df.apply(
                                lambda r: _display_level(str(r['原始变量名']), str(r['参照组']), mode='plot')
                                if str(r['变量类型']) == '分类' and str(r['参照组']) not in ('-', '', 'nan', 'None')
                                else '-',
                                axis=1,
                            )
                            mv_final = mv_df[
                                ['项(展示)', '参照(展示)', '原始变量名', '比较组', '参照组', '变量类型', '样本量n', '事件数(=1)', '非事件数(=0)', 'OR值', '95% CI', 'P值']
                            ]

                            mv_output = '多因素分析结果.xlsx'
                            try:
                                mv_final.to_excel(mv_output, index=False)
                                print(f"多因素结果已保存至: {mv_output}")
                            except PermissionError:
                                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                                alt = f"多因素分析结果_{ts}.xlsx"
                                mv_final.to_excel(alt, index=False)
                                print(
                                    f"提示：无法写入 '{mv_output}'（可能正被 Excel 打开）。已改为保存到: {alt}"
                                )
                        else:
                            print("多因素模型拟合成功，但未产生系数（可能所有变量被剔除/常量）。")
                            mv_res_df = None


# ==========================================
# 6. 合并单因素 + 多因素为总表（论文式示例表）
# ==========================================
print("\n=== 生成单+多因素汇总总表（示例表样式） ===")


def _format_p_value(p: float) -> str:
    if p is None or (isinstance(p, float) and (np.isnan(p) or not np.isfinite(p))):
        return ""
    try:
        p2 = float(p)
    except Exception:
        return ""
    if p2 < 0.001:
        return "p<0.001"
    if p2 < 0.01:
        return "p<0.01"
    if p2 < 0.05:
        return "p<0.05"
    return f"p={p2:.3f}".rstrip('0').rstrip('.')


def _format_or_ci_p(or_val: object, ci_low: object, ci_high: object, p: object) -> str:
    try:
        or_f = float(or_val)
        lo_f = float(ci_low)
        hi_f = float(ci_high)
    except Exception:
        return ""
    if not (np.isfinite(or_f) and np.isfinite(lo_f) and np.isfinite(hi_f)):
        return ""
    ptxt = _format_p_value(float(p)) if p is not None else ""
    if ptxt:
        return f"{or_f:.2f} ({lo_f:.2f}-{hi_f:.2f}, {ptxt})"
    return f"{or_f:.2f} ({lo_f:.2f}-{hi_f:.2f})"


def _format_count_pct(count: int, denom: int) -> str:
    if denom <= 0:
        return ""
    pct = 100.0 * float(count) / float(denom)
    return f"{int(count)} ({pct:.1f}%)"


def _make_or_lookup(
    res: Optional[pd.DataFrame],
) -> Tuple[Dict[Tuple[str, str], Dict[str, object]], Dict[str, str]]:
    lookup: Dict[Tuple[str, str], Dict[str, object]] = {}
    ref_map: Dict[str, str] = {}
    if res is None:
        return lookup, ref_map
    need_cols = {'原始变量名', '比较组', '参照组', 'OR值', '95% CI 下限', '95% CI 上限', 'P值'}
    if not need_cols.issubset(set(res.columns)):
        return lookup, ref_map
    for _, row in res.iterrows():
        var = str(row.get('原始变量名', ''))
        comp = str(row.get('比较组', '-'))
        ref = str(row.get('参照组', '-'))
        if var and ref and ref not in ('-', 'nan', 'None') and var not in ref_map:
            ref_map[var] = ref
        lookup[(var, comp)] = {
            'OR值': row.get('OR值', np.nan),
            '95% CI 下限': row.get('95% CI 下限', np.nan),
            '95% CI 上限': row.get('95% CI 上限', np.nan),
            'P值': row.get('P值', np.nan),
            '参照组': ref,
        }
    return lookup, ref_map


def _clean_for_summary(orig_name: str, s: pd.Series) -> pd.Series:
    if orig_name in ORDINAL_TREND_VARS:
        return parse_ordinal_trend(s)
    if infer_is_categorical(orig_name, s):
        return clean_categorical_series(s)
    return to_numeric_series(s)


def _median_iqr_text(x: pd.Series) -> str:
    x2 = pd.to_numeric(x, errors='coerce').dropna()
    if len(x2) == 0:
        return ""
    med = float(x2.median())
    q1 = float(x2.quantile(0.25))
    q3 = float(x2.quantile(0.75))
    return f"{med:.2f} ({q1:.2f}-{q3:.2f})"


try:
    uni_lookup, uni_ref_map = _make_or_lookup(uni_res_df if 'uni_res_df' in globals() else None)
    mv_lookup, mv_ref_map = _make_or_lookup(mv_res_df if 'mv_res_df' in globals() else None)

    if 'df' not in globals() or 'safe_target' not in globals() or 'orig_to_safe' not in globals():
        raise ValueError('未找到用于汇总的 df / safe_target / orig_to_safe（请确认单因素部分成功读取数据）。')

    summary_base = df.copy()
    summary_base = summary_base.dropna(subset=[safe_target])
    n0_all = int((summary_base[safe_target] == 0).sum())
    n1_all = int((summary_base[safe_target] == 1).sum())

    # 总表默认覆盖“全部单因素变量”（按单因素遍历顺序），多因素列缺失则留空
    if 'features' in globals() and 'safe_to_orig' in globals():
        vars_in_order = dedupe_preserve_order(
            [safe_to_orig[safe] for safe in features if safe in safe_to_orig]
        )
    else:
        vars_in_order = dedupe_preserve_order(MULTIVAR_FEATURES_ORIG)

    rows_out: List[Dict[str, object]] = []
    col_m0 = f"M0, N={n0_all}"
    col_m1 = f"M1, N={n1_all}"

    for orig_name in vars_in_order:
        if orig_name not in orig_to_safe:
            continue
        safe_name = orig_to_safe[orig_name]
        if safe_name not in summary_base.columns:
            continue

        s_raw = summary_base[safe_name]
        s_clean = _clean_for_summary(orig_name, s_raw)
        t = summary_base[safe_target]

        is_cat = infer_is_categorical(orig_name, s_raw) and (orig_name not in ORDINAL_TREND_VARS)

        if is_cat:
            rows_out.append({
                'Characteristic': f"{_display_name(orig_name, mode='plot')}, n(%)",
                col_m0: '',
                col_m1: '',
                'OR (univariable)': '',
                'OR (multivariable)': '',
            })

            levels = [str(x) for x in pd.unique(s_clean.dropna())]
            if not levels:
                continue

            ref = uni_ref_map.get(orig_name)
            if (not ref) and (orig_name in REFERENCE_MAP):
                cand = str(REFERENCE_MAP[orig_name])
                if cand in levels:
                    ref = cand
            if not ref:
                ref = sorted(levels)[0]

            ordered_levels = [ref] + sorted([lv for lv in levels if lv != ref])

            denom0 = int(((t == 0) & s_clean.notna()).sum())
            denom1 = int(((t == 1) & s_clean.notna()).sum())
            for lv in ordered_levels:
                cnt0 = int(((t == 0) & (s_clean == lv)).sum())
                cnt1 = int(((t == 1) & (s_clean == lv)).sum())
                m0_txt = _format_count_pct(cnt0, denom0)
                m1_txt = _format_count_pct(cnt1, denom1)

                if lv == ref:
                    uni_txt = 'refer'
                    mv_txt = 'refer' if (orig_name in mv_ref_map) else ''
                else:
                    u = uni_lookup.get((orig_name, lv))
                    uni_txt = _format_or_ci_p(u['OR值'], u['95% CI 下限'], u['95% CI 上限'], u['P值']) if u else ''
                    m = mv_lookup.get((orig_name, lv))
                    mv_txt = _format_or_ci_p(m['OR值'], m['95% CI 下限'], m['95% CI 上限'], m['P值']) if m else ''

                rows_out.append({
                    'Characteristic': str(lv),
                    col_m0: m0_txt,
                    col_m1: m1_txt,
                    'OR (univariable)': uni_txt,
                    'OR (multivariable)': mv_txt,
                })
        else:
            x0 = s_clean[t == 0]
            x1 = s_clean[t == 1]
            m0_txt = _median_iqr_text(x0)
            m1_txt = _median_iqr_text(x1)

            u = uni_lookup.get((orig_name, '-'))
            uni_txt = _format_or_ci_p(u['OR值'], u['95% CI 下限'], u['95% CI 上限'], u['P值']) if u else ''
            m = mv_lookup.get((orig_name, '-'))
            mv_txt = _format_or_ci_p(m['OR值'], m['95% CI 下限'], m['95% CI 上限'], m['P值']) if m else ''

            rows_out.append({
                'Characteristic': _display_name(orig_name, mode='plot'),
                col_m0: m0_txt,
                col_m1: m1_txt,
                'OR (univariable)': uni_txt,
                'OR (multivariable)': mv_txt,
            })

    if rows_out:
        out_df = pd.DataFrame(rows_out)
        out_file = '单因素+多因素汇总总表.xlsx'
        try:
            out_df.to_excel(out_file, index=False)
            print(f"汇总总表已保存至: {out_file}")
        except PermissionError:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            alt = f"单因素+多因素汇总总表_{ts}.xlsx"
            out_df.to_excel(alt, index=False)
            print(f"提示：无法写入 '{out_file}'（可能正被 Excel 打开）。已改为保存到: {alt}")
    else:
        print('未生成汇总总表：可能 MULTIVAR_FEATURES_ORIG 中的变量均不在数据中。')
except Exception as e:
    print(f"汇总总表生成失败（不影响单/多因素结果导出）。原因: {str(e)[:200]}")