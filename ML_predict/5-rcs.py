import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from patsy import build_design_matrices, dmatrix  # cr 在公式字符串里自动可用
from scipy.stats import chi2  # 新增：用于计算卡方检验P值
from scipy.signal import savgol_filter  # 新增：用于平滑 CI 锯齿
import os
import json
from typing import Any, Dict, List, Optional, Tuple
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning, PerfectSeparationError, HessianInversionWarning

import string

try:
    from display_names import display_name as _display_name, display_level as _display_level
except Exception:
    def _display_name(name: str, mode: str = "plot") -> str:  # type: ignore
        return str(name)

    def _display_level(var: str, level: str, mode: str = "plot") -> str:  # type: ignore
        return f"{var}({level})"

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

INPUT_FILE_PATH = r'tmp\merged_1634.xlsx'
OUTPUT_DIR = os.path.join('RCS', 'rcs_plots')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SUMMARY_PATH = os.path.join(OUTPUT_DIR, 'rcs_summary.xlsx')

# 仅挑选以下变量进行 RCS/曲线分析（按用户要求，已去除分类变量）
RCS_FEATURES: List[str] = [
    "Treg cells %",
    "LDH",
    "CD4+ count",
    "CEA",
    "CA199",
    "CA724",
    "RBC",

]

# 将分期/分级按"有序数值趋势项"处理（当前列表为空，因分类变量已移除）
ORDINAL_TREND_VARS: List[str] = []



# 曲线稳定性配置
RCS_DEFAULT_DF = 4          # 默认样条自由度（越大越容易波动）
RCS_MIN_UNIQUE_FOR_SPLINE = 10  # 唯一值少于该阈值时，不做 RCS（改线性/分类）
GRID_POINTS = 300           # 预测网格点数（越大越平滑，但耗时略增）
TRIM_Q_LOW = 0.01           # 画图与拟合使用的下分位数裁剪
TRIM_Q_HIGH = 0.99          # 画图与拟合使用的上分位数裁剪
Z_CLIP = 8.0                # 连续变量标准化后的截断，减少溢出
ETA_CLIP = 20.0             # 线性预测值截断，避免 exp 溢出
CI_SMOOTH_WINDOW = 31       # CI 平滑窗口大小（必须为奇数，越大越平滑）
CI_SMOOTH_POLYORDER = 3     # Savitzky-Golay 多项式阶数
CROSSING_VLINE_YMAX = 0.90  # 交点标注竖线的相对高度，避免遮挡左上角统计注释
REF_VLINE_YMAX = 0.28       # 参考值竖线高度；若与交点重合则不重复绘制
XLINE_MERGE_TOL = 1e-3      # 判断参考值竖线与交点竖线是否重合的容差

# 加载列名映射（中文 -> 英文），在绘图时用于替换标签
MAPPING_PATH = os.path.join(os.path.dirname(__file__), 'merge', 'column_name_mapping.json')
try:
    with open(MAPPING_PATH, 'r', encoding='utf-8') as mf:
        COLUMN_NAME_MAPPING = json.load(mf)
except Exception:
    COLUMN_NAME_MAPPING = {}

def translate_label(name: str) -> str:
    """将中文列名翻译为映射中的英文，找不到则返回原名。"""
    if name is None:
        return name
    mapped = COLUMN_NAME_MAPPING.get(name, name)
    # 再过一层统一展示名（去掉 num_/cat_ 前缀、连续变量补单位等）
    return _display_name(mapped, mode='plot')


def coerce_binary_target(series: pd.Series) -> pd.Series:
    """将二分类因变量稳健转换为 0/1 float。"""
    s = series.copy()
    # bool
    if s.dropna().map(type).isin([bool]).all():
        return s.astype(int).astype('float64')
    # numeric
    s_num = pd.to_numeric(s, errors='coerce')
    uniq = sorted(set(s_num.dropna().unique().tolist()))
    if set(uniq).issubset({0.0, 1.0}) and len(uniq) in (1, 2):
        return s_num.astype('float64')
    # string map
    s_str = s.astype(str).str.strip().replace({'nan': np.nan, 'NaN': np.nan, 'None': np.nan, '': np.nan})
    mapping_known = {
        '0': 0, '1': 1,
        'false': 0, 'true': 1,
        '否': 0, '是': 1,
        '无': 0, '有': 1,
        '阴性': 0, '阳性': 1,
        'negative': 0, 'positive': 1,
        'no': 0, 'yes': 1,
    }
    mapped = s_str.str.lower().map(mapping_known)
    if mapped.notna().sum() > 0 and mapped.dropna().nunique() in (1, 2):
        return mapped.astype('float64')

    raise ValueError(f"因变量不是明显二分类(0/1): {sorted(pd.unique(s_str.dropna()).tolist())[:10]}")


def parse_ordinal_trend(series: pd.Series) -> pd.Series:
    """将分期/分级等有序变量解析为数值等级（从字符串中提取数字）。"""
    s = series.copy()
    s = s.mask(s == '/', np.nan)
    as_num = pd.to_numeric(s, errors='coerce')
    if int(as_num.notna().sum()) == int(s.notna().sum()):
        return as_num
    s_str = s.astype(str).str.strip().replace({'nan': np.nan, 'NaN': np.nan, 'None': np.nan, '': np.nan})
    extracted = pd.to_numeric(s_str.str.extract(r'(\d+)')[0], errors='coerce')
    return as_num.fillna(extracted)


def winsorize_series(s: pd.Series, q_low: float, q_high: float) -> Tuple[pd.Series, float, float]:
    """对连续变量做分位数截断，降低极端值导致的曲线波动。"""
    s_num = pd.to_numeric(s, errors='coerce').astype('float64')
    non_na = s_num.dropna()
    if non_na.empty:
        return s_num, np.nan, np.nan
    lo = float(non_na.quantile(q_low))
    hi = float(non_na.quantile(q_high))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return s_num, lo, hi
    return s_num.clip(lo, hi), lo, hi


def sigmoid_stable(eta: np.ndarray) -> np.ndarray:
    """稳定的 sigmoid：先对 eta 截断，避免 exp 溢出。"""
    eta = np.clip(eta, -ETA_CLIP, ETA_CLIP)
    return 1.0 / (1.0 + np.exp(-eta))


def smooth_ci(arr: np.ndarray, window: int = CI_SMOOTH_WINDOW, polyorder: int = CI_SMOOTH_POLYORDER) -> np.ndarray:
    """对 CI 边界应用 Savitzky-Golay 平滑，消除锯齿。

    原理：用局部多项式拟合替代原始点，保留趋势但去掉高频波动。
    - window 必须为奇数，越大越平滑
    - polyorder 越小越平滑，但太小会失去曲线趋势
    """
    if not np.isfinite(arr).all():
        return arr
    n = len(arr)
    # 窗口不能超过数据长度，且必须为奇数
    w = min(window, n)
    if w % 2 == 0:
        w -= 1
    if w < polyorder + 2:
        return arr
    return savgol_filter(arr, window_length=w, polyorder=polyorder)


def zscore_clip(s: pd.Series, clip: float) -> Tuple[pd.Series, float, float]:
    s_num = pd.to_numeric(s, errors='coerce').astype('float64')
    mean = float(s_num.mean())
    std = float(s_num.std(ddof=0))
    if not np.isfinite(std) or std <= 0:
        return pd.Series(0.0, index=s_num.index, dtype='float64'), mean, std
    z = (s_num - mean) / std
    return z.clip(-clip, clip), mean, std


def _is_categorical_series(s: pd.Series) -> bool:
    if s.dtype == 'object' or s.dtype.name == 'category':
        return True
    nunique = int(s.dropna().nunique())
    return nunique > 0 and nunique <= 10


def _choose_rcs_df(n_unique: int, n_used: int) -> int:
    # 经验规则：样本/唯一值较少时降低 df，避免过拟合造成波动
    if n_unique < 30 or n_used < 300:
        return 3
    return RCS_DEFAULT_DF

def calculate_lrt(full_model, reduced_model):
    """Return the LRT p-value between two nested models."""
    lr_stat = -2 * (reduced_model.llf - full_model.llf)
    df_diff = full_model.df_model - reduced_model.df_model
    if df_diff <= 0:
        return 1.0
    return chi2.sf(lr_stat, df_diff)


def fit_logit_robust(formula: str, data: pd.DataFrame):
    """更稳健的 Logit 拟合：默认 -> lbfgs 重试 -> 正则化回退。

    目的：减少 ConvergenceWarning/奇异矩阵导致的锯齿与无效 CI。
    """
    # 1) 默认（把不收敛当作异常；HessianInversionWarning 不致命，允许继续）
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('error', category=ConvergenceWarning)
            warnings.filterwarnings('error', category=PerfectSeparationError)
            warnings.filterwarnings('ignore', category=HessianInversionWarning)
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            return smf.logit(formula=formula, data=data).fit(
                disp=0, maxiter=2000
            )
    except Exception:
        pass

    # 2) lbfgs
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('error', category=ConvergenceWarning)
            warnings.filterwarnings('ignore', category=HessianInversionWarning)
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            return smf.logit(formula=formula, data=data).fit(
                disp=0, method='lbfgs', maxiter=5000
            )
    except Exception:
        pass

    # 3) 正则化回退（通常能给出可用系数）
    model = smf.logit(formula=formula, data=data)
    return model.fit_regularized(
        alpha=0.5, L1_wt=0.0, disp=0, maxiter=5000
    )


def safe_add_legend(ax: plt.Axes):
    handles, labels = ax.get_legend_handles_labels()
    labels = [l for l in labels if l and not l.startswith('_')]
    if labels:
        ax.legend(loc='best')


def get_covariance_matrix_fallback(model) -> Tuple[Optional[np.ndarray], str]:
    """尽量获取参数协方差矩阵。

    返回 (cov, source)：
    - cov: 协方差矩阵或 None
    - source: 'cov_params' | 'pinv_hessian' | 'none'

    说明：
    - 部分变量会出现 Hessian 逆失败（HessianInversionWarning）或正则化拟合，导致 cov_params 不可用。
    - 这里用 -Hessian 的 Moore-Penrose 伪逆作为近似协方差，使 CI 仍可计算（但在共线/分离严重时 CI 可能很宽）。
    """
    try:
        cov = np.asarray(model.cov_params())
        if cov.ndim == 2 and cov.shape[0] == cov.shape[1] and np.isfinite(cov).any():
            return cov, 'cov_params'
    except Exception:
        pass

    try:
        hess = np.asarray(model.model.hessian(model.params))
        if hess.ndim == 2 and hess.shape[0] == hess.shape[1]:
            cov = np.linalg.pinv(-hess)
            return cov, 'pinv_hessian'
    except Exception:
        pass

    return None, 'none'


def make_montage_3x3(
    image_paths: List[str],
    out_path: str,
    nrows: int = 3,
    ncols: int = 3,
    title: Optional[str] = None,
    labels: bool = True,
    dpi: int = 300,
    panel_w: float = 5.2,
    panel_h: float = 4.3,
) -> str:
    """将单张 PNG 图拼成 nrows×ncols 的拼图并保存。

    - 自动去掉不存在的文件
    - 每个子图左上角可加 A/B/C... 标记（模仿示例图）
    """
    paths = [p for p in image_paths if isinstance(p, str) and p and os.path.exists(p)]
    if not paths:
        raise ValueError("没有可用于拼图的图片路径（文件不存在或列表为空）。")

    max_panels = nrows * ncols
    if len(paths) > max_panels:
        paths = paths[:max_panels]

    dpi = int(max(300, dpi))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * panel_w, nrows * panel_h))
    axes_arr = np.asarray(axes).reshape(-1)

    panel_letters = list(string.ascii_uppercase)
    for i, ax in enumerate(axes_arr):
        ax.axis('off')
        if i >= len(paths):
            continue
        img = plt.imread(paths[i])
        ax.imshow(img)
        if labels:
            ax.text(
                0.01,
                0.99,
                panel_letters[i] if i < len(panel_letters) else str(i + 1),
                transform=ax.transAxes,
                ha='left',
                va='top',
                fontsize=16,
                fontweight='bold',
                color='black',
                bbox=dict(boxstyle='square,pad=0.15', facecolor='white', edgecolor='none', alpha=0.9),
            )

    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold')

    plt.tight_layout(rect=(0, 0, 1, 0.98 if title else 1))
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def export_montages_from_results(
    results: List[Dict[str, Any]],
    out_dir: str,
    target_col: str,
    nrows: int = 3,
    ncols: int = 3,
    dpi: int = 300,
) -> List[str]:
    """把本次运行生成的所有单图，按 3×3 分页导出拼图。"""
    # 按 P_overall 从小到大排序（NaN 放最后），并用 focus 作为稳定的次级排序
    sortable: List[Tuple[float, str, str]] = []
    for r in results:
        if not isinstance(r, dict):
            continue
        p = r.get('p_overall')
        focus = str(r.get('focus', ''))
        save_name = r.get('save_name')
        if not isinstance(save_name, str) or not save_name:
            continue
        if not os.path.exists(save_name):
            continue
        try:
            p_val = float(p)
            if not np.isfinite(p_val):
                p_val = float('inf')
        except Exception:
            p_val = float('inf')
        sortable.append((p_val, focus, save_name))

    sortable.sort(key=lambda x: (x[0], x[1]))
    image_paths = [t[2] for t in sortable]
    if not image_paths:
        return []

    per_page = nrows * ncols
    out_paths: List[str] = []
    total_pages = int(np.ceil(len(image_paths) / per_page))
    for page_idx in range(total_pages):
        chunk = image_paths[page_idx * per_page : (page_idx + 1) * per_page]
        out_path = os.path.join(out_dir, f"RCS_MONTAGE_{target_col}_page{page_idx + 1}.png")
        title = f"RCS Montage ({nrows}×{ncols}) - {translate_label(target_col)}  (Page {page_idx + 1}/{total_pages})"
        saved = make_montage_3x3(
            chunk,
            out_path,
            nrows=nrows,
            ncols=ncols,
            title=title,
            labels=True,
            dpi=dpi,
        )
        out_paths.append(saved)
    return out_paths


def generate_rcs_plot(
    df,
    target_col,
    focus_col,
    adjust_cols=None,
    ref_value=None,
):
    """生成单变量曲线并返回关键统计量。

    - 连续变量：优先用 RCS（restricted cubic spline）
    - 唯一值过少/明显离散：回退为线性（趋势项）或分类（点/误差棒），避免锯齿和不稳定
    """

    adjust_cols = adjust_cols or []

    safe_focus = "FocusVar"
    safe_target = "TargetVar"

    missing_cols = [col for col in (target_col, focus_col) if col not in df.columns]
    if missing_cols:
        raise ValueError(f"数据缺少必要列: {missing_cols}")

    rename_dict = {focus_col: safe_focus, target_col: safe_target}
    df_clean = df.rename(columns=rename_dict).copy()

    # 目标变量二分类
    df_clean[safe_target] = coerce_binary_target(df_clean[safe_target])

    # 处理 focus 变量：有序趋势/连续/分类
    focus_raw = df_clean[safe_focus]
    if focus_col in ORDINAL_TREND_VARS:
        focus_num = parse_ordinal_trend(focus_raw)
        df_clean[safe_focus] = focus_num
        focus_kind = 'ordinal-trend'
    else:
        focus_kind = 'auto'

    # 连续变量：数值化 + winsorize（先不决定是否 spline）
    if focus_kind != 'ordinal-trend' and not _is_categorical_series(focus_raw):
        focus_num, clip_lo, clip_hi = winsorize_series(focus_raw, TRIM_Q_LOW, TRIM_Q_HIGH)
        df_clean[safe_focus] = focus_num
    else:
        clip_lo, clip_hi = np.nan, np.nan

    df_clean = df_clean.dropna(subset=[safe_target, safe_focus])
    if df_clean.empty:
        raise ValueError("清洗后数据为空，无法拟合模型。")

    # 统一计算 ref
    if ref_value is None:
        if _is_categorical_series(df_clean[safe_focus]):
            ref_value = str(df_clean[safe_focus].mode().iloc[0])
        else:
            ref_value = float(df_clean[safe_focus].median())

    safe_target_name = target_col.replace('/', '_')
    safe_focus_name = focus_col.replace('/', '_')
    save_name = os.path.join(OUTPUT_DIR, f"RCS_{safe_target_name}_{safe_focus_name}.png")

    if isinstance(ref_value, (int, float, np.floating)):
        print(f"Ref value: {float(ref_value):.4g}")
    else:
        print(f"Ref value: {ref_value}")
    print("Running statistical tests...")

    if adjust_cols:
        covariates_str = " + ".join(adjust_cols)
        formula_cov = f" + {covariates_str}"
    else:
        covariates_str = "1"
        formula_cov = ""

    # 先拟合 null（用于 overall）
    formula_null = f"{safe_target} ~ {covariates_str}"
    model_null = fit_logit_robust(formula_null, df_clean)

    focus_nunique = int(df_clean[safe_focus].dropna().nunique())
    n_used = int(len(df_clean))

    # A) 分类变量：画点/误差棒（避免把分类型变量做 spline 产生锯齿）
    if focus_col not in ORDINAL_TREND_VARS and _is_categorical_series(df_clean[safe_focus]):
        cov_src = 'categorical'  # 分类变量不计算协方差/CI
        clip_lo, clip_hi = np.nan, np.nan  # 分类变量无裁剪范围
        # 强制转字符串，保证水平稳定
        df_clean[safe_focus] = df_clean[safe_focus].astype(str).str.strip().replace({'nan': np.nan, 'NaN': np.nan, 'None': np.nan, '': np.nan})
        df_clean = df_clean.dropna(subset=[safe_focus])
        levels = sorted(df_clean[safe_focus].unique().tolist())
        ref_level = str(ref_value)
        if ref_level not in levels and levels:
            ref_level = levels[0]
        formula_cat = f"{safe_target} ~ C({safe_focus}, Treatment(reference='{ref_level}')){formula_cov}"
        model_cat = fit_logit_robust(formula_cat, df_clean)
        p_overall = float(calculate_lrt(model_cat, model_null))
        p_nonlinear = np.nan

        rows: List[Dict[str, Any]] = []
        for lvl in levels:
            if lvl == ref_level:
                rows.append({'level': lvl, 'or': 1.0, 'lo': 1.0, 'hi': 1.0})
                continue
            term = f"C({safe_focus}, Treatment(reference='{ref_level}'))[T.{lvl}]"
            if term not in model_cat.params.index:
                continue
            coef = float(model_cat.params[term])
            try:
                ci = model_cat.conf_int().loc[term].tolist()
                lo_i, hi_i = float(np.exp(ci[0])), float(np.exp(ci[1]))
            except Exception:
                lo_i, hi_i = np.nan, np.nan
            rows.append({'level': lvl, 'or': float(np.exp(coef)), 'lo': lo_i, 'hi': hi_i})

        fig, ax = plt.subplots(figsize=(9, 7))
        xs = np.arange(len(rows))
        ors = np.array([r['or'] for r in rows], dtype='float64')
        lo = np.array([r['lo'] for r in rows], dtype='float64')
        hi = np.array([r['hi'] for r in rows], dtype='float64')
        yerr = None
        if np.isfinite(lo).all() and np.isfinite(hi).all():
            yerr = [ors - lo, hi - ors]
        ax.errorbar(
            xs,
            ors,
            yerr=yerr,
            fmt='o',
            color='#d62728',
            ecolor='#1f77b4',
            capsize=3,
            label='Odds Ratio',
        )
        ax.axhline(y=1, color='gray', linestyle='--', linewidth=1)
        ax.set_xticks(xs)
        ax.set_xticklabels([r['level'] for r in rows], rotation=45, ha='right')
        method_label = 'categorical'

    # B) 连续/趋势：优先 RCS，否则线性
    else:
        # 为数值稳定：对用于拟合/预测的 focus 做 z-score 标准化并截断
        focus_model = f"{safe_focus}__z"
        z_s, z_mean, z_std = zscore_clip(df_clean[safe_focus], Z_CLIP)
        df_clean[focus_model] = z_s

        # 唯一值过少：线性趋势更稳
        use_spline = focus_nunique >= RCS_MIN_UNIQUE_FOR_SPLINE
        # 分期/分级通常唯一值很少，默认线性（趋势项）更稳定
        if focus_col in ORDINAL_TREND_VARS and focus_nunique <= 6:
            use_spline = False

        # 画图/预测范围：用分位数裁剪，避免极端尾部让曲线抖动
        x_non_na = df_clean[safe_focus].dropna().astype('float64')
        x_low = float(x_non_na.quantile(TRIM_Q_LOW))
        x_high = float(x_non_na.quantile(TRIM_Q_HIGH))
        if not np.isfinite(x_low) or not np.isfinite(x_high) or x_low >= x_high:
            x_low = float(x_non_na.min())
            x_high = float(x_non_na.max())
        x_range = np.linspace(x_low, x_high, GRID_POINTS)
        pred_data = pd.DataFrame({safe_focus: x_range})
        # 对预测网格做同样 z-score
        if np.isfinite(z_std) and z_std > 0:
            pred_data[focus_model] = ((pred_data[safe_focus].astype('float64') - z_mean) / z_std).clip(-Z_CLIP, Z_CLIP)
        else:
            pred_data[focus_model] = 0.0

        # adjust covariates：用 df_clean 的均值/众数，保证公式里变量齐全
        for col in adjust_cols:
            clean_col = col.replace("C(", "").replace(")", "")
            if clean_col in df_clean.columns:
                series = df_clean[clean_col]
                if _is_categorical_series(series):
                    pred_data[clean_col] = series.astype(str).mode().iloc[0]
                else:
                    pred_data[clean_col] = float(pd.to_numeric(series, errors='coerce').mean())

        if use_spline:
            df_rcs = _choose_rcs_df(focus_nunique, n_used)
            formula_rcs = f"{safe_target} ~ cr({focus_model}, df={df_rcs}){formula_cov}"
            model_rcs = fit_logit_robust(formula_rcs, df_clean)
            formula_linear = f"{safe_target} ~ {focus_model}{formula_cov}"
            model_linear = fit_logit_robust(formula_linear, df_clean)
            try:
                p_overall = float(calculate_lrt(model_rcs, model_null))
            except Exception:
                p_overall = np.nan
            try:
                p_nonlinear = float(calculate_lrt(model_rcs, model_linear))
            except Exception:
                p_nonlinear = np.nan
            method_label = f"rcs(df={df_rcs})"
            model_for_ref = model_rcs
        else:
            formula_linear = f"{safe_target} ~ {focus_model}{formula_cov}"
            model_linear = fit_logit_robust(formula_linear, df_clean)
            try:
                p_overall = float(calculate_lrt(model_linear, model_null))
            except Exception:
                p_overall = np.nan
            p_nonlinear = np.nan
            method_label = "linear"
            model_for_ref = model_linear

        # 预测（彻底避免 statsmodels 内部 exp/log 的 RuntimeWarning）：
        # 1) 用 patsy design_info 构造设计矩阵
        X_new = build_design_matrices([model_for_ref.model.data.design_info], pred_data, return_type='dataframe')[0]
        eta = model_for_ref.model.predict(model_for_ref.params, exog=X_new, which='linear')
        prob = sigmoid_stable(np.asarray(eta, dtype='float64'))

        # 2) CI（优先 cov_params；失败则用伪逆 Hessian 回退；仍失败则置空）
        ci_lower_prob = np.full_like(prob, np.nan, dtype='float64')
        ci_upper_prob = np.full_like(prob, np.nan, dtype='float64')
        cov, cov_src = get_covariance_matrix_fallback(model_for_ref)
        if cov is not None:
            try:
                Xv = np.asarray(X_new, dtype='float64')
                # var(eta) = x^T Cov x
                var_eta = np.einsum('ij,jk,ik->i', Xv, cov, Xv)
                se_eta = np.sqrt(np.maximum(var_eta, 0.0))
                eta_lo = np.clip(np.asarray(eta, dtype='float64') - 1.96 * se_eta, -ETA_CLIP, ETA_CLIP)
                eta_hi = np.clip(np.asarray(eta, dtype='float64') + 1.96 * se_eta, -ETA_CLIP, ETA_CLIP)
                ci_lower_prob = sigmoid_stable(eta_lo)
                ci_upper_prob = sigmoid_stable(eta_hi)
            except Exception:
                cov_src = 'none'
        else:
            cov_src = 'none'

        pred_summary = pd.DataFrame(
            {
                'predicted': prob,
                'ci_lower': ci_lower_prob,
                'ci_upper': ci_upper_prob,
            }
        )
        prob = np.clip(pred_summary['predicted'], 1e-6, 1 - 1e-6)
        prob_ci_lower = np.clip(pred_summary['ci_lower'], 1e-6, 1 - 1e-6)
        prob_ci_upper = np.clip(pred_summary['ci_upper'], 1e-6, 1 - 1e-6)

        # 对 CI 应用平滑，消除协方差不稳定导致的锯齿
        prob_ci_lower = smooth_ci(np.asarray(prob_ci_lower))
        prob_ci_upper = smooth_ci(np.asarray(prob_ci_upper))

        odds = prob / (1 - prob)
        odds_ci_lower = prob_ci_lower / (1 - prob_ci_lower)
        odds_ci_upper = prob_ci_upper / (1 - prob_ci_upper)

        ref_row = pd.DataFrame({safe_focus: [float(ref_value)]})
        if np.isfinite(z_std) and z_std > 0:
            ref_row[focus_model] = float(np.clip((float(ref_value) - z_mean) / z_std, -Z_CLIP, Z_CLIP))
        else:
            ref_row[focus_model] = 0.0
        X_ref = build_design_matrices([model_for_ref.model.data.design_info], ref_row, return_type='dataframe')[0]
        eta_ref = float(model_for_ref.model.predict(model_for_ref.params, exog=X_ref, which='linear')[0])
        ref_prob = float(np.clip(sigmoid_stable(np.array([eta_ref]))[0], 1e-6, 1 - 1e-6))
        ref_odds = ref_prob / (1 - ref_prob)

        or_values = odds / ref_odds
        ci_lower = odds_ci_lower / ref_odds
        ci_upper = odds_ci_upper / ref_odds

        def _find_crossings(x: np.ndarray, y: np.ndarray, level: float = 1.0) -> List[float]:
            """返回曲线 y 与水平线 y=level 的所有交点 x（线性插值）。"""
            x = np.asarray(x, dtype='float64')
            y = np.asarray(y, dtype='float64')
            if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 2:
                return []
            out: List[float] = []
            tol = 1e-9
            yy = y - float(level)

            # 只在相邻点都有限时寻找交点
            finite = np.isfinite(x) & np.isfinite(yy)
            idx = np.where(finite[:-1] & finite[1:])[0]
            for i in idx:
                x0, x1 = float(x[i]), float(x[i + 1])
                y0, y1 = float(yy[i]), float(yy[i + 1])

                # 端点恰好在 level 上
                if abs(y0) <= tol:
                    out.append(x0)
                if abs(y1) <= tol:
                    out.append(x1)

                # 符号变化：线性插值求交点
                if (y0 > 0 and y1 < 0) or (y0 < 0 and y1 > 0):
                    denom = (y1 - y0)
                    if abs(denom) > 0:
                        t = (0.0 - y0) / denom
                        if 0.0 <= t <= 1.0:
                            out.append(x0 + t * (x1 - x0))

            # 去重（保持顺序）
            uniq: List[float] = []
            for v in sorted(out):
                if not uniq or abs(v - uniq[-1]) > 1e-6:
                    uniq.append(v)
            return uniq

        # 计算 OR=1 的交点（若有多个，通常是 0~2 个）
        crossing_xs = _find_crossings(x_range, np.asarray(or_values, dtype='float64'), level=1.0)

        # 针对 CA199：两个交点往往很近且第一个不明显，按用户要求只显示第二个（x 更大者）
        if focus_col == 'CA199' and len(crossing_xs) >= 2:
            crossing_xs = [max(crossing_xs)]

        fig, ax = plt.subplots(figsize=(8, 7))
        if np.isfinite(ci_lower).all() and np.isfinite(ci_upper).all():
            ax.fill_between(
                x_range,
                ci_lower,
                ci_upper,
                color='#1f77b4',
                alpha=0.15,
                label='95% CI',
                interpolate=True,
            )
        ax.plot(x_range, or_values, color='#d62728', linewidth=2.5, label='Odds Ratio')
        ax.axhline(y=1, color='gray', linestyle='--', linewidth=1)

        # 标注与 OR=1 的交点：画竖线并标出 x 值（仿示例图）
        for cx in crossing_xs:
            ax.axvline(
                x=float(cx),
                ymin=0.0,
                ymax=CROSSING_VLINE_YMAX,
                color='gray',
                linestyle='-',
                linewidth=1,
            )
            ax.annotate(
                f"{float(cx):.2f}",
                xy=(float(cx), 1.0),
                xytext=(0, 6),
                textcoords='offset points',
                ha='center',
                va='bottom',
                fontsize=11,
                color='black',
            )

        if isinstance(ref_value, (int, float, np.floating)):
            ref_x = float(ref_value)
            overlaps_crossing = any(abs(ref_x - float(cx)) <= XLINE_MERGE_TOL for cx in crossing_xs)
            if not overlaps_crossing:
                ax.axvline(
                    x=ref_x,
                    ymin=0.0,
                    ymax=REF_VLINE_YMAX,
                    color='gray',
                    linestyle='--',
                    linewidth=1,
                )
        sns.rugplot(df_clean[safe_focus], height=0.03, color='black', alpha=0.25, ax=ax)

    # 使用映射后的英文标签用于图形展示
    focus_label = translate_label(focus_col)
    target_label = translate_label(target_col)

    # 统一出图样式：
    # - 去掉标题
    # - 去掉图例
    # - 将结果（Poverall / Pnon-linear）放到图内左上角，不显示方法标签
    ax.set_title('')
    if np.isfinite(p_overall) and np.isfinite(p_nonlinear):
        stat_text = f"Poverall = {p_overall:.4f}\nPnon-linear = {p_nonlinear:.4f}"
    elif np.isfinite(p_overall):
        stat_text = f"Poverall = {p_overall:.4f}"
    else:
        stat_text = ""
    if stat_text:
        ax.text(
            0.02,
            0.98,
            stat_text,
            transform=ax.transAxes,
            ha='left',
            va='top',
            fontsize=13,
            color='black',
        )
    ax.set_xlabel(focus_label, fontsize=12)
    ax.set_ylabel('Odds Ratio (95%CI)', fontsize=12)
    # 去掉图例（包括 95% CI / Odds Ratio / Ref）
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
    ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    # plt.show()
    plt.close(fig)

    print(f"Plot saved: {save_name}")

    return {
        'target': target_col,
        'focus': focus_col,
        'p_overall': p_overall,
        'p_nonlinear': p_nonlinear,
        'ref_value': ref_value,
        'method': method_label,
        'n': int(len(df_clean)),
        'n_unique': int(df_clean[safe_focus].dropna().nunique()),
        'x_min': (
            float(pd.to_numeric(df_clean[safe_focus], errors='coerce').min())
            if not _is_categorical_series(df_clean[safe_focus])
            else np.nan
        ),
        'x_max': (
            float(pd.to_numeric(df_clean[safe_focus], errors='coerce').max())
            if not _is_categorical_series(df_clean[safe_focus])
            else np.nan
        ),
        'clip_lo': float(clip_lo) if np.isfinite(clip_lo) else np.nan,
        'clip_hi': float(clip_hi) if np.isfinite(clip_hi) else np.nan,
        'ci_source': cov_src if method_label != 'categorical' else 'categorical',
        'save_name': save_name,
    }


if __name__ == "__main__":

    file_path = INPUT_FILE_PATH
    df = pd.read_excel(file_path)

    # 只保留用户指定变量（存在于数据中）
    missing = [c for c in RCS_FEATURES if c not in df.columns]
    if missing:
        print(f"提示：以下变量在数据中不存在，将跳过: {missing}")
    features_present = [c for c in RCS_FEATURES if c in df.columns]

    target_col = 'Metastasis'
    total = len(features_present)

    results = []

    for idx, focus in enumerate(features_present, start=1):
        # adjust_cols = [col for col in FORCE_CONTINUOUS if col != focus]
        adjust_cols = []
        print(f"[{idx}/{total}] 正在处理: {focus}")
        try:
            res = generate_rcs_plot(
                df,
                target_col=target_col,
                focus_col=focus,
                adjust_cols=adjust_cols,
            )
            results.append(res)
        except Exception as exc:
            print(f"--> 跳过 {focus}: {exc}")

    if results:
        summary_df = pd.DataFrame(results)
        summary_df.to_excel(SUMMARY_PATH, index=False)
        print(f"汇总表已保存至: {SUMMARY_PATH}")

        # 额外输出：将单图合并为 3×3 拼图（模仿示例图）
        try:
            montage_paths = export_montages_from_results(
                results=results,
                out_dir=OUTPUT_DIR,
                target_col=target_col,
                nrows=3,
                ncols=3,
                dpi=300,
            )
            if montage_paths:
                print("拼图已保存:")
                for p in montage_paths:
                    print(f"- {p}")
            else:
                print("未生成拼图：未找到有效的单图输出文件。")
        except Exception as exc:
            print(f"拼图生成失败（不影响单图/汇总表输出）: {exc}")
    else:
        print("未生成任何有效结果，未输出汇总表。")