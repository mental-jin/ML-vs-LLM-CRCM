from __future__ import annotations

import os
import json
import pandas as pd
import numpy as np
import miceforest as mf
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from pathlib import Path
from scipy import stats
from scipy.stats import shapiro, normaltest, kstest

import sklearn
from sklearn.model_selection import KFold, StratifiedKFold

from display_names import display_name as _display_name
from display_names import display_names as _display_names


def normalize_categorical_series(s: pd.Series) -> pd.Series:
    """统一规范化分类变量的取值，避免 0 / 0.0 / ' 0 ' 这类重复水平。

    说明：miceforest 对 pandas category 的 categories 很敏感；若 categories 是 [0.0, 1.0]
    但后续兜底填充值是 '0'/'1'，会因为不在 categories 内而再次变回缺失，出现“插补后仍空白”。
    """
    if s is None:
        return s

    # 先转字符串并 strip（注意：pd.NA 会变成 '<NA>'）
    s_str = s.astype(str).str.strip()
    s_str = s_str.replace({
        "/": np.nan,
        "": np.nan,
        "nan": np.nan,
        "NaN": np.nan,
        "<NA>": np.nan,
        "None": np.nan,
        "NONE": np.nan,
    })

    # 对可解析为数值的类别进行统一：整数 -> '0'/'1'；非整数 -> %.10g
    num = pd.to_numeric(s_str, errors="coerce")
    out = s_str.copy()
    is_num = num.notna() & np.isfinite(num)
    if is_num.any():
        is_int = is_num & np.isclose(num, np.round(num))
        if is_int.any():
            out.loc[is_int] = num.loc[is_int].round().astype("Int64").astype(str)
        is_float = is_num & ~is_int
        if is_float.any():
            out.loc[is_float] = num.loc[is_float].map(lambda x: f"{x:.10g}")
    return out


def normalized_categorical_value_counts(s: pd.Series) -> pd.Series:
    """对分类变量先做水平规范化，再统计频数。"""
    normalized = normalize_categorical_series(s).dropna()
    if normalized.empty:
        return pd.Series(dtype="int64")
    return normalized.value_counts()

NUM_IMPUTATION_DATASETS = 5  # 插补数据集数量
NUM_ITERATIONS = 5  # 每个数据集的迭代次数
MISSING_THRESHOLD = 0.7  # 缺失率阈值（超过此值的列将被删除）

# 插补策略：
# - "foldwise_oof": 按折训练 MICE，并只对该折验证集插补，最终拼成 out-of-fold(OOF) 的 combined_imputed.xlsx（性能评估推荐/无泄漏）
# - "global_fullfit": 在全量数据上做一次 MICE 并合并（仅建议用于最终全量训练/部署，不建议用于性能评估）
IMPUTATION_MODE = "foldwise_oof"

# fold-wise MICE 的 CV 设置
CV_N_SPLITS = 5
CV_SHUFFLE = True
CV_RANDOM_STATE = 42

# 若目标列可用（且无缺失/类别数合理），优先用 StratifiedKFold 以保持事件率
CV_STRATIFY_TARGET = "Metastasis"

# fold-wise MICE 的资源友好默认参数（OOF 性能评估足够，且显著降低内存占用）
# 说明：save_all_iterations_data=True 是 miceforest 6.0.5 做 impute_new_data 的硬要求；
# 因此需要通过减少 datasets/iterations 来控制内存。
FOLDWISE_NUM_DATASETS = 1
FOLDWISE_NUM_ITERATIONS = 3

# 可选：每次建模只抽取部分行训练 LightGBM（0=全量）。数值越小越省内存/更快，但可能略降插补质量。
FOLDWISE_DATA_SUBSET = 0

# 输出格式："auto"（大表用 parquet，小表用 xlsx） | "excel" | "parquet"
OUTPUT_FORMAT = "auto"
# auto 模式下，单表单元格数量超过阈值时使用 parquet
AUTO_PARQUET_CELL_THRESHOLD = 2_000_000

# 输入输出路径
FILE_PATH = f'merge/cleaned_data.xlsx'
OUTPUT_PATH = 'mice/imputed_combined.xlsx'

# 输入文件路径（兼容旧代码中使用的变量名）
INPUT_FILE = FILE_PATH

# 确保 OUTPUT_DIR 已定义：优先使用 OUTPUT_PATH 的目录，若为空则使用 INPUT 文件目录下的 'mice' 子目录
_out_dir = os.path.dirname(OUTPUT_PATH)
if _out_dir:
    OUTPUT_DIR = _out_dir
else:
    base_dir = os.path.dirname(FILE_PATH) or '.'
    OUTPUT_DIR = os.path.join(base_dir, 'mice')

COMBINED_OUTPUT = os.path.join(OUTPUT_DIR, 'combined_imputed.xlsx')
STATISTICAL_COMPARISON_OUTPUT = os.path.join(OUTPUT_DIR, 'statistical_comparison.xlsx')
ONEHOT_OUTPUT = os.path.join(OUTPUT_DIR, 'combined_imputed_onehot.xlsx')

COMBINED_OUTPUT_PARQUET = os.path.join(OUTPUT_DIR, 'combined_imputed.parquet')
ONEHOT_OUTPUT_PARQUET = os.path.join(OUTPUT_DIR, 'combined_imputed_onehot.parquet')

# 不参与插补的列（如ID、姓名等）
NON_IMPUTE_COLS = ['Number', 'Name', 'VisitNumber', 'ReportTime']

EXCLUDE_FROM_IMPUTATION = list(dict.fromkeys([*NON_IMPUTE_COLS]))

# 强制指定为分类和连续变量的列
FORCE_CONTINUOUS: list[str] = [
       "C1Q","Creatinine","Albumin","ALP","ALT",
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

        "Age", "BMI", "Tumor size", "Tumor volume", "Ki67","TNLE","PLN"
]

FORCE_CATEGORICAL: list[str] = [
        "Sex", "T stage", "N stage", "Differentiation grade", "Vascular invasion", "Perineural invasion",
        "Carcinoma nodule", "CDX2", "MLH1", "MSH2", "PMS2", "MSH6", "HER2", "Family history",
        "Colonic obstruction", "Hypertension", "Diabetes", "Coronary artery disease", "Hyperlipidemia",
        "Chemotherapy", "BRAF mutant", "KRAS mutant", "NRAS mutant", "HER2 mutant",
        "NTRK1 mutant", "NTRK2 mutant", "NTRK3 mutant", "RET mutant", "MSI-H",
        "mGPS"
]


# 正态性检验的显著性水平
NORMALITY_ALPHA = 0.05

# 绘图采样行数（避免大表热力图/直方图爆内存）
PLOT_SAMPLE_ROWS = 2000

# 缺失值概览图：默认最多绘制多少个变量（None 表示不限制）
MISSING_OVERVIEW_MAX_COLS: int | None = None


# ===== 辅助函数 =====

def create_output_dir(directory: str) -> None:
    """创建输出目录"""
    Path(directory).mkdir(parents=True, exist_ok=True)
    print(f"✓ 输出目录已创建: {directory}")


def _can_write_parquet() -> bool:
    """是否具备 parquet 写入引擎（pyarrow/fastparquet）。"""
    import importlib.util

    return (
        importlib.util.find_spec("pyarrow") is not None
        or importlib.util.find_spec("fastparquet") is not None
    )


def _choose_table_format(df: pd.DataFrame) -> str:
    """返回 'excel' 或 'parquet'。"""
    fmt = (OUTPUT_FORMAT or "auto").strip().lower()
    if fmt in {"excel", "xlsx"}:
        return "excel"
    if fmt in {"parquet", "pq"}:
        return "parquet"

    # auto
    n_cells = int(df.shape[0]) * int(df.shape[1])
    if n_cells >= AUTO_PARQUET_CELL_THRESHOLD:
        return "parquet"
    return "excel"


def save_table(df: pd.DataFrame, xlsx_path: str, parquet_path: str) -> str:
    """按配置保存表格，返回实际写出的路径。"""
    chosen = _choose_table_format(df)
    if chosen == "parquet":
        if _can_write_parquet():
            df.to_parquet(parquet_path, index=False)
            return parquet_path
        print("  ! 选择了 parquet 输出，但未检测到 pyarrow/fastparquet，已回退为 Excel")

    df.to_excel(xlsx_path, index=False)
    return xlsx_path


def load_data(file_path: str) -> pd.DataFrame:
    """加载数据文件"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件不存在: {file_path}")
    
    df = pd.read_excel(file_path)
    # 兼容源数据里用 '/' 表示缺失值的情况
    df = df.replace('/', pd.NA)
    print(f"✓ 数据已加载: {df.shape[0]} 行, {df.shape[1]} 列")
    return df


def analyze_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """分析缺失数据情况"""
    missing_stats = pd.DataFrame({
        '列名': df.columns,
        '缺失数量': df.isnull().sum().values,
        '缺失比例': (df.isnull().sum() / len(df)).values,
        '数据类型': df.dtypes.values
    })
    missing_stats.insert(1, '列名_展示', [_display_name(c, mode="plot") or str(c) for c in df.columns])
    missing_stats = missing_stats[missing_stats['缺失数量'] > 0].sort_values('缺失比例', ascending=False)
    
    print(f"\n缺失数据摘要:")
    print(f"  - 总共 {len(df.columns)} 列")
    print(f"  - 有缺失的列: {len(missing_stats)} 列")
    if len(missing_stats) > 0:
        print(f"  - 最高缺失率: {missing_stats['缺失比例'].max():.2%}")
    
    return missing_stats


def plot_missing_pattern(df: pd.DataFrame, output_path: str) -> None:
    """绘制缺失数据模式图"""
    missing_cols = df.columns[df.isnull().any()].tolist()
    if not missing_cols:
        print("  ! 没有缺失数据，跳过绘图")
        return

    # 大数据集下热力图会非常耗内存，采样行以避免 OOM
    df_plot = df
    if len(df_plot) > PLOT_SAMPLE_ROWS:
        df_plot = df_plot.sample(n=PLOT_SAMPLE_ROWS, random_state=42)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(df_plot[missing_cols].isnull(), cbar=True, cmap='viridis', yticklabels=False)
    plt.title('Missing Data Pattern')
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 缺失模式图已保存: {output_path}")


def plot_missing_overview(
    df: pd.DataFrame,
    output_path: str,
    cols: list[str] | None = None,
    max_cols: int | None = MISSING_OVERVIEW_MAX_COLS,
    include_missing_pct_in_label: bool = True,
    dpi: int = 300,
) -> None:
    """绘制变量缺失值概览图。

    - x 轴：变量
    - y 轴：患者（行）数量
    - 黑色线条：该患者该变量缺失

    为了可读性，当变量数量过多时默认只绘制缺失率最高的前 max_cols 个。
    """
    if cols is None:
        cols = list(df.columns)

    if not cols:
        print("  ! 未提供可绘制的列，跳过缺失概览图")
        return

    cols = [c for c in cols if c in df.columns]
    if not cols:
        print("  ! 需绘制的列在数据中均不存在，跳过缺失概览图")
        return

    missing_rate = df[cols].isnull().mean().astype(float)
    if (missing_rate > 0).sum() == 0:
        print("  ! 没有缺失数据，跳过缺失概览图")
        return

    if max_cols is not None and len(cols) > max_cols:
        # 变量太多时，优先展示缺失更严重的变量
        top_cols = missing_rate.sort_values(ascending=False).head(max_cols).index.tolist()
        print(f"  ! 变量过多({len(cols)}列)，缺失概览图仅绘制缺失率最高的前 {max_cols} 列")
        cols = top_cols
        missing_rate = missing_rate.loc[cols]

    mask = df[cols].isnull().to_numpy()
    n_rows, n_cols = mask.shape

    # 为每个缺失单元格创建一个短竖线段，视觉上接近示例图的“黑色线条”效果
    segments: list[list[tuple[float, float]]] = []
    for j in range(n_cols):
        ys = np.flatnonzero(mask[:, j])
        if ys.size == 0:
            continue
        x = float(j)
        # y 轴对应患者（行），每个缺失点画一个短竖线
        segments.extend([[(x, float(y) - 0.45), (x, float(y) + 0.45)] for y in ys])

    # 画布大小随列数/行数伸缩；不再对宽度做硬性上限（用户要求强制全画）
    fig_w = max(16.0, 4.0 + n_cols * 0.28)
    fig_h = 8.0 if n_rows <= 3000 else 10.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_facecolor('#f2f2f2')

    if segments:
        # 线宽略加粗，保证 300dpi 下仍清晰
        lc = LineCollection(segments, colors='black', linewidths=0.8)
        ax.add_collection(lc)

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)  # 让 0 在顶部，符合常见 missingness 矩阵视觉

    ax.set_xticks(range(n_cols))
    if include_missing_pct_in_label:
        labels = [f"{_display_name(c, mode='plot') or c} ({missing_rate.loc[c] * 100:.0f}%)" for c in cols]
    else:
        labels = [_display_name(c, mode='plot') or str(c) for c in cols]
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)

    ax.set_xlabel('variables')
    ax.set_ylabel('observations')
    ax.set_title('Missingness overview (black lines = missing)')

    # 去掉整幅图的“框框”（坐标轴外框 & 图例外框）
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis='both', which='both', length=0)

    # 图例：仅标注缺失线条
    legend_handle = Line2D([0], [0], color='black', linewidth=1.5, label='Missing')
    ax.legend(handles=[legend_handle], loc='upper right', frameon=False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=int(dpi), bbox_inches='tight')
    plt.close()
    print(f"✓ 缺失值概览图已保存: {output_path}")


def _kernel_num_datasets(kernel: mf.ImputationKernel) -> int:
    """兼容不同 miceforest 版本，获取数据集数量。"""
    for attr in ("dataset_count", "num_datasets", "n_datasets"):
        value = getattr(kernel, attr, None)
        if value is None:
            continue
        try:
            raw = value() if callable(value) else value
            if isinstance(raw, (int, np.integer)):
                return int(raw)
            if isinstance(raw, str) and raw.strip().isdigit():
                return int(raw.strip())
        except Exception:
            continue
    raise AttributeError("无法从 ImputationKernel 获取数据集数量（dataset_count/num_datasets/n_datasets 均不可用）")


def _kernel_complete_data(kernel: mf.ImputationKernel, dataset: int) -> pd.DataFrame:
    """兼容不同 miceforest 版本的 complete_data 返回值。"""
    df = kernel.complete_data(dataset=dataset)
    if df is None:
        raise RuntimeError(f"kernel.complete_data(dataset={dataset}) 返回 None")
    return df


def remove_high_missing_columns(df: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    """删除缺失率过高的列"""
    missing_rate = df.isnull().sum() / len(df)
    cols_to_drop = missing_rate[missing_rate > threshold].index.tolist()
    
    if cols_to_drop:
        print(f"\n警告: 以下列缺失率超过 {threshold:.0%}，将被删除:")
        for col in cols_to_drop:
            print(f"  - {col}: {missing_rate[col]:.2%}")
        df = df.drop(columns=cols_to_drop)
    else:
        print(f"✓ 没有列的缺失率超过 {threshold:.0%}")
    
    return df


def identify_variable_types(df: pd.DataFrame, 
                           non_impute_cols: list,
                           force_categorical: list,
                           force_continuous: list) -> tuple[list, list]:
    """识别分类变量和连续变量"""
    impute_cols = [col for col in df.columns if col not in non_impute_cols]
    
    categorical_vars = []
    continuous_vars = []
    
    for col in impute_cols:
        if col in force_categorical:
            categorical_vars.append(col)
        elif col in force_continuous:
            continuous_vars.append(col)
        else:
            # 自动识别
            if df[col].dtype == 'object' or df[col].nunique() < 10:
                categorical_vars.append(col)
            else:
                continuous_vars.append(col)
    
    print(f"\n变量类型识别:")
    print(f"  - 分类变量 ({len(categorical_vars)}): {categorical_vars[:5]}{'...' if len(categorical_vars) > 5 else ''}")
    print(f"  - 连续变量 ({len(continuous_vars)}): {continuous_vars[:5]}{'...' if len(continuous_vars) > 5 else ''}")
    
    return categorical_vars, continuous_vars


def prepare_data_for_imputation(df: pd.DataFrame, 
                                categorical_vars: list,
                                non_impute_cols: list) -> tuple[pd.DataFrame, dict[str, list]]:
    """准备数据用于插补。

    医学论文/机器学习建议：
    - 分类变量保持 pandas 的 category dtype，让 miceforest/LightGBM 以“分类特征”方式建模
      （避免把类别当作有序数值回归 + 四舍五入）。
    - 连续变量尽量转为数值（float），把非法字符统一为 NaN。

    返回：
    - df_impute：用于插补的 DataFrame（包含原始所有列；被排除的列不会进入 kernel）
    - categorical_levels：每个分类变量的全局类别水平（用于跨数据集/合并时对齐）
    """
    df_impute = df.copy()
    categorical_levels: dict[str, list] = {}

    # 先统一把分类列转为 category 并记录 levels
    for col in categorical_vars:
        if col not in df_impute.columns:
            continue
        # 关键：先做类别值规范化，避免 0/0.0/空白 等导致 categories 类型不一致
        df_impute[col] = normalize_categorical_series(df_impute[col])
        df_impute[col] = df_impute[col].astype("category")
        categorical_levels[col] = list(df_impute[col].cat.categories)

    # 再把“应为数值”的列尽量转成数值（尤其是 dtype=object 但实际是数字的列）
    impute_cols = [col for col in df_impute.columns if col not in non_impute_cols]
    for col in impute_cols:
        if col in categorical_vars:
            continue
        if pd.api.types.is_object_dtype(df_impute[col]):
            df_impute[col] = pd.to_numeric(df_impute[col], errors="coerce")

    print(f"✓ 已设置分类变量为 category: {len(categorical_levels)} 列")
    return df_impute, categorical_levels


def perform_mice_imputation(df: pd.DataFrame,
                            num_datasets: int,
                            num_iterations: int,
                            categorical_vars: list,
                            non_impute_cols: list) -> mf.ImputationKernel:
    """执行MICE插补"""
    impute_cols = [col for col in df.columns if col not in non_impute_cols]
    
    print(f"\n开始MICE插补:")
    print(f"  - 插补数据集数量: {num_datasets}")
    print(f"  - 迭代次数: {num_iterations}")
    print(f"  - 插补列数: {len(impute_cols)}")
    
    # miceforest 6.x 支持 pandas category dtype；但不接受任意 object
    df_to_impute = df[impute_cols].copy()
    for col in df_to_impute.columns:
        if pd.api.types.is_object_dtype(df_to_impute[col]):
            if col in categorical_vars:
                df_to_impute[col] = df_to_impute[col].astype("category")
            else:
                df_to_impute[col] = pd.to_numeric(df_to_impute[col], errors="coerce")

    # 创建MICE kernel（使用当前 miceforest 参数名）
    # save_all_iterations_data=True 会保留每次迭代的全量数据，极易导致内存不足
    kernel = mf.ImputationKernel(
        df_to_impute,
        num_datasets=num_datasets,
        save_all_iterations_data=False,
        random_state=42
    )

    # 执行插补
    kernel.mice(iterations=num_iterations, verbose=True)
    
    print("✓ MICE插补完成")
    return kernel


def _try_build_cv_splitter(
    df: pd.DataFrame,
    *,
    n_splits: int,
    shuffle: bool,
    random_state: int,
    stratify_target: str | None,
) -> tuple[object, np.ndarray | None]:
    """构建 CV splitter。

    - 优先 StratifiedKFold（当目标列存在、可用且类别数合理）
    - 否则回退到 KFold
    """
    if stratify_target and stratify_target in df.columns:
        y = df[stratify_target]
        y_non_missing = y.dropna()
        # 分层要求：不能有缺失；且至少 2 类；每类样本数应 >= n_splits
        if len(y_non_missing) == len(y):
            y_values = y.to_numpy()
            uniq, counts = np.unique(y_values, return_counts=True)
            if len(uniq) >= 2 and counts.min() >= n_splits:
                splitter = StratifiedKFold(
                    n_splits=int(n_splits),
                    shuffle=bool(shuffle),
                    random_state=int(random_state) if shuffle else None,
                )
                return splitter, y_values

    splitter = KFold(
        n_splits=int(n_splits),
        shuffle=bool(shuffle),
        random_state=int(random_state) if shuffle else None,
    )
    return splitter, None


def foldwise_mice_oof_imputation(
    df_prepared: pd.DataFrame,
    original_df: pd.DataFrame,
    *,
    categorical_vars: list[str],
    continuous_vars: list[str],
    categorical_levels: dict[str, list],
    excluded_cols: list[str],
    num_datasets: int,
    num_iterations: int,
    data_subset: int = 0,
    n_splits: int,
    shuffle: bool,
    random_state: int,
    stratify_target: str | None,
) -> pd.DataFrame:
    """按折(OOF)执行 MICE：每个样本的插补仅由“未包含该样本”的训练折学到的模型生成。

    产出：与 original_df 行数一致的 DataFrame（包含 excluded_cols 原样保留 + 插补列）。
    """
    impute_cols = [c for c in df_prepared.columns if c not in excluded_cols]
    df_to_impute = df_prepared[impute_cols].copy()

    # miceforest 不接受任意 object：再次兜底转型
    for col in df_to_impute.columns:
        if pd.api.types.is_object_dtype(df_to_impute[col]):
            if col in categorical_vars:
                df_to_impute[col] = df_to_impute[col].astype("category")
            else:
                df_to_impute[col] = pd.to_numeric(df_to_impute[col], errors="coerce")

    cat_cols = [c for c in categorical_vars if c in df_to_impute.columns]
    cont_cols = [c for c in continuous_vars if c in df_to_impute.columns]

    splitter, y = _try_build_cv_splitter(
        original_df,
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
        stratify_target=stratify_target,
    )

    n_rows = len(df_to_impute)
    cont_out: dict[str, np.ndarray] = {c: np.full(n_rows, np.nan, dtype=np.float32) for c in cont_cols}
    cat_out_codes: dict[str, np.ndarray] = {c: np.full(n_rows, -1, dtype=np.int32) for c in cat_cols}

    # 记录全局 categories（用于 codes→category 映射，保证跨折一致）
    cat_categories: dict[str, list] = {c: list(categorical_levels.get(c, [])) for c in cat_cols}

    all_indices = np.arange(n_rows)
    fold_iter = splitter.split(all_indices, y) if y is not None else splitter.split(all_indices)

    print(f"\n开始 fold-wise MICE (OOF) 插补:")
    print(f"  - folds: {n_splits}")
    print(f"  - datasets per fold: {num_datasets}")
    print(f"  - iterations: {num_iterations}")
    print(f"  - stratified: {'Yes' if y is not None else 'No'}")

    for fold_id, (train_idx, val_idx) in enumerate(fold_iter, start=1):
        print(f"\n[Fold {fold_id}/{n_splits}] train={len(train_idx)} val={len(val_idx)}")

        # miceforest 要求输入 DataFrame 的 index 为 RangeIndex（否则会触发断言）
        # 这里 reset_index(drop=True) 不影响我们用 val_idx 回写全局数组的逻辑
        train_data = df_to_impute.iloc[train_idx].copy().reset_index(drop=True)
        val_data = df_to_impute.iloc[val_idx].copy().reset_index(drop=True)

        # 训练折上拟合 MICE
        kernel = mf.ImputationKernel(
            train_data,
            num_datasets=int(num_datasets),
            # miceforest 6.0.5: impute_new_data 需要训练时保存迭代数据以重建插补过程
            save_all_iterations_data=True,
            data_subset=int(data_subset),
            random_state=int(random_state) + int(fold_id),
        )
        kernel.mice(iterations=int(num_iterations), verbose=True)

        # 用训练折学到的模型插补该折验证集（无泄漏）
        datasets = list(range(int(num_datasets)))
        imputed_val = kernel.impute_new_data(
            val_data,
            datasets=datasets,
            iterations=int(num_iterations),
            save_all_iterations_data=True,
            random_state=int(random_state) + 10_000 + int(fold_id),
            verbose=False,
        )

        # 在验证集内做“多数据集→单表”的聚合（连续取均值，分类取众数）
        n_val = len(val_idx)
        cont_acc_fold: dict[str, np.ndarray] = {c: np.zeros(n_val, dtype=np.float32) for c in cont_cols}
        cat_values_fold: dict[str, list[np.ndarray]] = {c: [] for c in cat_cols}

        for d in datasets:
            val_d = imputed_val.complete_data(dataset=int(d))
            val_d = decode_categorical_variables(val_d, categorical_vars, categorical_levels)

            for c in cont_cols:
                cont_acc_fold[c] += val_d[c].to_numpy(dtype=np.float32, copy=False)

            for c in cat_cols:
                if isinstance(val_d[c].dtype, pd.CategoricalDtype):
                    if not cat_categories.get(c):
                        cat_categories[c] = list(val_d[c].cat.categories)
                    codes = val_d[c].cat.codes.to_numpy(dtype=np.int32, copy=False)
                else:
                    tmp = val_d[c].astype("category")
                    if not cat_categories.get(c):
                        cat_categories[c] = list(tmp.cat.categories)
                    codes = tmp.cat.codes.to_numpy(dtype=np.int32, copy=False)
                cat_values_fold[c].append(codes)

            del val_d

        # 写回全局 OOF 结果（只填 val_idx）
        for c in cont_cols:
            cont_out[c][val_idx] = cont_acc_fold[c] / float(len(datasets))

        for c in cat_cols:
            mode_codes = _rowwise_mode_small_k(cat_values_fold[c]).astype(np.int32, copy=False)
            cat_out_codes[c][val_idx] = mode_codes

        del kernel, imputed_val

    # 组装最终 OOF DataFrame
    combined_df = pd.DataFrame(index=original_df.index)
    for c in cont_cols:
        combined_df[c] = cont_out[c]

    for c in cat_cols:
        codes = cat_out_codes[c]
        levels = cat_categories.get(c) or categorical_levels.get(c) or []
        cats = np.asarray(levels, dtype=object)
        out = np.full(len(codes), np.nan, dtype=object)
        valid = (codes >= 0) & (codes < len(cats))
        if valid.any():
            out[valid] = cats[codes[valid]]
        combined_df[c] = pd.Categorical(out, categories=list(levels))

        # 若仍有 NA（极少数情况），用全表观测众数兜底
        if combined_df[c].isna().any():
            # 注意：必须用“规范化后的众数”，且填充值必须在 levels(categories) 内
            fallback_mode = normalize_categorical_series(original_df[c]).dropna().mode()
            if len(fallback_mode):
                fill_val = fallback_mode.iloc[0]
                if (not levels) or (fill_val in set(map(str, levels))):
                    combined_df[c] = combined_df[c].astype(object).fillna(fill_val).astype(
                        pd.CategoricalDtype(categories=list(levels))
                    )

    combined_df = decode_categorical_variables(combined_df, categorical_vars, categorical_levels)

    non_impute_df = original_df[[c for c in excluded_cols if c in original_df.columns]].copy()
    combined_df = pd.concat([non_impute_df, combined_df], axis=1)
    return combined_df


def decode_categorical_variables(
    df: pd.DataFrame,
    categorical_vars: list,
    categorical_levels: dict[str, list] | None = None,
) -> pd.DataFrame:
    """确保分类变量 dtype/levels 对齐。

    当前版本不再做“编码→四舍五入→反解码”。
    若提供 categorical_levels，则会把对应列强制为该 categories 的 category dtype。
    """
    out = df.copy()
    if categorical_levels is None:
        # 尽量保持 category dtype
        for col in categorical_vars:
            if col in out.columns and not pd.api.types.is_categorical_dtype(out[col]):
                out[col] = out[col].astype("category")
        return out

    for col in categorical_vars:
        if col not in out.columns:
            continue
        levels = categorical_levels.get(col)
        if levels is None:
            out[col] = out[col].astype("category")
            continue
        out[col] = pd.Categorical(out[col].astype(object), categories=levels)
    return out


def _rowwise_mode_small_k(values: list[np.ndarray]) -> np.ndarray:
    """对每一行求众数，K 很小（比如 2~10）时的省内存实现。

    values: 形如 [arr0, arr1, ...]，每个 arr 形状为 (n,)
    返回: (n,) 的众数数组
    """
    if not values:
        raise ValueError("values 不能为空")
    if len(values) == 1:
        return values[0]

    stacked = np.stack(values, axis=1)  # (n, K)
    stacked.sort(axis=1)
    n_rows, k = stacked.shape

    mode_val = stacked[:, 0].copy()
    max_count = np.ones(n_rows, dtype=np.uint8)
    current_val = stacked[:, 0].copy()
    current_count = np.ones(n_rows, dtype=np.uint8)

    for j in range(1, k):
        same = stacked[:, j] == current_val
        current_count = np.where(same, current_count + 1, 1).astype(np.uint8, copy=False)
        current_val = np.where(same, current_val, stacked[:, j])
        better = current_count > max_count
        max_count = np.where(better, current_count, max_count)
        mode_val = np.where(better, current_val, mode_val)

    return mode_val


def save_and_combine_imputed_datasets(
    kernel: mf.ImputationKernel,
    original_df: pd.DataFrame,
    categorical_vars: list,
    continuous_vars: list,
    categorical_levels: dict[str, list],
    non_impute_cols: list,
    output_dir: str,
    combined_output_xlsx: str,
    combined_output_parquet: str,
) -> pd.DataFrame:
    """边保存各个插补数据集，边合并为一个数据集，避免把所有 DataFrame 留在内存里。"""
    num_datasets = _kernel_num_datasets(kernel)
    print(f"\n保存并合并插补数据集（共 {num_datasets} 个）...")

    # kernel 输入的数据列（仅插补列）
    impute_cols: list[str] | None = None
    maybe_df = getattr(kernel, "imputation_data", None)
    if isinstance(maybe_df, pd.DataFrame):
        impute_cols = list(maybe_df.columns)
    if impute_cols is None:
        # 兼容不同版本的属性名
        for attr in ("data", "working_data"):
            obj = getattr(kernel, attr, None)
            if isinstance(obj, pd.DataFrame):
                impute_cols = list(obj.columns)
                break
    if impute_cols is None:
        raise AttributeError("无法从 kernel 获取插补列名（imputation_data/data/working_data 均不可用）")

    cat_cols = [c for c in categorical_vars if c in impute_cols]
    cont_cols = [c for c in continuous_vars if c in impute_cols]

    n_rows = len(original_df)
    cont_acc: dict[str, np.ndarray] = {c: np.zeros(n_rows, dtype=np.float32) for c in cont_cols}
    cat_values: dict[str, list[np.ndarray]] = {c: [] for c in cat_cols}
    cat_categories: dict[str, list] = {c: list(categorical_levels.get(c, [])) for c in cat_cols}

    non_impute_df = original_df[[c for c in non_impute_cols if c in original_df.columns]].copy()

    for i in range(num_datasets):
        imputed = _kernel_complete_data(kernel, dataset=i)

        # 聚合（先用编码/数值形式聚合，避免 object/string 占用内存）
        for c in cont_cols:
            cont_acc[c] += imputed[c].to_numpy(dtype=np.float32, copy=False)
        for c in cat_cols:
            # 使用 category codes 做众数聚合；确保 levels 在第一次出现时记录
            if isinstance(imputed[c].dtype, pd.CategoricalDtype):
                if not cat_categories.get(c):
                    cat_categories[c] = list(imputed[c].cat.categories)
                codes = imputed[c].cat.codes.to_numpy(dtype=np.int32, copy=False)
            else:
                # 兜底：转 category 再取 codes
                tmp = imputed[c].astype("category")
                if not cat_categories.get(c):
                    cat_categories[c] = list(tmp.cat.categories)
                codes = tmp.cat.codes.to_numpy(dtype=np.int32, copy=False)
            cat_values[c].append(codes)

        # 保存每个插补数据集（写完即释放）
        imputed_to_save = decode_categorical_variables(imputed, categorical_vars, categorical_levels)
        out_df = pd.concat([non_impute_df, imputed_to_save], axis=1)
        xlsx_path = os.path.join(output_dir, f'imputed_dataset_{i+1}.xlsx')
        parquet_path = os.path.join(output_dir, f'imputed_dataset_{i+1}.parquet')
        actual_path = save_table(out_df, xlsx_path, parquet_path)
        print(f"✓ 已保存插补数据集 {i+1}: {actual_path}")

        del out_df, imputed_to_save, imputed

    # 构建合并结果
    combined_imputed: dict[str, np.ndarray] = {}
    for c in cont_cols:
        combined_imputed[c] = cont_acc[c] / float(num_datasets)
    for c in cat_cols:
        combined_imputed[c] = _rowwise_mode_small_k(cat_values[c])

    # 连续列直接构建；分类列先用 codes 的众数，再映射回 category
    combined_df = pd.DataFrame(index=original_df.index)
    for c in cont_cols:
        combined_df[c] = combined_imputed[c]
    for c in cat_cols:
        codes = combined_imputed[c].astype(np.int32, copy=False)
        levels = cat_categories.get(c) or categorical_levels.get(c) or []
        cats = np.asarray(levels, dtype=object)
        out = np.full(len(codes), np.nan, dtype=object)
        valid = (codes >= 0) & (codes < len(cats))
        if valid.any():
            out[valid] = cats[codes[valid]]
        combined_df[c] = pd.Categorical(out, categories=list(levels))

    combined_df = decode_categorical_variables(combined_df, categorical_vars, categorical_levels)
    combined_df = pd.concat([non_impute_df, combined_df], axis=1)

    actual_path = save_table(combined_df, combined_output_xlsx, combined_output_parquet)
    print(f"✓ 已保存合并数据集: {actual_path}")
    return combined_df


def save_run_metadata(
    *,
    output_dir: str,
    input_file: str,
    df_shape: tuple[int, int],
    missing_threshold: float,
    num_datasets: int,
    num_iterations: int,
    excluded_cols: list[str],
    categorical_vars: list[str],
    continuous_vars: list[str],
    categorical_levels: dict[str, list],
    imputation_mode: str | None = None,
    cv_n_splits: int | None = None,
    cv_shuffle: bool | None = None,
    cv_random_state: int | None = None,
    cv_stratify_target: str | None = None,
) -> None:
    meta = {
        "input_file": input_file,
        "n_rows": int(df_shape[0]),
        "n_cols": int(df_shape[1]),
        "miceforest_version": getattr(mf, "__version__", "unknown"),
        "pandas_version": getattr(pd, "__version__", "unknown"),
        "numpy_version": getattr(np, "__version__", "unknown"),
        "sklearn_version": getattr(sklearn, "__version__", "unknown"),
        "missing_threshold": float(missing_threshold),
        "num_imputation_datasets": int(num_datasets),
        "num_iterations": int(num_iterations),
        "excluded_from_imputation": excluded_cols,
        "categorical_vars": categorical_vars,
        "continuous_vars": continuous_vars,
        "categorical_levels": {k: [str(x) for x in v] for k, v in categorical_levels.items()},
        "random_state": 42,
        "imputation_mode": imputation_mode,
        "cv_n_splits": cv_n_splits,
        "cv_shuffle": cv_shuffle,
        "cv_random_state": cv_random_state,
        "cv_stratify_target": cv_stratify_target,
    }
    out = os.path.join(output_dir, "imputation_run_meta.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"✓ 运行元数据已保存: {out}")


def qc_missing_only_summary(
    original_df: pd.DataFrame,
    imputed_df: pd.DataFrame,
    categorical_vars: list[str],
    continuous_vars: list[str],
    output_file: str,
) -> pd.DataFrame:
    """论文更常用的插补质控：只比较“原本缺失的位置”的插补值。

    连续变量：观测值 vs 插补(缺失位) 的均值/中位数/范围、KS 检验（可选）
    分类变量：观测分布 vs 插补(缺失位) 分布、卡方检验
    """
    rows: list[dict[str, object]] = []

    # 连续
    for var in continuous_vars:
        if var not in original_df.columns or var not in imputed_df.columns:
            continue
        miss_mask = original_df[var].isna()
        if miss_mask.sum() == 0:
            continue
        obs = pd.to_numeric(original_df.loc[~miss_mask, var], errors="coerce").dropna()
        imp = pd.to_numeric(imputed_df.loc[miss_mask, var], errors="coerce").dropna()
        if len(obs) < 10 or len(imp) < 10:
            continue

        row: dict[str, object] = {
            "变量": var,
            "变量类型": "连续变量",
            "missing_n": int(miss_mask.sum()),
            "obs_n": int(len(obs)),
            "imputed_n": int(len(imp)),
            "obs_mean": float(obs.mean()),
            "imp_mean": float(imp.mean()),
            "obs_median": float(obs.median()),
            "imp_median": float(imp.median()),
            "obs_min": float(obs.min()),
            "obs_max": float(obs.max()),
            "imp_min": float(imp.min()),
            "imp_max": float(imp.max()),
        }
        try:
            # KS：分布是否明显不同（注意：它不是“合理性”的充分必要条件，仅作报警）
            ks = stats.ks_2samp(obs, imp)
            row["KS_stat"] = float(ks.statistic)
            row["KS_p"] = float(ks.pvalue)
        except Exception:
            row["KS_stat"] = np.nan
            row["KS_p"] = np.nan
        rows.append(row)

    # 分类
    for var in categorical_vars:
        if var not in original_df.columns or var not in imputed_df.columns:
            continue
        miss_mask = original_df[var].isna()
        if miss_mask.sum() == 0:
            continue
        obs = normalize_categorical_series(original_df.loc[~miss_mask, var]).dropna()
        imp = normalize_categorical_series(imputed_df.loc[miss_mask, var]).dropna()
        if len(obs) < 10 or len(imp) < 10:
            continue

        obs_counts = normalized_categorical_value_counts(obs)
        imp_counts = normalized_categorical_value_counts(imp)
        cats = sorted(set(obs_counts.index) | set(imp_counts.index))
        contingency = np.array([[obs_counts.get(c, 0) for c in cats], [imp_counts.get(c, 0) for c in cats]])
        row2: dict[str, object] = {
            "变量": var,
            "变量类型": "分类变量",
            "missing_n": int(miss_mask.sum()),
            "obs_n": int(len(obs)),
            "imputed_n": int(len(imp)),
            "obs_top": str(obs_counts.index[0]) if len(obs_counts) else "",
            "imp_top": str(imp_counts.index[0]) if len(imp_counts) else "",
        }
        try:
            chi2_stat, p, dof, expected = stats.chi2_contingency(contingency)
            row2["chi2"] = float(chi2_stat)
            row2["chi2_p"] = float(p)
        except Exception:
            row2["chi2"] = np.nan
            row2["chi2_p"] = np.nan
        rows.append(row2)

    out_df = pd.DataFrame(rows)
    out_df.to_excel(output_file, index=False)
    print(f"✓ 缺失位插补质控已保存: {output_file}")
    return out_df


def plot_imputation_distributions(original_df: pd.DataFrame,
                                  imputed_df: pd.DataFrame,
                                  categorical_vars: list,
                                  continuous_vars: list,
                                  output_path: str,
                                  max_vars: int = 6) -> None:
    """绘制插补前后的分布对比图"""
    # 选择有缺失值的变量进行可视化
    vars_to_plot = []
    for var in continuous_vars + categorical_vars:
        if var in original_df.columns and original_df[var].isnull().any():
            vars_to_plot.append(var)
        if len(vars_to_plot) >= max_vars:
            break
    
    if not vars_to_plot:
        print("  ! 没有需要可视化的变量")
        return
    
    n_vars = len(vars_to_plot)
    fig, axes = plt.subplots(n_vars, 2, figsize=(12, 4*n_vars))
    
    if n_vars == 1:
        axes = axes.reshape(1, -1)
    
    for idx, var in enumerate(vars_to_plot):
        var_disp = _display_name(var, mode="plot") or str(var)
        # 原始数据（仅非缺失值）
        original_values = original_df[var].dropna()
        # 插补后的数据
        imputed_values = imputed_df[var]

        # 避免绘图使用全量数据导致内存压力
        if len(original_values) > PLOT_SAMPLE_ROWS:
            original_values = original_values.sample(n=PLOT_SAMPLE_ROWS, random_state=42)
        if len(imputed_values) > PLOT_SAMPLE_ROWS:
            imputed_values = imputed_values.sample(n=PLOT_SAMPLE_ROWS, random_state=42)
        
        if var in continuous_vars:
            # 连续变量：直方图
            axes[idx, 0].hist(original_values, bins=30, alpha=0.7, color='blue', edgecolor='black')
            axes[idx, 0].set_title(f'{var_disp} - Original (non-missing)')
            axes[idx, 0].set_ylabel('Frequency')
            
            axes[idx, 1].hist(imputed_values, bins=30, alpha=0.7, color='green', edgecolor='black')
            axes[idx, 1].set_title(f'{var_disp} - After Imputation')
        else:
            # 分类变量：条形图
            original_counts = original_values.value_counts()
            imputed_counts = imputed_values.value_counts()
            
            axes[idx, 0].bar(range(len(original_counts)), original_counts.values, alpha=0.7, color='blue')
            axes[idx, 0].set_xticks(range(len(original_counts)))
            axes[idx, 0].set_xticklabels(original_counts.index, rotation=45, ha='right')
            axes[idx, 0].set_title(f'{var_disp} - Original (non-missing)')
            axes[idx, 0].set_ylabel('Count')
            
            axes[idx, 1].bar(range(len(imputed_counts)), imputed_counts.values, alpha=0.7, color='green')
            axes[idx, 1].set_xticks(range(len(imputed_counts)))
            axes[idx, 1].set_xticklabels(imputed_counts.index, rotation=45, ha='right')
            axes[idx, 1].set_title(f'{var_disp} - After Imputation')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 分布对比图已保存: {output_path}")


def assess_imputation_quality(original_df: pd.DataFrame,
                              imputed_df: pd.DataFrame,
                              output_dir: str,
                              categorical_vars: list[str] | None = None,
                              continuous_vars: list[str] | None = None) -> None:
    """评估插补质量"""
    print("\n=== 插补质量评估 ===")
    
    report_lines = []
    report_lines.append("插补质量评估报告")
    report_lines.append("=" * 50)
    
    # 1. 缺失值填充情况
    original_missing = original_df.isnull().sum().sum()
    imputed_missing = imputed_df.isnull().sum().sum()
    
    report_lines.append(f"\n1. 缺失值填充情况:")
    report_lines.append(f"   - 插补前总缺失值: {original_missing}")
    report_lines.append(f"   - 插补后总缺失值: {imputed_missing}")
    report_lines.append(f"   - 填充率: {(1 - imputed_missing/max(original_missing, 1)) * 100:.2f}%")
    
    # 2. 各列的缺失值变化
    report_lines.append(f"\n2. 各列缺失值变化:")
    for col in original_df.columns:
        orig_miss = original_df[col].isnull().sum()
        imp_miss = imputed_df[col].isnull().sum()
        if orig_miss > 0:
            report_lines.append(f"   - {_display_name(col, mode='plot') or col}: {orig_miss} → {imp_miss}")
    
    # 3. 连续变量统计量对比（仅对连续变量做 mean/std；分类变量不做均值）
    report_lines.append(f"\n3. 连续变量统计量对比:")
    if continuous_vars is None:
        # 兜底：按 dtype 推断
        continuous_vars = [c for c in original_df.columns if pd.api.types.is_numeric_dtype(original_df[c])]

    for col in continuous_vars:
        if col not in original_df.columns or col not in imputed_df.columns:
            continue
        if not original_df[col].isnull().any():
            continue

        orig_vals = pd.to_numeric(original_df[col], errors="coerce")
        imp_vals = pd.to_numeric(imputed_df[col], errors="coerce")
        orig_mean = float(orig_vals.mean())
        orig_std = float(orig_vals.std())
        imp_mean = float(imp_vals.mean())
        imp_std = float(imp_vals.std())

        report_lines.append(f"\n   {col}:")
        report_lines.append(f"     插补前: 均值={orig_mean:.2f}, 标准差={orig_std:.2f}")
        report_lines.append(f"     插补后: 均值={imp_mean:.2f}, 标准差={imp_std:.2f}")
    
    # 保存报告
    report_path = os.path.join(output_dir, 'imputation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print('\n'.join(report_lines))
    print(f"\n✓ 评估报告已保存: {report_path}")


def check_normality(data: pd.Series) -> tuple[bool, str]:
    """
    检验数据是否服从正态分布
    返回: (是否正态, 检验方法)
    """
    # 移除缺失值
    # 对连续变量，确保可转为数值；否则 Shapiro 会因字符串报错
    data_clean = pd.to_numeric(data.dropna(), errors='coerce').dropna()
    
    if len(data_clean) < 3:
        return False, "样本量不足"
    
    # 使用Shapiro-Wilk检验（适用于样本量 < 5000）
    if len(data_clean) < 5000:
        _, p_value = shapiro(data_clean)
        method = "Shapiro-Wilk"
    else:
        # 大样本使用Kolmogorov-Smirnov检验
        _, p_value = normaltest(data_clean)
        method = "D'Agostino-Pearson"
    
    is_normal = p_value > NORMALITY_ALPHA
    return is_normal, method


def perform_statistical_comparison(original_df: pd.DataFrame,
                                   imputed_df: pd.DataFrame,
                                   categorical_vars: list,
                                   continuous_vars: list,
                                   output_file: str) -> pd.DataFrame:
    """
    对插补前后的数据进行统计学比较
    - 正态连续变量: t检验, 均值±标准差
    - 非正态连续变量: Mann-Whitney U检验, 中位数(Q1-Q3)
    - 分类变量: 卡方检验, 频数(百分比)
    """
    print("\n=== 统计学比较分析 ===")
    
    results = []
    
    # 分析连续变量
    for var in continuous_vars:
        if var not in original_df.columns:
            continue
        
        # 获取插补前后的数据（插补前仅使用非缺失值）
        original_data = pd.to_numeric(original_df[var], errors='coerce').dropna()
        imputed_data = pd.to_numeric(imputed_df[var], errors='coerce').dropna()
        
        if len(original_data) < 2 or len(imputed_data) < 2:
            continue
        
        # 检验正态性
        is_normal_orig, method_orig = check_normality(original_data)
        is_normal_imp, method_imp = check_normality(imputed_data)
        is_normal = is_normal_orig and is_normal_imp
        
        row = {'变量': var, '变量类型': '连续变量'}
        
        if is_normal:
            # 正态分布：使用t检验
            mean_orig = original_data.mean()
            std_orig = original_data.std()
            mean_imp = imputed_data.mean()
            std_imp = imputed_data.std()
            
            # 独立样本t检验
            t_stat, p_value = stats.ttest_ind(original_data, imputed_data)
            
            row['插补前'] = f"{mean_orig:.2f}±{std_orig:.2f}"
            row['插补后'] = f"{mean_imp:.2f}±{std_imp:.2f}"
            row['检验方法'] = "Independent t-test"
            row['检验统计量'] = f"t={t_stat:.3f}"
            row['P值'] = f"{p_value:.4f}"
            row['分布类型'] = "正态分布"
            
        else:
            # 非正态分布：使用Mann-Whitney U检验
            median_orig = original_data.median()
            q1_orig = original_data.quantile(0.25)
            q3_orig = original_data.quantile(0.75)
            
            median_imp = imputed_data.median()
            q1_imp = imputed_data.quantile(0.25)
            q3_imp = imputed_data.quantile(0.75)
            
            # Mann-Whitney U检验
            u_stat, p_value = stats.mannwhitneyu(original_data, imputed_data, alternative='two-sided')
            
            row['插补前'] = f"{median_orig:.2f} ({q1_orig:.2f}-{q3_orig:.2f})"
            row['插补后'] = f"{median_imp:.2f} ({q1_imp:.2f}-{q3_imp:.2f})"
            row['检验方法'] = "Mann-Whitney U test"
            row['检验统计量'] = f"U={u_stat:.1f}"
            row['P值'] = f"{p_value:.4f}"
            row['分布类型'] = "偏态分布"
        
        results.append(row)
    
    # 分析分类变量
    for var in categorical_vars:
        if var not in original_df.columns:
            continue
        
        # 获取插补前后的数据
        # 统一转为字符串，避免出现 int/str 混合导致排序/集合操作报错
        original_data = normalize_categorical_series(original_df[var]).dropna()
        imputed_data = normalize_categorical_series(imputed_df[var]).dropna()
        
        if len(original_data) < 1 or len(imputed_data) < 1:
            continue
        
        # 统计频数和百分比
        orig_counts = normalized_categorical_value_counts(original_data)
        imp_counts = normalized_categorical_value_counts(imputed_data)
        
        # 获取所有类别
        all_categories = sorted(set(orig_counts.index) | set(imp_counts.index))
        
        # 构建列联表
        contingency_table = []
        for cat in all_categories:
            orig_n = orig_counts.get(cat, 0)
            imp_n = imp_counts.get(cat, 0)
            contingency_table.append([orig_n, imp_n])
        
        contingency_table = np.array(contingency_table).T
        
        # 卡方检验
        try:
            chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
            chi2_result = f"χ²={chi2_stat:.3f}"
            p_val_str = f"{p_value:.4f}"
        except:
            chi2_result = "N/A"
            p_val_str = "N/A"
        
        # 格式化输出
        orig_str_parts = []
        imp_str_parts = []
        for cat in all_categories:
            orig_n = orig_counts.get(cat, 0)
            orig_pct = (orig_n / len(original_data) * 100)
            imp_n = imp_counts.get(cat, 0)
            imp_pct = (imp_n / len(imputed_data) * 100)
            
            orig_str_parts.append(f"{cat}: {orig_n}({orig_pct:.1f}%)")
            imp_str_parts.append(f"{cat}: {imp_n}({imp_pct:.1f}%)")
        
        row = {
            '变量': var,
            '变量类型': '分类变量',
            '插补前': "; ".join(orig_str_parts),
            '插补后': "; ".join(imp_str_parts),
            '检验方法': "Chi-square test",
            '检验统计量': chi2_result,
            'P值': p_val_str,
            '分布类型': 'N/A'
        }
        
        results.append(row)
    
    # 创建结果DataFrame
    results_df = pd.DataFrame(results)
    
    # 保存到Excel
    results_df.to_excel(output_file, index=False)
    print(f"✓ 统计比较结果已保存: {output_file}")
    
    # 打印摘要
    print("\n统计比较摘要:")
    print(f"  - 分析变量总数: {len(results)}")
    print(f"  - 连续变量: {len([r for r in results if r['变量类型'] == '连续变量'])}")
    print(f"  - 分类变量: {len([r for r in results if r['变量类型'] == '分类变量'])}")
    
    # 显示前几行
    print("\n前5个变量的比较结果:")
    print(results_df.head().to_string(index=False))
    
    return results_df


def perform_onehot_encoding(df: pd.DataFrame,
                            categorical_vars: list,
                            output_file: str) -> pd.DataFrame:
    """
    对分类变量进行one-hot编码
    """
    print("\n=== One-Hot编码 ===")

    cols = [c for c in categorical_vars if c in df.columns]
    if not cols:
        df.to_excel(output_file, index=False)
        print(f"  ! 未找到可编码的分类变量，直接保存原表: {output_file}")
        return df

    # 一次性 get_dummies，避免逐列 concat 产生大量中间对象
    df_encoded = pd.get_dummies(df, columns=cols, prefix=cols, drop_first=False, dtype=np.uint8)
    for var in cols:
        created = [c for c in df_encoded.columns if c.startswith(f"{var}_")]
        print(f"  ✓ {var}: 编码为 {len(created)} 列")
    
    # 保存（one-hot 表通常更大，auto 模式更可能选择 parquet）
    parquet_path = os.path.splitext(output_file)[0] + ".parquet"
    actual_path = save_table(df_encoded, output_file, parquet_path)
    print(f"✓ One-hot编码后的数据已保存: {actual_path}")
    print(f"  - 原始列数: {len(df.columns)}")
    print(f"  - 编码后列数: {len(df_encoded.columns)}")
    
    return df_encoded


# ===== 主函数 =====

def main():
    """主函数：执行完整的MICE插补流程"""
    print("=" * 60)
    print("MICE 多重插补程序")
    print("=" * 60)
    
    # 1. 创建输出目录
    create_output_dir(OUTPUT_DIR)
    
    # 2. 加载数据
    df = load_data(INPUT_FILE)
    
    # 3. 分析缺失数据
    missing_stats = analyze_missing_data(df)
    
    # 4. 绘制缺失模式图
    plot_missing_pattern(df, os.path.join(OUTPUT_DIR, 'missing_pattern.png'))

    # 4b. 绘制变量缺失值概览图（示例图风格：黑色线条表示缺失）
    # 强制绘制研究指定变量（按 FORCE_CONTINUOUS + FORCE_CATEGORICAL 顺序；自动跳过数据中不存在的列）
    forced_cols = list(dict.fromkeys([*FORCE_CONTINUOUS, *FORCE_CATEGORICAL]))
    plot_missing_overview(
        df,
        os.path.join(OUTPUT_DIR, 'missing_overview.png'),
        cols=forced_cols,
        max_cols=None,
        dpi=300,
    )

    # 额外导出矢量版 PDF（放大不糊）
    plot_missing_overview(
        df,
        os.path.join(OUTPUT_DIR, 'missing_overview.pdf'),
        cols=forced_cols,
        max_cols=None,
        dpi=300,
    )
    
    # 5. 删除缺失率过高的列
    df = remove_high_missing_columns(df, threshold=MISSING_THRESHOLD)
    
    # 6. 识别变量类型
    categorical_vars, continuous_vars = identify_variable_types(
        df, EXCLUDE_FROM_IMPUTATION, FORCE_CATEGORICAL, FORCE_CONTINUOUS
    )
    
    # 7. 准备数据（编码分类变量）
    df_prepared, categorical_levels = prepare_data_for_imputation(
        df, categorical_vars, EXCLUDE_FROM_IMPUTATION
    )
    
    # 8~10. 插补并生成 combined_imputed（性能评估默认走 fold-wise OOF，无泄漏）
    mode = (IMPUTATION_MODE or "foldwise_oof").strip().lower()
    meta_num_datasets = NUM_IMPUTATION_DATASETS
    meta_num_iterations = NUM_ITERATIONS
    if mode == "foldwise_oof":
        meta_num_datasets = FOLDWISE_NUM_DATASETS
        meta_num_iterations = FOLDWISE_NUM_ITERATIONS
        combined_df = foldwise_mice_oof_imputation(
            df_prepared,
            df,
            categorical_vars=[c for c in categorical_vars if c in df.columns],
            continuous_vars=[c for c in continuous_vars if c in df.columns],
            categorical_levels=categorical_levels,
            excluded_cols=EXCLUDE_FROM_IMPUTATION,
            num_datasets=FOLDWISE_NUM_DATASETS,
            num_iterations=FOLDWISE_NUM_ITERATIONS,
            data_subset=FOLDWISE_DATA_SUBSET,
            n_splits=CV_N_SPLITS,
            shuffle=CV_SHUFFLE,
            random_state=CV_RANDOM_STATE,
            stratify_target=CV_STRATIFY_TARGET,
        )
        actual_path = save_table(combined_df, COMBINED_OUTPUT, COMBINED_OUTPUT_PARQUET)
        print(f"✓ 已保存 OOF(按折) 合并插补数据集: {actual_path}")
    elif mode == "global_fullfit":
        kernel = perform_mice_imputation(
            df_prepared,
            NUM_IMPUTATION_DATASETS,
            NUM_ITERATIONS,
            categorical_vars,
            EXCLUDE_FROM_IMPUTATION,
        )
        combined_df = save_and_combine_imputed_datasets(
            kernel,
            df,
            categorical_vars,
            continuous_vars,
            categorical_levels,
            EXCLUDE_FROM_IMPUTATION,
            OUTPUT_DIR,
            COMBINED_OUTPUT,
            COMBINED_OUTPUT_PARQUET,
        )
    else:
        raise ValueError(f"未知 IMPUTATION_MODE: {IMPUTATION_MODE}")

    # 10b. 保存可复现实验的元数据（论文/审计）
    save_run_metadata(
        output_dir=OUTPUT_DIR,
        input_file=INPUT_FILE,
        df_shape=df.shape,
        missing_threshold=MISSING_THRESHOLD,
        num_datasets=meta_num_datasets,
        num_iterations=meta_num_iterations,
        excluded_cols=EXCLUDE_FROM_IMPUTATION,
        categorical_vars=[c for c in categorical_vars if c in df.columns],
        continuous_vars=[c for c in continuous_vars if c in df.columns],
        categorical_levels=categorical_levels,
        imputation_mode=mode,
        cv_n_splits=CV_N_SPLITS if mode == "foldwise_oof" else None,
        cv_shuffle=CV_SHUFFLE if mode == "foldwise_oof" else None,
        cv_random_state=CV_RANDOM_STATE if mode == "foldwise_oof" else None,
        cv_stratify_target=CV_STRATIFY_TARGET if mode == "foldwise_oof" else None,
    )
    
    # 11. 绘制插补前后分布对比图
    plot_imputation_distributions(
        df, combined_df, categorical_vars, continuous_vars,
        os.path.join(OUTPUT_DIR, 'imputation_distributions.png')
    )
    
    
    # 12. One-hot编码（新增）
    df_onehot = perform_onehot_encoding(
        combined_df, categorical_vars, ONEHOT_OUTPUT
    )
    
    print("\n" + "=" * 60)
    print("✓ 所有步骤完成!")
    print("=" * 60)
    print(f"\n输出文件:")
    print(f"  1. 合并插补数据: {COMBINED_OUTPUT} 或 {COMBINED_OUTPUT_PARQUET}（取决于 OUTPUT_FORMAT）")
    print(f"  2. 统计比较结果: {STATISTICAL_COMPARISON_OUTPUT}")
    print(f"  3. One-hot编码数据: {ONEHOT_OUTPUT} 或 {ONEHOT_OUTPUT_PARQUET}（取决于 OUTPUT_FORMAT）")
    print(f"  4. 其他文件保存在: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()