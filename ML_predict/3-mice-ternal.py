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
    """Normalize categorical values uniformly to avoid duplicate categories like 0 / 0.0 / ' 0 '.

    Note: miceforest is highly sensitive to pandas category levels; if categories are [0.0, 1.0]
    but fallback values are '0'/'1', it causes issues where data remains missing after imputation.
    """
    if s is None:
        return s

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
    """Normalize levels first, then calculate value counts for categorical series."""
    normalized = normalize_categorical_series(s).dropna()
    if normalized.empty:
        return pd.Series(dtype="int64")
    return normalized.value_counts()

NUM_IMPUTATION_DATASETS = 5  # Number of imputation datasets
NUM_ITERATIONS = 5  # Iterations per dataset
MISSING_THRESHOLD = 0.7  # Missing rate threshold (columns exceeding this will be dropped)

# Imputation Strategy:
# - "foldwise_oof": Train MICE foldwise and only impute the validation fold to assemble out-of-fold combined_imputed.xlsx (Leakage-free evaluation)
# - "global_fullfit": Single MICE run on the full data (Only recommended for final full deployment training, not performance evaluation)
IMPUTATION_MODE = "foldwise_oof"

# CV settings for fold-wise MICE
CV_N_SPLITS = 5
CV_SHUFFLE = True
CV_RANDOM_STATE = 42

# Prioritize StratifiedKFold if target column is available to maintain the event rate
CV_STRATIFY_TARGET = "Metastasis"

# Fold-wise MICE resource-friendly default parameters
FOLDWISE_NUM_DATASETS = 1
FOLDWISE_NUM_ITERATIONS = 3

# Optional: subset rows trained by LightGBM to save memory (0 = full data).
FOLDWISE_DATA_SUBSET = 0

# Output format: "auto" (parquet for large tables, xlsx for small tables) | "excel" | "parquet"
OUTPUT_FORMAT = "auto"
AUTO_PARQUET_CELL_THRESHOLD = 2_000_000

# Input/Output paths
FILE_PATH = f'merge/cleaned_data.xlsx'
OUTPUT_PATH = 'mice/imputed_combined.xlsx'
INPUT_FILE = FILE_PATH

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

# Columns excluded from imputation (e.g. IDs, names)
NON_IMPUTE_COLS = ['Number', 'Name', 'VisitNumber', 'ReportTime']
EXCLUDE_FROM_IMPUTATION = list(dict.fromkeys([*NON_IMPUTE_COLS]))

# Explicitly assigned categorical and continuous column lists
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

# Normality significance level
NORMALITY_ALPHA = 0.05

# Row sampling to avoid OOM during large heatmap plotting
PLOT_SAMPLE_ROWS = 2000
MISSING_OVERVIEW_MAX_COLS: int | None = None


# ===== Helper Functions =====

def create_output_dir(directory: str) -> None:
    """Create output directory"""
    Path(directory).mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory created: {directory}")


def _can_write_parquet() -> bool:
    """Check if parquet engine (pyarrow/fastparquet) is installed."""
    import importlib.util
    return (
        importlib.util.find_spec("pyarrow") is not None
        or importlib.util.find_spec("fastparquet") is not None
    )


def _choose_table_format(df: pd.DataFrame) -> str:
    """Return 'excel' or 'parquet' depending on settings and data size."""
    fmt = (OUTPUT_FORMAT or "auto").strip().lower()
    if fmt in {"excel", "xlsx"}:
        return "excel"
    if fmt in {"parquet", "pq"}:
        return "parquet"

    n_cells = int(df.shape[0]) * int(df.shape[1])
    if n_cells >= AUTO_PARQUET_CELL_THRESHOLD:
        return "parquet"
    return "excel"


def save_table(df: pd.DataFrame, xlsx_path: str, parquet_path: str) -> str:
    """Save table based on configuration format, returns actual path saved."""
    chosen = _choose_table_format(df)
    if chosen == "parquet":
        if _can_write_parquet():
            df.to_parquet(parquet_path, index=False)
            return parquet_path
        print("  ! Parquet format selected, but pyarrow/fastparquet not detected. Falling back to Excel.")

    df.to_excel(xlsx_path, index=False)
    return xlsx_path


def load_data(file_path: str) -> pd.DataFrame:
    """Load data file"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file does not exist: {file_path}")
    
    df = pd.read_excel(file_path)
    df = df.replace('/', pd.NA)
    print(f"✓ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def analyze_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze missing data characteristics."""
    missing_stats = pd.DataFrame({
        'Column Name': df.columns,
        'Missing Count': df.isnull().sum().values,
        'Missing Ratio': (df.isnull().sum() / len(df)).values,
        'Data Type': df.dtypes.values
    })
    missing_stats.insert(1, 'Display Name', [_display_name(c, mode="plot") or str(c) for c in df.columns])
    missing_stats = missing_stats[missing_stats['Missing Count'] > 0].sort_values('Missing Ratio', ascending=False)
    
    print(f"\nMissing Data Summary:")
    print(f"  - Total {len(df.columns)} columns")
    print(f"  - Columns with missing values: {len(missing_stats)} columns")
    if len(missing_stats) > 0:
        print(f"  - Maximum missing rate: {missing_stats['Missing Ratio'].max():.2%}")
    
    return missing_stats


def plot_missing_pattern(df: pd.DataFrame, output_path: str) -> None:
    """Draw missing data pattern heatmaps."""
    missing_cols = df.columns[df.isnull().any()].tolist()
    if not missing_cols:
        print("  ! No missing data detected, skipping plot.")
        return

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
    print(f"✓ Missing pattern plot saved: {output_path}")


def plot_missing_overview(
    df: pd.DataFrame,
    output_path: str,
    cols: list[str] | None = None,
    max_cols: int | None = MISSING_OVERVIEW_MAX_COLS,
    include_missing_pct_in_label: bool = True,
    dpi: int = 300,
) -> None:
    """Draw column missingness overview map where black lines denote missing observations."""
    if cols is None:
        cols = list(df.columns)

    if not cols:
        print("  ! No columns provided, skipping missingness overview plot.")
        return

    cols = [c for c in cols if c in df.columns]
    if not cols:
        print("  ! Specified columns do not exist in data, skipping missingness overview plot.")
        return

    missing_rate = df[cols].isnull().mean().astype(float)
    if (missing_rate > 0).sum() == 0:
        print("  ! No missing data detected, skipping missingness overview plot.")
        return

    if max_cols is not None and len(cols) > max_cols:
        top_cols = missing_rate.sort_values(ascending=False).head(max_cols).index.tolist()
        print(f"  ! Too many columns ({len(cols)}), only plotting top {max_cols} columns with highest missing rates.")
        cols = top_cols
        missing_rate = missing_rate.loc[cols]

    mask = df[cols].isnull().to_numpy()
    n_rows, n_cols = mask.shape

    segments: list[list[tuple[float, float]]] = []
    for j in range(n_cols):
        ys = np.flatnonzero(mask[:, j])
        if ys.size == 0:
            continue
        x = float(j)
        segments.extend([[(x, float(y) - 0.45), (x, float(y) + 0.45)] for y in ys])

    fig_w = max(16.0, 4.0 + n_cols * 0.28)
    fig_h = 8.0 if n_rows <= 3000 else 10.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_facecolor('#f2f2f2')

    if segments:
        lc = LineCollection(segments, colors='black', linewidths=0.8)
        ax.add_collection(lc)

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)

    ax.set_xticks(range(n_cols))
    if include_missing_pct_in_label:
        labels = [f"{_display_name(c, mode='plot') or c} ({missing_rate.loc[c] * 100:.0f}%)" for c in cols]
    else:
        labels = [_display_name(c, mode='plot') or str(c) for c in cols]
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)

    ax.set_xlabel('variables')
    ax.set_ylabel('observations')
    ax.set_title('Missingness overview (black lines = missing)')

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis='both', which='both', length=0)

    legend_handle = Line2D([0], [0], color='black', linewidth=1.5, label='Missing')
    ax.legend(handles=[legend_handle], loc='upper right', frameon=False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=int(dpi), bbox_inches='tight')
    plt.close()
    print(f"✓ Missingness overview plot saved: {output_path}")


def _kernel_num_datasets(kernel: mf.ImputationKernel) -> int:
    """Version compatibility wrapper to extract dataset counts from kernel."""
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
    raise AttributeError("Cannot retrieve dataset count from ImputationKernel.")


def _kernel_complete_data(kernel: mf.ImputationKernel, dataset: int) -> pd.DataFrame:
    """Version compatibility wrapper for complete_data."""
    df = kernel.complete_data(dataset=dataset)
    if df is None:
        raise RuntimeError(f"kernel.complete_data(dataset={dataset}) returned None")
    return df


def remove_high_missing_columns(df: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    """Drop columns that exceed the specified missingness rate threshold."""
    missing_rate = df.isnull().sum() / len(df)
    cols_to_drop = missing_rate[missing_rate > threshold].index.tolist()
    
    if cols_to_drop:
        print(f"\nWarning: The following columns exceed missing rate threshold {threshold:.0%} and will be dropped:")
        for col in cols_to_drop:
            print(f"  - {col}: {missing_rate[col]:.2%}")
        df = df.drop(columns=cols_to_drop)
    else:
        print(f"✓ No columns exceed missingness threshold {threshold:.0%}")
    
    return df


def identify_variable_types(df: pd.DataFrame, 
                           non_impute_cols: list,
                           force_categorical: list,
                           force_continuous: list) -> tuple[list, list]:
    """Classify columns into continuous and categorical variables."""
    impute_cols = [col for col in df.columns if col not in non_impute_cols]
    
    categorical_vars = []
    continuous_vars = []
    
    for col in impute_cols:
        if col in force_categorical:
            categorical_vars.append(col)
        elif col in force_continuous:
            continuous_vars.append(col)
        else:
            if df[col].dtype == 'object' or df[col].nunique() < 10:
                categorical_vars.append(col)
            else:
                continuous_vars.append(col)
    
    print(f"\nVariable Type Identification:")
    print(f"  - Categorical variables ({len(categorical_vars)}): {categorical_vars[:5]}{'...' if len(categorical_vars) > 5 else ''}")
    print(f"  - Continuous variables ({len(continuous_vars)}): {continuous_vars[:5]}{'...' if len(continuous_vars) > 5 else ''}")
    
    return categorical_vars, continuous_vars


def prepare_data_for_imputation(df: pd.DataFrame, 
                                categorical_vars: list,
                                non_impute_cols: list) -> tuple[pd.DataFrame, dict[str, list]]:
    """Format columns properly for miceforest handling."""
    df_impute = df.copy()
    categorical_levels: dict[str, list] = {}

    for col in categorical_vars:
        if col not in df_impute.columns:
            continue
        df_impute[col] = normalize_categorical_series(df_impute[col])
        df_impute[col] = df_impute[col].astype("category")
        categorical_levels[col] = list(df_impute[col].cat.categories)

    impute_cols = [col for col in df_impute.columns if col not in non_impute_cols]
    for col in impute_cols:
        if col in categorical_vars:
            continue
        if pd.api.types.is_object_dtype(df_impute[col]):
            df_impute[col] = pd.to_numeric(df_impute[col], errors="coerce")

    print(f"✓ Set categorical columns to pandas category dtype: {len(categorical_levels)} columns")
    return df_impute, categorical_levels


def perform_mice_imputation(df: pd.DataFrame,
                            num_datasets: int,
                            num_iterations: int,
                            categorical_vars: list,
                            non_impute_cols: list) -> mf.ImputationKernel:
    """Execute MICE Imputation process."""
    impute_cols = [col for col in df.columns if col not in non_impute_cols]
    
    print(f"\nStarting MICE Imputation:")
    print(f"  - Imputation Datasets: {num_datasets}")
    print(f"  - Iterations: {num_iterations}")
    print(f"  - Imputed Columns Count: {len(impute_cols)}")
    
    df_to_impute = df[impute_cols].copy()
    for col in df_to_impute.columns:
        if pd.api.types.is_object_dtype(df_to_impute[col]):
            if col in categorical_vars:
                df_to_impute[col] = df_to_impute[col].astype("category")
            else:
                df_to_impute[col] = pd.to_numeric(df_to_impute[col], errors="coerce")

    kernel = mf.ImputationKernel(
        df_to_impute,
        num_datasets=num_datasets,
        save_all_iterations_data=False,
        random_state=42
    )

    kernel.mice(iterations=num_iterations, verbose=True)
    print("✓ MICE imputation process completed")
    return kernel


def _try_build_cv_splitter(
    df: pd.DataFrame,
    *,
    n_splits: int,
    shuffle: bool,
    random_state: int,
    stratify_target: str | None,
) -> tuple[object, np.ndarray | None]:
    """Construct CV splitter prioritizing StratifiedKFold if criteria met."""
    if stratify_target and stratify_target in df.columns:
        y = df[stratify_target]
        y_non_missing = y.dropna()
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
    """Execute Fold-wise OOF MICE imputation to avoid target leakage."""
    impute_cols = [c for c in df_prepared.columns if c not in excluded_cols]
    df_to_impute = df_prepared[impute_cols].copy()

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
    cat_categories: dict[str, list] = {c: list(categorical_levels.get(c, [])) for c in cat_cols}

    all_indices = np.arange(n_rows)
    fold_iter = splitter.split(all_indices, y) if y is not None else splitter.split(all_indices)

    print(f"\nStarting fold-wise MICE (OOF) imputation:")
    print(f"  - folds: {n_splits}")
    print(f"  - datasets per fold: {num_datasets}")
    print(f"  - iterations: {num_iterations}")
    print(f"  - stratified: {'Yes' if y is not None else 'No'}")

    for fold_id, (train_idx, val_idx) in enumerate(fold_iter, start=1):
        print(f"\n[Fold {fold_id}/{n_splits}] train={len(train_idx)} val={len(val_idx)}")

        train_data = df_to_impute.iloc[train_idx].copy().reset_index(drop=True)
        val_data = df_to_impute.iloc[val_idx].copy().reset_index(drop=True)

        kernel = mf.ImputationKernel(
            train_data,
            num_datasets=int(num_datasets),
            save_all_iterations_data=True,
            data_subset=int(data_subset),
            random_state=int(random_state) + int(fold_id),
        )
        kernel.mice(iterations=int(num_iterations), verbose=True)

        datasets = list(range(int(num_datasets)))
        imputed_val = kernel.impute_new_data(
            val_data,
            datasets=datasets,
            iterations=int(num_iterations),
            save_all_iterations_data=True,
            random_state=int(random_state) + 10_000 + int(fold_id),
            verbose=False,
        )

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

        for c in cont_cols:
            cont_out[c][val_idx] = cont_acc_fold[c] / float(len(datasets))

        for c in cat_cols:
            mode_codes = _rowwise_mode_small_k(cat_values_fold[c]).astype(np.int32, copy=False)
            cat_out_codes[c][val_idx] = mode_codes

        del kernel, imputed_val

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

        if combined_df[c].isna().any():
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
    """Align categories and formatting specifications for datasets."""
    out = df.copy()
    if categorical_levels is None:
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
    """Calculate row-wise mode optimized for low K counts to save memory space."""
    if not values:
        raise ValueError("values cannot be empty")
    if len(values) == 1:
        return values[0]

    stacked = np.stack(values, axis=1)
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
    """Iteratively aggregate and save individual data streams to optimize heap overhead."""
    num_datasets = _kernel_num_datasets(kernel)
    print(f"\nSaving and combining imputed datasets (Total {num_datasets})...")

    impute_cols: list[str] | None = None
    maybe_df = getattr(kernel, "imputation_data", None)
    if isinstance(maybe_df, pd.DataFrame):
        impute_cols = list(maybe_df.columns)
    if impute_cols is None:
        for attr in ("data", "working_data"):
            obj = getattr(kernel, attr, None)
            if isinstance(obj, pd.DataFrame):
                impute_cols = list(obj.columns)
                break
    if impute_cols is None:
        raise AttributeError("Unable to retrieve imputation columns from kernel attributes.")

    cat_cols = [c for c in categorical_vars if c in impute_cols]
    cont_cols = [c for c in continuous_vars if c in impute_cols]

    n_rows = len(original_df)
    cont_acc: dict[str, np.ndarray] = {c: np.zeros(n_rows, dtype=np.float32) for c in cont_cols}
    cat_values: dict[str, list[np.ndarray]] = {c: [] for c in cat_cols}
    cat_categories: dict[str, list] = {c: list(categorical_levels.get(c, [])) for c in cat_cols}

    non_impute_df = original_df[[c for c in non_impute_cols if c in original_df.columns]].copy()

    for i in range(num_datasets):
        imputed = _kernel_complete_data(kernel, dataset=i)

        for c in cont_cols:
            cont_acc[c] += imputed[c].to_numpy(dtype=np.float32, copy=False)
        for c in cat_cols:
            if isinstance(imputed[c].dtype, pd.CategoricalDtype):
                if not cat_categories.get(c):
                    cat_categories[c] = list(imputed[c].cat.categories)
                codes = imputed[c].cat.codes.to_numpy(dtype=np.int32, copy=False)
            else:
                tmp = imputed[c].astype("category")
                if not cat_categories.get(c):
                    cat_categories[c] = list(tmp.cat.categories)
                codes = tmp.cat.codes.to_numpy(dtype=np.int32, copy=False)
            cat_values[c].append(codes)

        imputed_to_save = decode_categorical_variables(imputed, categorical_vars, categorical_levels)
        out_df = pd.concat([non_impute_df, imputed_to_save], axis=1)
        xlsx_path = os.path.join(output_dir, f'imputed_dataset_{i+1}.xlsx')
        parquet_path = os.path.join(output_dir, f'imputed_dataset_{i+1}.parquet')
        actual_path = save_table(out_df, xlsx_path, parquet_path)
        print(f"✓ Saved imputed dataset {i+1}: {actual_path}")

        del out_df, imputed_to_save, imputed

    combined_imputed: dict[str, np.ndarray] = {}
    for c in cont_cols:
        combined_imputed[c] = cont_acc[c] / float(num_datasets)
    for c in cat_cols:
        combined_imputed[c] = _rowwise_mode_small_k(cat_values[c])

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
    print(f"✓ Saved combined dataset: {actual_path}")
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
    print(f"✓ Execution run metadata saved: {out}")


def qc_missing_only_summary(
    original_df: pd.DataFrame,
    imputed_df: pd.DataFrame,
    categorical_vars: list[str],
    continuous_vars: list[str],
    output_file: str,
) -> pd.DataFrame:
    """Generate Quality Control analysis summaries targeting imputed frames strictly matching former empty indexes."""
    rows: list[dict[str, object]] = []

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
            "Variable": var,
            "Variable Type": "Continuous",
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
            ks = stats.ks_2samp(obs, imp)
            row["KS_stat"] = float(ks.statistic)
            row["KS_p"] = float(ks.pvalue)
        except Exception:
            row["KS_stat"] = np.nan
            row["KS_p"] = np.nan
        rows.append(row)

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
            "Variable": var,
            "Variable Type": "Categorical",
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
    print(f"✓ Missing-only quality control analysis summary saved: {output_file}")
    return out_df


def plot_imputation_distributions(original_df: pd.DataFrame,
                                  imputed_df: pd.DataFrame,
                                  categorical_vars: list,
                                  continuous_vars: list,
                                  output_path: str,
                                  max_vars: int = 6) -> None:
    """Plot distributional shifts contrasting values post-imputation."""
    vars_to_plot = []
    for var in continuous_vars + categorical_vars:
        if var in original_df.columns and original_df[var].isnull().any():
            vars_to_plot.append(var)
        if len(vars_to_plot) >= max_vars:
            break
    
    if not vars_to_plot:
        print("  ! No columns qualified for plotting.")
        return
    
    n_vars = len(vars_to_plot)
    fig, axes = plt.subplots(n_vars, 2, figsize=(12, 4*n_vars))
    
    if n_vars == 1:
        axes = axes.reshape(1, -1)
    
    for idx, var in enumerate(vars_to_plot):
        var_disp = _display_name(var, mode="plot") or str(var)
        original_values = original_df[var].dropna()
        imputed_values = imputed_df[var]

        if len(original_values) > PLOT_SAMPLE_ROWS:
            original_values = original_values.sample(n=PLOT_SAMPLE_ROWS, random_state=42)
        if len(imputed_values) > PLOT_SAMPLE_ROWS:
            imputed_values = imputed_values.sample(n=PLOT_SAMPLE_ROWS, random_state=42)
        
        if var in continuous_vars:
            axes[idx, 0].hist(original_values, bins=30, alpha=0.7, color='blue', edgecolor='black')
            axes[idx, 0].set_title(f'{var_disp} - Original (non-missing)')
            axes[idx, 0].set_ylabel('Frequency')
            
            axes[idx, 1].hist(imputed_values, bins=30, alpha=0.7, color='green', edgecolor='black')
            axes[idx, 1].set_title(f'{var_disp} - After Imputation')
        else:
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
    print(f"✓ Distribution comparison plot saved: {output_path}")


def assess_imputation_quality(original_df: pd.DataFrame,
                              imputed_df: pd.DataFrame,
                              output_dir: str,
                              categorical_vars: list[str] | None = None,
                              continuous_vars: list[str] | None = None) -> None:
    """Assess overall imputation results quality metrics."""
    print("\n=== Imputation Quality Evaluation ===")
    
    report_lines = []
    report_lines.append("Imputation Quality Evaluation Report")
    report_lines.append("=" * 50)
    
    original_missing = original_df.isnull().sum().sum()
    imputed_missing = imputed_df.isnull().sum().sum()
    
    report_lines.append(f"\n1. Missing Values Filling Status:")
    report_lines.append(f"   - Total missing values before imputation: {original_missing}")
    report_lines.append(f"   - Total missing values after imputation: {imputed_missing}")
    report_lines.append(f"   - Completion Rate: {(1 - imputed_missing/max(original_missing, 1)) * 100:.2f}%")
    
    report_lines.append(f"\n2. Missing Value Changes by Column:")
    for col in original_df.columns:
        orig_miss = original_df[col].isnull().sum()
        imp_miss = imputed_df[col].isnull().sum()
        if orig_miss > 0:
            report_lines.append(f"   - {_display_name(col, mode='plot') or col}: {orig_miss} → {imp_miss}")
    
    report_lines.append(f"\n3. Statistical Comparison for Continuous Variables:")
    if continuous_vars is None:
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
        report_lines.append(f"     Before Imputation: Mean={orig_mean:.2f}, Std={orig_std:.2f}")
        report_lines.append(f"     After Imputation:  Mean={imp_mean:.2f}, Std={imp_std:.2f}")
    
    report_path = os.path.join(output_dir, 'imputation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print('\n'.join(report_lines))
    print(f"\n✓ Imputation evaluation report saved: {report_path}")


def check_normality(data: pd.Series) -> tuple[bool, str]:
    """Check whether a given sequence follows a normal distribution configuration."""
    data_clean = pd.to_numeric(data.dropna(), errors='coerce').dropna()
    
    if len(data_clean) < 3:
        return False, "Insufficient sample size"
    
    if len(data_clean) < 5000:
        _, p_value = shapiro(data_clean)
        method = "Shapiro-Wilk"
    else:
        _, p_value = normaltest(data_clean)
        method = "D'Agostino-Pearson"
    
    is_normal = p_value > NORMALITY_ALPHA
    return is_normal, method


def perform_statistical_comparison(original_df: pd.DataFrame,
                                   imputed_df: pd.DataFrame,
                                   categorical_vars: list,
                                   continuous_vars: list,
                                   output_file: str) -> pd.DataFrame:
    """Run baseline tests comparing distributions pre and post MICE pipeline runs."""
    print("\n=== Statistical Comparison Analysis ===")
    results = []
    
    for var in continuous_vars:
        if var not in original_df.columns:
            continue
        
        original_data = pd.to_numeric(original_df[var], errors='coerce').dropna()
        imputed_data = pd.to_numeric(imputed_df[var], errors='coerce').dropna()
        
        if len(original_data) < 2 or len(imputed_data) < 2:
            continue
        
        is_normal_orig, method_orig = check_normality(original_data)
        is_normal_imp, method_imp = check_normality(imputed_data)
        is_normal = is_normal_orig and is_normal_imp
        
        row = {'Variable': var, 'Variable Type': 'Continuous'}
        
        if is_normal:
            mean_orig = original_data.mean()
            std_orig = original_data.std()
            mean_imp = imputed_data.mean()
            std_imp = imputed_data.std()
            
            t_stat, p_value = stats.ttest_ind(original_data, imputed_data)
            
            row['Before Imputation'] = f"{mean_orig:.2f}±{std_orig:.2f}"
            row['After Imputation'] = f"{mean_imp:.2f}±{std_imp:.2f}"
            row['Test Method'] = "Independent t-test"
            row['Statistic'] = f"t={t_stat:.3f}"
            row['P-value'] = f"{p_value:.4f}"
            row['Distribution Type'] = "Normal"
            
        else:
            median_orig = original_data.median()
            q1_orig = original_data.quantile(0.25)
            q3_orig = original_data.quantile(0.75)
            
            median_imp = imputed_data.median()
            q1_imp = imputed_data.quantile(0.25)
            q3_imp = imputed_data.quantile(0.75)
            
            u_stat, p_value = stats.mannwhitneyu(original_data, imputed_data, alternative='two-sided')
            
            row['Before Imputation'] = f"{median_orig:.2f} ({q1_orig:.2f}-{q3_orig:.2f})"
            row['After Imputation'] = f"{median_imp:.2f} ({q1_imp:.2f}-{q3_imp:.2f})"
            row['Test Method'] = "Mann-Whitney U test"
            row['Statistic'] = f"U={u_stat:.1f}"
            row['P-value'] = f"{p_value:.4f}"
            row['Distribution Type'] = "Skewed"
        
        results.append(row)
    
    for var in categorical_vars:
        if var not in original_df.columns:
            continue
        
        original_data = normalize_categorical_series(original_df[var]).dropna()
        imputed_data = normalize_categorical_series(imputed_df[var]).dropna()
        
        if len(original_data) < 1 or len(imputed_data) < 1:
            continue
        
        orig_counts = normalized_categorical_value_counts(original_data)
        imp_counts = normalized_categorical_value_counts(imputed_data)
        
        all_categories = sorted(set(orig_counts.index) | set(imp_counts.index))
        
        contingency_table = []
        for cat in all_categories:
            orig_n = orig_counts.get(cat, 0)
            imp_n = imp_counts.get(cat, 0)
            contingency_table.append([orig_n, imp_n])
        
        contingency_table = np.array(contingency_table).T
        
        try:
            chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
            chi2_result = f"χ²={chi2_stat:.3f}"
            p_val_str = f"{p_value:.4f}"
        except:
            chi2_result = "N/A"
            p_val_str = "N/A"
        
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
            'Variable': var,
            'Variable Type': 'Categorical',
            'Before Imputation': "; ".join(orig_str_parts),
            'After Imputation': "; ".join(imp_str_parts),
            'Test Method': "Chi-square test",
            'Statistic': chi2_result,
            'P-value': p_val_str,
            'Distribution Type': 'N/A'
        }
        
        results.append(row)
    
    results_df = pd.DataFrame(results)
    results_df.to_excel(output_file, index=False)
    print(f"✓ Statistical comparison results saved: {output_file}")
    
    print("\nStatistical Comparison Summary:")
    print(f"  - Total analyzed variables: {len(results)}")
    print(f"  - Continuous variables: {len([r for r in results if r['Variable Type'] == 'Continuous'])}")
    print(f"  - Categorical variables: {len([r for r in results if r['Variable Type'] == 'Categorical'])}")
    
    print("\nTop 5 rows of comparison results:")
    print(results_df.head().to_string(index=False))
    
    return results_df


def perform_onehot_encoding(df: pd.DataFrame,
                            categorical_vars: list,
                            output_file: str) -> pd.DataFrame:
    """Execute One-Hot structural mapping transformation across target categories."""
    print("\n=== One-Hot Encoding ===")

    cols = [c for c in categorical_vars if c in df.columns]
    if not cols:
        df.to_excel(output_file, index=False)
        print(f"  ! No encoding matching elements discovered. Exiting directly: {output_file}")
        return df

    df_encoded = pd.get_dummies(df, columns=cols, prefix=cols, drop_first=False, dtype=np.uint8)
    for var in cols:
        created = [c for c in df_encoded.columns if c.startswith(f"{var}_")]
        print(f"  ✓ {var}: encoded into {len(created)} columns")
    
    parquet_path = os.path.splitext(output_file)[0] + ".parquet"
    actual_path = save_table(df_encoded, output_file, parquet_path)
    print(f"✓ One-hot encoded dataset saved: {actual_path}")
    print(f"  - Original columns count: {len(df.columns)}")
    print(f"  - Post-encoding columns count: {len(df_encoded.columns)}")
    
    return df_encoded


# ===== Main Execution Context =====

def main():
    print("=" * 60)
    print("MICE Multivariate Imputation Execution Pipeline")
    print("=" * 60)
    
    create_output_dir(OUTPUT_DIR)
    df = load_data(INPUT_FILE)
    missing_stats = analyze_missing_data(df)
    plot_missing_pattern(df, os.path.join(OUTPUT_DIR, 'missing_pattern.png'))

    forced_cols = list(dict.fromkeys([*FORCE_CONTINUOUS, *FORCE_CATEGORICAL]))
    plot_missing_overview(
        df,
        os.path.join(OUTPUT_DIR, 'missing_overview.png'),
        cols=forced_cols,
        max_cols=None,
        dpi=300,
    )
    plot_missing_overview(
        df,
        os.path.join(OUTPUT_DIR, 'missing_overview.pdf'),
        cols=forced_cols,
        max_cols=None,
        dpi=300,
    )
    
    df = remove_high_missing_columns(df, threshold=MISSING_THRESHOLD)
    
    categorical_vars, continuous_vars = identify_variable_types(
        df, EXCLUDE_FROM_IMPUTATION, FORCE_CATEGORICAL, FORCE_CONTINUOUS
    )
    
    df_prepared, categorical_levels = prepare_data_for_imputation(
        df, categorical_vars, EXCLUDE_FROM_IMPUTATION
    )
    
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
        print(f"✓ Saved OOF (foldwise) aggregated dataset: {actual_path}")
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
        raise ValueError(f"Unknown IMPUTATION_MODE parameter passed: {IMPUTATION_MODE}")

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
    
    plot_imputation_distributions(
        df, combined_df, categorical_vars, continuous_vars,
        os.path.join(OUTPUT_DIR, 'imputation_distributions.png')
    )
    
    df_onehot = perform_onehot_encoding(
        combined_df, categorical_vars, ONEHOT_OUTPUT
    )
    
    print("\n" + "=" * 60)
    print("✓ Execution completed successfully!")
    print("=" * 60)
    print(f"\nProduced Files Summary:")
    print(f"  1. Aggregated Imputation Table:  {COMBINED_OUTPUT} or {COMBINED_OUTPUT_PARQUET}")
    print(f"  2. Baseline Comparison Profile:  {STATISTICAL_COMPARISON_OUTPUT}")
    print(f"  3. One-hot Formatted Table Output: {ONEHOT_OUTPUT} or {ONEHOT_OUTPUT_PARQUET}")
    print(f"  4. Supplemental charts logged under directory matching target path: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()