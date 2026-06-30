# -*- coding: utf-8 -*-
"""
Simple Excel outlier processing and plotting script (Hardcoded paths)

Functions:
1. Read the preset Excel file (data.xlsx in the same directory, default to the first sheet).
2. Remove %, replace > and < prefixes and try to convert to numeric values. 
   > is converted to 1.1 times, < is converted to 0.9 times.
3. Define outliers for numeric columns using the IQR method (Q1-1.5*IQR, Q3+1.5*IQR), 
   and clip outliers to the boundary values.
4. Default output pagination "mosaic" boxplot (4x6): each variable is split into 
   Before/After subplots to enhance contrast; single column plots are optionally enabled.
5. Save the modified data as a new Excel file after all columns are processed (output/cleaned_data.xlsx).

Notes:
- Only numeric columns are processed (non-numeric columns are skipped).
- Neither the original nor the modified list contains null values (NaN); 
   if a column has no valid numeric values, no image will be generated for it.
- For simplicity, paths and parameters are hardcoded.
"""

import os
import sys
import re
import string
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import unicodedata
import json

from display_names import display_name as _display_name

# Resolve missing Chinese/full-width font issues in Matplotlib (common on Windows):
# Prioritize common Chinese fonts; if not available, Matplotlib falls back automatically.
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # Resolve negative sign displaying as a square box

# Make overall style close to example image (gray background + white grid)
plt.style.use('ggplot')

# --------------------------- Hardcoded Files and Directories ---------------------------
# Preset Excel filename: please place data.xlsx in the same directory as this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MERGE_DIR = os.path.join(SCRIPT_DIR, 'merge')
INPUT_EXCEL = os.path.join(MERGE_DIR, '228-show.xlsx')  # Hardcoded input

OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'merge_otclimit')      # Result output directory
OUTPUT_EXCEL = os.path.join(OUTPUT_DIR, '228 show-cleaned_data.xlsx')

# Keep only "mosaic" layout output (closer to example plot). Change to True to restore single column plots.
SAVE_SINGLE_PLOTS = False

# Optional: Hardcode column names to be processed (ensuring they are continuous numeric variables).
# If empty, all numeric columns will be processed by default.
TARGET_COLUMNS: List[str] = [
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
       "Monocyte count","CRP","Iron","Reticulocyte  %","NLR","PLR","LMR","SII","PNI",

        "Age", "BMI", "Tumor size", "Tumor volume", "Ki67","TNLE","PLN"
]

# Explicitly excluded columns (not processed even if they are numeric).
EXCLUDE_COLUMNS: List[str] = []

# Automatically skip "low cardinality numeric columns" (commonly numerical encoded categorical variables)
AUTO_SKIP_LOW_CARDINALITY_NUMERIC = True

# If the count of unique non-null values in a numeric column is <= this threshold, it is treated as categorical and skipped
LOW_CARDINALITY_MAX_UNIQUE = 8

# Also refer to the unique value ratio to avoid misclassifying continuous variables when the sample size is large
# unique_ratio = nunique / non_na_count
LOW_CARDINALITY_MAX_UNIQUE_RATIO = 0.05

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load column name mapping (Chinese -> English), use empty dict if it does not exist
MAPPING_PATH = os.path.join(MERGE_DIR, 'column_name_mapping.json')
try:
    with open(MAPPING_PATH, 'r', encoding='utf-8') as mf:
        COLUMN_NAME_MAPPING = json.load(mf)
except Exception:
    COLUMN_NAME_MAPPING = {}


def _to_numeric_series(df: pd.DataFrame, col_name: str) -> pd.Series:
    """Safely get a numeric series from df[col_name]. Missing/invalid -> NaN."""
    if col_name not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col_name], errors='coerce')


def _safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    """Safe division: denom==0 or NaN -> NaN."""
    denom2 = denom.replace(0, np.nan)
    out = numer / denom2
    return out.replace([np.inf, -np.inf], np.nan)


def add_derived_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Add 7 derived columns.

    For rows where required raw inputs are missing (or invalid), a missing-mask is recorded.
    Later, before exporting, those rows will be filled with '/'.
    """
    df = df.copy()

    neut = _to_numeric_series(df, 'Neutrophil count')
    lymph = _to_numeric_series(df, 'Lymphocyte count')
    mono = _to_numeric_series(df, 'Monocyte count')
    plate = _to_numeric_series(df, 'Platelet count')
    alb = _to_numeric_series(df, 'Albumin')
    crp = _to_numeric_series(df, 'CRP')

    missing_masks: dict[str, pd.Series] = {}

    # 1) NLR
    df['NLR'] = _safe_div(neut, lymph)
    missing_masks['NLR'] = neut.isna() | lymph.isna() | (lymph == 0)

    # 2) PLR
    df['PLR'] = _safe_div(plate, lymph)
    missing_masks['PLR'] = plate.isna() | lymph.isna() | (lymph == 0)

    # 3) LMR
    df['LMR'] = _safe_div(lymph, mono)
    missing_masks['LMR'] = lymph.isna() | mono.isna() | (mono == 0)

    # 4) SII
    df['SII'] = _safe_div(plate * neut, lymph)
    missing_masks['SII'] = plate.isna() | neut.isna() | lymph.isna() | (lymph == 0)

    # 5) PNI
    df['PNI'] = alb * 0.1 + lymph * 5
    missing_masks['PNI'] = alb.isna() | lymph.isna()

    # 6) mGPS
    mgps = pd.Series(np.nan, index=df.index)
    # CRP <= 10 -> 0 (only needs CRP)
    mgps.loc[crp.notna() & (crp <= 10)] = 0
    # CRP > 10 needs albumin to decide 1/2
    mgps.loc[crp.notna() & (crp > 10) & alb.notna() & (alb >= 35)] = 1
    mgps.loc[crp.notna() & (crp > 10) & alb.notna() & (alb < 35)] = 2
    df['mGPS'] = mgps
    missing_masks['mGPS'] = crp.isna() | ((crp > 10) & alb.isna())

    return df, missing_masks


def safe_filename(name: str) -> str:
    """Convert column name to a string suitable as a filename (remove illegal characters)."""
    text = unicodedata.normalize('NFKC', str(name)).strip()
    text = re.sub(r'[^0-9A-Za-z._-]+', '_', text)
    text = text.strip('._-')
    if not text:
        return 'col'
    return text[:120]


def _panel_title_for_col(col_name: str) -> str:
    """Return the display name (with units) for plot titles, using mapping if available."""
    norm_name = unicodedata.normalize('NFKC', str(col_name))
    mapped_name = COLUMN_NAME_MAPPING.get(norm_name, COLUMN_NAME_MAPPING.get(str(col_name), norm_name))
    return _display_name(mapped_name, mode="plot") or str(mapped_name)


def _page_label(i: int) -> str:
    """A, B, ..., Z, AA, AB ..."""
    letters = string.ascii_uppercase
    out = ""
    n = i
    while True:
        out = letters[n % 26] + out
        n = n // 26 - 1
        if n < 0:
            break
    return out


def plot_boxgrid_pages(
    cols: List[str],
    original_lists: Dict[str, List[float]],
    corrected_lists: Dict[str, List[float]],
    out_dir: str,
    ncols: int = 6,
    nrows: int = 4,
    dpi: int = 300,
) -> List[str]:
    """Combine Before/After boxplots of multiple variables into paginated large grid plots.

    Key point: Each variable has one panel, split into left/right (Before / After) 
    with separate y-axes to maximize contrast.
    """
    os.makedirs(out_dir, exist_ok=True)

    valid_cols = [c for c in cols if (len(original_lists.get(c, [])) > 0 or len(corrected_lists.get(c, [])) > 0)]
    if not valid_cols:
        return []

    per_page = int(ncols) * int(nrows)
    pages: List[str] = []

    boxprops = dict(linewidth=0.95, color="black")
    whiskerprops = dict(linewidth=0.95, color="black")
    capprops = dict(linewidth=0.95, color="black")
    medianprops = dict(linewidth=1.2, color="black")
    flierprops = dict(marker="o", markerfacecolor="#ef4444", markeredgecolor="#ef4444", markersize=1.8, alpha=0.6)

    for page_idx in range((len(valid_cols) + per_page - 1) // per_page):
        chunk = valid_cols[page_idx * per_page : (page_idx + 1) * per_page]

        fig = plt.figure(figsize=(13.2, 8.8))
        outer = fig.add_gridspec(
            nrows,
            ncols,
            left=0.02,
            right=0.995,
            bottom=0.03,
            top=0.95,
            wspace=0.18,
            hspace=0.32,
        )

        fig.text(0.01, 0.99, _page_label(page_idx), ha="left", va="top", fontsize=18, fontweight="bold")

        for i, col in enumerate(chunk):
            r = i // ncols
            c = i % ncols
            sub = outer[r, c].subgridspec(1, 2, wspace=0.05)

            ax_b = fig.add_subplot(sub[0, 0])
            ax_a = fig.add_subplot(sub[0, 1])

            orig = original_lists.get(col, [])
            corr = corrected_lists.get(col, [])

            for ax in (ax_b, ax_a):
                ax.set_facecolor("#EBEBEB")
                ax.grid(True, color="white", linewidth=0.8)
                ax.tick_params(axis="x", labelsize=7, length=2)
                ax.tick_params(axis="y", labelsize=7, length=2)

            if len(orig) > 0:
                bp0 = ax_b.boxplot(
                    orig,
                    vert=True,
                    patch_artist=True,
                    tick_labels=["Before"],
                    widths=0.78,
                    showfliers=True,
                    flierprops=flierprops,
                    boxprops=boxprops,
                    whiskerprops=whiskerprops,
                    capprops=capprops,
                    medianprops=medianprops,
                )
                for b in bp0.get("boxes", []):
                    b.set_facecolor("white")
            else:
                ax_b.axis("off")

            if len(corr) > 0:
                bp1 = ax_a.boxplot(
                    corr,
                    vert=True,
                    patch_artist=True,
                    tick_labels=["After"],
                    widths=0.78,
                    showfliers=True,
                    flierprops=flierprops,
                    boxprops=boxprops,
                    whiskerprops=whiskerprops,
                    capprops=capprops,
                    medianprops=medianprops,
                )
                for b in bp1.get("boxes", []):
                    b.set_facecolor("white")
            else:
                ax_a.axis("off")

            ax_a.tick_params(axis="y", labelleft=False)

            title = _panel_title_for_col(col)
            combined = Bbox.union([ax_b.get_position(), ax_a.get_position()])
            title_x = (combined.x0 + combined.x1) / 2
            title_y = min(combined.y1 + 0.006, 0.965)
            fig.text(title_x, title_y, title, ha="center", va="bottom", fontsize=8.5)

        out_path = os.path.join(out_dir, f"boxgrid_{_page_label(page_idx)}.png")
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)
        pages.append(out_path)

    return pages


def plot_boxgrid_single(
    cols: List[str],
    original_lists: Dict[str, List[float]],
    corrected_lists: Dict[str, List[float]],
    out_dir: str,
    ncols: int = 4,
    dpi: int = 300,
    filename: str = "boxgrid_all.png",
) -> str | None:
    """Combine Before/After boxplots of all variables into a single large mosaic image."""
    os.makedirs(out_dir, exist_ok=True)

    valid_cols = [c for c in cols if (len(original_lists.get(c, [])) > 0 or len(corrected_lists.get(c, [])) > 0)]
    if not valid_cols:
        return None

    ncols = max(1, int(ncols))
    nrows = int(np.ceil(len(valid_cols) / ncols))

    fig_w = 13.2
    fig_h = max(3.2, 2.05 * nrows)
    fig = plt.figure(figsize=(fig_w, fig_h))

    outer = fig.add_gridspec(
        nrows,
        ncols,
        left=0.02,
        right=0.995,
        bottom=0.03,
        top=0.98,
        wspace=0.20,
        hspace=0.38,
    )

    boxprops = dict(linewidth=0.95, color="black")
    whiskerprops = dict(linewidth=0.95, color="black")
    capprops = dict(linewidth=0.95, color="black")
    medianprops = dict(linewidth=1.2, color="black")
    flierprops = dict(marker="o", markerfacecolor="#ef4444", markeredgecolor="#ef4444", markersize=1.8, alpha=0.6)

    for i, col in enumerate(valid_cols):
        r = i // ncols
        c = i % ncols
        sub = outer[r, c].subgridspec(1, 2, wspace=0.05)

        ax_b = fig.add_subplot(sub[0, 0])
        ax_a = fig.add_subplot(sub[0, 1])

        orig = original_lists.get(col, [])
        corr = corrected_lists.get(col, [])

        for ax in (ax_b, ax_a):
            ax.set_facecolor("#EBEBEB")
            ax.grid(True, color="white", linewidth=0.8)
            ax.tick_params(axis="x", labelsize=7, length=2)
            ax.tick_params(axis="y", labelsize=7, length=2)

        if len(orig) > 0:
            bp0 = ax_b.boxplot(
                orig,
                vert=True,
                patch_artist=True,
                tick_labels=["Before"],
                widths=0.78,
                showfliers=True,
                flierprops=flierprops,
                boxprops=boxprops,
                whiskerprops=whiskerprops,
                capprops=capprops,
                medianprops=medianprops,
            )
            for b in bp0.get("boxes", []):
                b.set_facecolor("white")
        else:
            ax_b.axis("off")

        if len(corr) > 0:
            bp1 = ax_a.boxplot(
                corr,
                vert=True,
                patch_artist=True,
                tick_labels=["After"],
                widths=0.78,
                showfliers=True,
                flierprops=flierprops,
                boxprops=boxprops,
                whiskerprops=whiskerprops,
                capprops=capprops,
                medianprops=medianprops,
            )
            for b in bp1.get("boxes", []):
                b.set_facecolor("white")
        else:
            ax_a.axis("off")

        ax_a.tick_params(axis="y", labelleft=False)

        title = _panel_title_for_col(col)
        combined = Bbox.union([ax_b.get_position(), ax_a.get_position()])
        title_x = (combined.x0 + combined.x1) / 2
        title_y = min(combined.y1 + 0.006, 0.99)
        fig.text(title_x, title_y, title, ha="center", va="bottom", fontsize=8.5)

    total_cells = nrows * ncols
    for j in range(len(valid_cols), total_cells):
        r = j // ncols
        c = j % ncols
        ax = fig.add_subplot(outer[r, c])
        ax.axis("off")

    out_path = os.path.join(out_dir, filename)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def select_numeric_continuous_columns(df: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
    """Select numeric columns that look continuous."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    skipped: dict[str, str] = {}

    for col in list(numeric_cols):
        if col in EXCLUDE_COLUMNS:
            numeric_cols.remove(col)
            skipped[col] = 'EXCLUDE_COLUMNS'

    if not AUTO_SKIP_LOW_CARDINALITY_NUMERIC:
        return numeric_cols, skipped

    selected: list[str] = []
    for col in numeric_cols:
        s = df[col]
        non_na = s.dropna()
        n = int(non_na.shape[0])
        if n == 0:
            skipped[col] = 'no non-NA values'
            continue

        nunique = int(non_na.nunique(dropna=True))
        unique_ratio = nunique / max(n, 1)

        if nunique <= LOW_CARDINALITY_MAX_UNIQUE and unique_ratio <= LOW_CARDINALITY_MAX_UNIQUE_RATIO:
            skipped[col] = f'low-cardinality numeric (nunique={nunique}, ratio={unique_ratio:.3f})'
            continue

        selected.append(col)

    return selected, skipped


def compute_iqr_bounds(values: pd.Series) -> Tuple[float, float, float, float]:
    """Calculate IQR and its upper/lower bounds."""
    q1 = values.quantile(0.25, interpolation='linear')
    q3 = values.quantile(0.75, interpolation='linear')
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return q1, q3, lower, upper


def normalize_cell_value(value: Any) -> Any:
    """Normalize cell value: remove %, process > and < prefixes and try converting to float."""
    if pd.isna(value):
        return value

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return np.nan

        if text.endswith('%'):
            text = text[:-1].strip()
            if not text:
                return np.nan

        if text[0] in {'>', '<'}:
            remainder = text[1:].strip()
            if remainder:
                try:
                    numeric = float(remainder)
                    factor = 1.1 if text[0] == '>' else 0.9
                    return numeric * factor
                except ValueError:
                    pass

        try:
            return float(text)
        except ValueError:
            return text

    return value


def process_column(series: pd.Series) -> Tuple[List[float], List[float], pd.Series]:
    """Process a column using IQR method."""
    original_list = series.dropna().tolist()
    if len(original_list) == 0:
        return [], [], series

    q1, q3, lower, upper = compute_iqr_bounds(series.dropna())
    corrected_series = series.clip(lower=lower, upper=upper)
    corrected_list = corrected_series.dropna().tolist()

    return original_list, corrected_list, corrected_series


def plot_boxpair(col_name: str, original: List[float], corrected: List[float], save_path: str) -> None:
    """Draw side-by-side boxplots: original on the left, modified on the right."""
    if len(original) == 0 and len(corrected) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.2))
    display_name = _panel_title_for_col(col_name)

    flierprops = dict(marker="o", markerfacecolor="#ef4444", markeredgecolor="#ef4444", markersize=2.2, alpha=0.6)
    boxprops = dict(linewidth=0.9, color="black")
    whiskerprops = dict(linewidth=0.9, color="black")
    capprops = dict(linewidth=0.9, color="black")
    medianprops = dict(linewidth=1.1, color="black")

    for ax in axes:
        ax.set_facecolor("#EBEBEB")
        ax.grid(True, color="white", linewidth=0.8)

    if len(original) > 0:
        bp0 = axes[0].boxplot(
            original,
            vert=True,
            patch_artist=True,
            tick_labels=["Before"],
            showfliers=True,
            flierprops=flierprops,
            boxprops=boxprops,
            whiskerprops=whiskerprops,
            capprops=capprops,
            medianprops=medianprops,
        )
        for b in bp0.get("boxes", []):
            b.set_facecolor("white")
    axes[0].set_title(f"{display_name} - Before", fontsize=10)

    if len(corrected) > 0:
        bp1 = axes[1].boxplot(
            corrected,
            vert=True,
            patch_artist=True,
            tick_labels=["After"],
            showfliers=True,
            flierprops=flierprops,
            boxprops=boxprops,
            whiskerprops=whiskerprops,
            capprops=capprops,
            medianprops=medianprops,
        )
        for b in bp1.get("boxes", []):
            b.set_facecolor("white")
    axes[1].set_title(f"{display_name} - After", fontsize=10)

    fig.suptitle(f"{display_name} (Before vs After)", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def main() -> int:
    if not os.path.exists(INPUT_EXCEL):
        print(f'Input file not found: {INPUT_EXCEL}\n')
        return 1

    try:
        df = pd.read_excel(INPUT_EXCEL, sheet_name=0, na_values=['/'])
    except Exception as e:
        print(f'Failed to read Excel: {e}')
        return 1

    df.replace(r'^\s*/\s*$', np.nan, regex=True, inplace=True)

    try:
        df = df.map(normalize_cell_value)
    except Exception:
        df = df.applymap(normalize_cell_value)

    df, derived_missing_masks = add_derived_columns(df)

    if TARGET_COLUMNS:
        numeric_cols = [c for c in TARGET_COLUMNS if c in df.columns]
        if not numeric_cols:
            print('Columns in TARGET_COLUMNS were not found in the data, please check names.')
            return 1
    else:
        numeric_cols, skipped = select_numeric_continuous_columns(df)
        if skipped:
            print('Automatically skipping numeric columns that look categorical (No IQR clipping):')
            for col, reason in sorted(skipped.items()):
                print(f'  - {col}: {reason}')

        if len(numeric_cols) == 0:
            print('No numeric columns detected, check data format.')
            return 0

    corrected_df = df.copy()

    original_lists: Dict[str, List[float]] = {}
    corrected_lists: Dict[str, List[float]] = {}

    for idx, col in enumerate(numeric_cols, start=1):
        series = df[col]
        original_list, corrected_list, corrected_series = process_column(series)

        if len(original_list) > 0:
            original_lists[col] = original_list
        if len(corrected_list) > 0:
            corrected_lists[col] = corrected_list

        corrected_df[col] = corrected_series

        if SAVE_SINGLE_PLOTS:
            if len(original_list) > 0 or len(corrected_list) > 0:
                img_name = f'box_{idx:02d}_{safe_filename(col)}.png'
                img_path = os.path.join(OUTPUT_DIR, img_name)
                plot_boxpair(str(col), original_list, corrected_list, img_path)
                print(f'Boxplot saved: {img_path}')
            else:
                print(f'Column "{col}" has no valid data, skipping plot.')

    try:
        out_img = plot_boxgrid_single(
            cols=numeric_cols,
            original_lists=original_lists,
            corrected_lists=corrected_lists,
            out_dir=OUTPUT_DIR,
            ncols=4,
            dpi=300,
            filename="boxgrid_all.png",
        )
        if out_img:
            print("Mosaic boxplot saved (Single page, 4-column layout):")
            print(f"  - {out_img}")
    except Exception as e:
        print(f"[WARN] Mosaic boxplot generation failed (does not affect Excel export): {e}")

    try:
        for col_name, miss_mask in derived_missing_masks.items():
            if col_name in corrected_df.columns:
                if corrected_df[col_name].dtype != object:
                    corrected_df[col_name] = corrected_df[col_name].astype(object)
                mask = miss_mask | corrected_df[col_name].isna()
                corrected_df.loc[mask, col_name] = '/'

        corrected_df.to_excel(OUTPUT_EXCEL, index=False)
        print(f'Cleaned Excel saved: {OUTPUT_EXCEL}')
    except Exception as e:
        print(f'Failed to save Excel: {e}')
        return 1

    if SAVE_SINGLE_PLOTS:
        print(f'Processed: {len(numeric_cols)} numeric columns. Images generated: {len(os.listdir(OUTPUT_DIR)) - 1}.')
    else:
        print(f'Processed: {len(numeric_cols)} numeric columns. Mosaics generated: {1 if "out_img" in locals() and out_img else 0}.')
    return 0


if __name__ == '__main__':
    sys.exit(main())