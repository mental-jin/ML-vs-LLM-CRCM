# -*- coding: utf-8 -*-
"""
简单的 Excel 异常值处理与绘图脚本（硬编码路径）

功能：
1. 读取预设的 Excel 文件（同目录下的 data.xlsx，默认读取首个工作表）。
2. 删除%,替换>和<前缀并尝试转为数值。>转换为1.1倍，<转换为0.9倍。
3. 对数值型列按 IQR 方法（Q1-1.5*IQR，Q3+1.5*IQR）界定异常值，并将异常值裁剪为界限值。
4. 默认输出分页“拼图”箱线图（4×6）：每个变量拆成 Before/After 两个小子图，增强对比；单列图可选开启。
5. 所有列处理完成后，将修正后的数据另存为新的 Excel（output/cleaned_data.xlsx）。

说明：
- 仅处理数值型列（非数值列跳过）。
- 原值 list 与修正后 list 均不包含空值（NaN）；若该列无有效数值，则不生成该列图片。
- 为保持简单，路径与参数均使用硬编码。
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

# 解决中文/全角符号在 Matplotlib 下的字体缺失问题（Windows 常见）：
# 优先使用常见中文字体；若系统无这些字体，Matplotlib 将自动回退。
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

# 让整体更接近示例图（灰底+白色网格）
plt.style.use('ggplot')

# --------------------------- 硬编码的文件与目录 ---------------------------
# 预设 Excel 文件名：请将 data.xlsx 放到与本脚本同一目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MERGE_DIR = os.path.join(SCRIPT_DIR, 'merge')
INPUT_EXCEL = os.path.join(MERGE_DIR, '228-show.xlsx')  # 硬编码输入

OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'merge_otclimit')      # 结果输出目录
OUTPUT_EXCEL = os.path.join(OUTPUT_DIR, '228 show-cleaned_data.xlsx')

# 只保留“拼图”输出（更贴近示例图）。如需恢复单列图，把它改为 True。
SAVE_SINGLE_PLOTS = False

# 可选：硬编码需要处理的列名（保证为连续数值变量）。
# 为空列表时，默认处理所有数值型列。
# 示例：TARGET_COLUMNS = ["身高", "体重", "BMI"]
# TARGET_COLUMNS: List[str] = ["肿瘤最大径", "肿瘤体积"]
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

# 显式排除列（即便它们是数值列也不处理）。
EXCLUDE_COLUMNS: List[str] = []

# 只处理连续型变量的辅助配置：
# - 当 TARGET_COLUMNS 为空时，会在所有数值列里自动剔除“看起来像分类变量”的列（例如 0/1/2 编码）。
# - 你也可以在这里手工排除某些列（即使它们是数值列）。


# 自动跳过“低基数数值列”（常见为数值编码分类变量）
AUTO_SKIP_LOW_CARDINALITY_NUMERIC = True

# 若某数值列的非空唯一值数量 <= 该阈值，则默认认为更像分类变量并跳过
LOW_CARDINALITY_MAX_UNIQUE = 8

# 同时参考唯一值占比（避免样本量很大时把连续变量误判为分类）
# unique_ratio = nunique / non_na_count
LOW_CARDINALITY_MAX_UNIQUE_RATIO = 0.05

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 加载列名映射（中文 -> 英文），若不存在则使用空映射
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
    """将列名转为适合作为文件名的字符串（移除非法字符）。"""
    # Windows 兼容：尽量只保留安全字符，避免因不可见字符/特殊符号导致保存失败
    text = unicodedata.normalize('NFKC', str(name)).strip()
    # 仅保留字母数字与少量安全符号，其余全部替换为下划线
    text = re.sub(r'[^0-9A-Za-z._-]+', '_', text)
    text = text.strip('._-')
    if not text:
        return 'col'
    # 避免超长文件名
    return text[:120]


def _panel_title_for_col(col_name: str) -> str:
    """返回用于图标题的展示名（含单位），并尽量做中英文映射。"""
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
    """将多个变量的 Before/After 箱线图拼成分页大图（更接近示例图）。

    关键点：每个变量一个小面板，但面板内拆成左右两块（Before / After），
    让两者各自独立 y 轴范围：对比会更明显（Before 更“压缩”且极端值更突出，After 更“均匀/展开”）。

    返回生成的图片路径列表。
    """
    os.makedirs(out_dir, exist_ok=True)

    # 仅保留有有效数据的列
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

        # 更窄、更满：减小画布与边距，减少空白
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

        # 大图左上角 A/B/C… 标注
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

            # 省空间：右侧不显示 y tick labels
            ax_a.tick_params(axis="y", labelleft=False)

            # 变量名居中：位于 Before/After 两张图的中间（用 fig.text 放在两张子图的联合 bbox 上方）
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
    """将所有变量的 Before/After 箱线图拼成一张大图。

    - 每个变量一个面板；面板内为左右 Before/After 两个子图。
    - 外层固定每行 ncols 个变量面板（按需求默认 4）。

    返回生成图片路径；若无有效列则返回 None。
    """
    os.makedirs(out_dir, exist_ok=True)

    valid_cols = [c for c in cols if (len(original_lists.get(c, [])) > 0 or len(corrected_lists.get(c, [])) > 0)]
    if not valid_cols:
        return None

    ncols = max(1, int(ncols))
    nrows = int(np.ceil(len(valid_cols) / ncols))

    # 经验尺寸：宽度固定，按行数增长高度；避免过度空白
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

    # 若最后一行不满，关闭多余面板
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
    """Select numeric columns that look continuous.

    Returns:
        (selected_columns, skipped_reason_by_column)
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    skipped: dict[str, str] = {}

    # 先排除用户显式指定的列
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

        # 低基数 + 低占比：更像分类变量（例如 0/1/2 或 1~4 分级）
        if nunique <= LOW_CARDINALITY_MAX_UNIQUE and unique_ratio <= LOW_CARDINALITY_MAX_UNIQUE_RATIO:
            skipped[col] = f'low-cardinality numeric (nunique={nunique}, ratio={unique_ratio:.3f})'
            continue

        selected.append(col)

    return selected, skipped


def compute_iqr_bounds(values: pd.Series) -> Tuple[float, float, float, float]:
    """
    计算 IQR 及其上下界限。
    返回：(Q1, Q3, lower_bound, upper_bound)
    """
    # dropna 仅用于分位数计算，不改变原 Series
    q1 = values.quantile(0.25, interpolation='linear')
    q3 = values.quantile(0.75, interpolation='linear')
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return q1, q3, lower, upper


def normalize_cell_value(value: Any) -> Any:
    """标准化单元格值：去掉%, 处理前缀>,<并尝试转为数值。"""
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
    """
    按 IQR 方法处理一列数据：
    - 原值列表：去除空值后的值
    - 修正列表：将异常值裁剪至界限后，再去除空值
    - 返回修正后的 Series（保持与原索引一致）
    """
    # 原值（去除空值）
    original_list = series.dropna().tolist()
    if len(original_list) == 0:
        # 无有效数据，返回原样
        return [], [], series

    # 计算上下界限
    q1, q3, lower, upper = compute_iqr_bounds(series.dropna())

    # 使用 clip 将异常值裁剪到界限内，NaN 保持 NaN 不变
    corrected_series = series.clip(lower=lower, upper=upper)

    # 修正后列表（去除空值）
    corrected_list = corrected_series.dropna().tolist()

    return original_list, corrected_list, corrected_series


def plot_boxpair(col_name: str, original: List[float], corrected: List[float], save_path: str) -> None:
    """
    绘制并排箱线图：左侧原值、右侧修正后，保存图片。
    """
    # 若没有有效数据，不绘图
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
    # 基本检查：输入文件是否存在
    if not os.path.exists(INPUT_EXCEL):
        print(f'未找到输入文件：{INPUT_EXCEL}\n')
        return 1

    # 读取 Excel（默认首个工作表）
    try:
        # 将 "/" 读作缺失值
        df = pd.read_excel(INPUT_EXCEL, sheet_name=0, na_values=['/'])
    except Exception as e:
        print(f'读取 Excel 失败：{e}')
        return 1

    # 保险起见：将形如"/"（两侧可能有空白）的内容统一视为缺失
    df.replace(r'^\s*/\s*$', np.nan, regex=True, inplace=True)

    # 读取后立即清洗：去掉%以及处理以>/<开头的值
    # pandas 新版本中 applymap 已弃用，优先使用 DataFrame.map
    try:
        df = df.map(normalize_cell_value)
    except Exception:
        df = df.applymap(normalize_cell_value)

    # 增加派生列（先以数值形式加入，便于后续异常值处理/绘图；导出前再按规则写入“/”）
    df, derived_missing_masks = add_derived_columns(df)

    # 列选择：若设置了 TARGET_COLUMNS，则仅处理该列表；否则处理所有数值型列
    if TARGET_COLUMNS:
        # 仅保留在数据中存在的列名（用户保证为数值型）
        numeric_cols = [c for c in TARGET_COLUMNS if c in df.columns]
        if not numeric_cols:
            print('TARGET_COLUMNS 中的列未在数据中找到，请检查列名。')
            return 1
    else:
        numeric_cols, skipped = select_numeric_continuous_columns(df)
        if skipped:
            print('自动跳过疑似分类/不可处理的数值列（不会做IQR裁剪）：')
            for col, reason in sorted(skipped.items()):
                print(f'  - {col}: {reason}')

        if len(numeric_cols) == 0:
            print('未检测到数值型列，可检查数据格式或内容。')
            return 0

    corrected_df = df.copy()

    # 用于（可选）保存非空列表，若为空则不存入
    original_lists: Dict[str, List[float]] = {}
    corrected_lists: Dict[str, List[float]] = {}

    for idx, col in enumerate(numeric_cols, start=1):
        series = df[col]
        original_list, corrected_list, corrected_series = process_column(series)

        # 仅当列表非空时才保存到 dict
        if len(original_list) > 0:
            original_lists[col] = original_list
        if len(corrected_list) > 0:
            corrected_lists[col] = corrected_list

        # 写回修正后的列
        corrected_df[col] = corrected_series

        # 单列图（可选，默认关闭）
        if SAVE_SINGLE_PLOTS:
            if len(original_list) > 0 or len(corrected_list) > 0:
                img_name = f'box_{idx:02d}_{safe_filename(col)}.png'
                img_path = os.path.join(OUTPUT_DIR, img_name)
                plot_boxpair(str(col), original_list, corrected_list, img_path)
                print(f'已保存箱线图：{img_path}')
            else:
                print(f'列 "{col}" 无有效数据，跳过绘图。')

    # 额外输出“拼图版”箱线图（更接近示例图风格）：每个变量一个小面板，分页 A/B/C...
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
            print("已保存拼图箱线图（单张，4列排版）：")
            print(f"  - {out_img}")
    except Exception as e:
        print(f"[WARN] 拼图箱线图生成失败（不影响Excel导出）：{e}")

    # 保存修正后的 Excel
    try:
        # 将派生列中“由于原始数据缺失/无效导致无法计算”的位置写入 '/'
        for col_name, miss_mask in derived_missing_masks.items():
            if col_name in corrected_df.columns:
                # 写入字符串占位符前先转为 object，避免将来 pandas 对 float 列赋值字符串报错
                if corrected_df[col_name].dtype != object:
                    corrected_df[col_name] = corrected_df[col_name].astype(object)
                mask = miss_mask | corrected_df[col_name].isna()
                corrected_df.loc[mask, col_name] = '/'

        corrected_df.to_excel(OUTPUT_EXCEL, index=False)
        print(f'已保存修正后的 Excel：{OUTPUT_EXCEL}')
    except Exception as e:
        print(f'保存 Excel 失败：{e}')
        return 1

    # 可选：简单输出有无被记录的列
    if SAVE_SINGLE_PLOTS:
        print(f'已处理数值列：{len(numeric_cols)} 列。生成图片：{len(os.listdir(OUTPUT_DIR)) - 1} 张（含 Excel 文件除外）。')
    else:
        print(f'已处理数值列：{len(numeric_cols)} 列。已生成拼图：{1 if "out_img" in locals() and out_img else 0} 张。')
    return 0


if __name__ == '__main__':
    sys.exit(main())