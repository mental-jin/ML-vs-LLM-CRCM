"""
独立脚本：重新绘制 Variable Heatmap
直接复制原脚本的 save_variable_heatmaps_by_model 函数，确保输出完全一致

用法：
    python redraw_variable_heatmap.py
    python redraw_variable_heatmap.py --run_dir ml_results/run_20260206_234650

参数说明：
    --run_dir: 已运行的结果目录（默认: ml_results/run_20260206_234650）
    --target: 因变量列名（默认: Metastasis）
    --top_n: 显示的特征数量（默认: 20，与原脚本一致）
    --models: 指定要绘制的模型（逗号分隔，如 LR,RF,XGB；默认: 全部）
    --output_dir: 输出目录（默认为 run_dir/fig_new）
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin


# ============ 自定义类（与原脚本保持一致，用于加载模型）============
class ColumnSelector(TransformerMixin, BaseEstimator):
    """特征列选择器"""
    def __init__(self, indices: np.ndarray):
        self.indices = np.asarray(indices, dtype=int)

    def fit(self, X: Any, y: Any = None):
        return self

    def transform(self, X: Any):
        if sp.issparse(X):
            return X[:, self.indices]
        return np.asarray(X)[:, self.indices]


class ToDense(TransformerMixin, BaseEstimator):
    """稀疏矩阵转密集矩阵"""
    def fit(self, X: Any, y: Any = None):
        return self

    def transform(self, X: Any):
        if sp.issparse(X):
            return X.toarray()
        return np.asarray(X)


# ============ 常量定义 ============
DEMOGRAPHIC_COLUMNS_EXCLUDE = ["Sex"]

# ============ 字体大小配置（可手动调节）============
FONTSIZE_FEATURE_LABEL = 12      # 热图右侧特征标签字体大小
FONTSIZE_ANNOT_LABEL = 12         # 注释行（Predicted value/Outcome/Group）标签字体大小
FONTSIZE_LEGEND_TITLE = 12       # Legend 标题（变量名）字体大小
FONTSIZE_LEGEND_TEXT = 10         # Legend 色块旁标签字体大小
FONTSIZE_COLORBAR_TICK = 10       # 连续变量 colorbar 刻度标签字体大小


# ============ 辅助函数 ============
def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _safe_float32_matrix(X: Any) -> Any:
    if sp.issparse(X):
        X = X.tocsr(copy=False)
        return X.astype(np.float32) if X.dtype != np.float32 else X
    X = np.asarray(X)
    return X.astype(np.float32) if X.dtype != np.float32 else X


def get_feature_names(preprocessor) -> list[str]:
    try:
        names = preprocessor.get_feature_names_out()
        return [str(x) for x in names]
    except Exception:
        return []


def _clean_display_feature_names(names: list[str]) -> list[str]:
    """移除 sklearn 前缀（如 num__、cat__）"""
    out: list[str] = []
    for n in names:
        if "__" in n:
            n = n.split("__", 1)[-1]
        out.append(n)
    return out


# ============ 辅助函数：解析预处理后特征名 → 原始列名 ============
def _parse_internal_feature_name(
    feature_name: str, raw_columns: list[str]
) -> tuple[str | None, str | None, bool]:
    """解析 sklearn ColumnTransformer 生成的特征名称。

    返回 (raw_column_name, category_value_str, is_categorical)
    例如:
        'num__CEA'               → ('CEA', None, False)
        'cat__Carcinoma nodule_0' → ('Carcinoma nodule', '0', True)
        'cat__T stage_T1-T2'     → ('T stage', 'T1-T2', True)
    """
    if feature_name.startswith("num__"):
        col = feature_name[5:]
        return (col, None, False) if col in raw_columns else (None, None, False)
    if feature_name.startswith("cat__"):
        rest = feature_name[5:]
        # 按列名长度从长到短匹配（避免下划线歧义）
        for col in sorted(raw_columns, key=len, reverse=True):
            if rest.startswith(col + "_"):
                return col, rest[len(col) + 1:], True
            if rest == col:
                return col, None, True
        # 回退：从右侧切一刀
        parts = rest.rsplit("_", 1)
        if len(parts) == 2:
            return (parts[0], parts[1], True) if parts[0] in raw_columns else (None, None, True)
        return (None, None, True)
    return (None, None, False)


def _convert_cat_val(s: str):
    """将类别值字符串转为合适类型（int > float > str）。"""
    try:
        return int(s)
    except (ValueError, TypeError):
        try:
            return float(s)
        except (ValueError, TypeError):
            return s


# ============ 原脚本的 save_variable_heatmaps_by_model 函数（完整复制）============
def save_variable_heatmaps_by_model(
    out_dir: Path,
    X_train: Any,
    X_test: Any,
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_proba_train_by_model: dict[str, np.ndarray],
    y_proba_test_by_model: dict[str, np.ndarray],
    importance_tables: dict[str, pd.DataFrame],
    top_n: int | None = None,
    X_external: Any | None = None,
    y_external: np.ndarray | None = None,
    y_proba_external_by_model: dict[str, np.ndarray] | None = None,
    raw_df: pd.DataFrame | None = None,
) -> None:
    """每个模型单独输出一张"变量热图"，仿照示例图风格：

    - 行：该模型特征（默认显示该模型全部入模特征；如指定 top_n 则取 Top-N）
    - 列：样本（训练集 + 测试集 + 外部验证集拼接）
    - 特征标签在右侧
    - 额外三行注释：Predicted value / Outcome / Group
    - 底部legend区：每个变量一个垂直colorbar，网格布局

    说明：为避免不同变量量纲影响颜色，本函数对每个变量行做 min-max 归一化到 [0,1]。
    """
    if not importance_tables:
        print("[FIG] No importance tables; skip variable heatmaps.")
        return

    _ensure_dir(out_dir)

    # 判断是否有外部验证集
    has_external = (
        X_external is not None
        and y_external is not None
        and y_proba_external_by_model is not None
        and len(y_external) > 0
    )

    # 拼接样本
    if has_external:
        if sp.issparse(X_train) or sp.issparse(X_test) or sp.issparse(X_external):
            X_all = sp.vstack([X_train, X_test, X_external], format="csr")
        else:
            X_all = np.vstack([np.asarray(X_train), np.asarray(X_test), np.asarray(X_external)])
        y_all = np.concatenate([
            np.asarray(y_train).ravel(),
            np.asarray(y_test).ravel(),
            np.asarray(y_external).ravel()
        ]).astype(int)
        group_all = np.array(
            ["Training set"] * int(len(y_train))
            + ["Test set"] * int(len(y_test))
            + ["External set"] * int(len(y_external)),
            dtype=object
        )
    else:
        if sp.issparse(X_train) or sp.issparse(X_test):
            X_all = sp.vstack([X_train, X_test], format="csr")
        else:
            X_all = np.vstack([np.asarray(X_train), np.asarray(X_test)])
        y_all = np.concatenate([np.asarray(y_train).ravel(), np.asarray(y_test).ravel()]).astype(int)
        group_all = np.array(["Training set"] * int(len(y_train)) + ["Test set"] * int(len(y_test)), dtype=object)

    outcome_all = np.where(y_all == 1, "M1", "M0")

    key_order = ["LR", "DT", "RF", "SVM", "KNN", "NB", "XGB", "SGBT", "NNET"]
    present = [k for k in key_order if k in importance_tables]
    present += [k for k in importance_tables.keys() if k not in present]

    for key in present:
        imp_df = importance_tables.get(key)
        if imp_df is None or imp_df.empty:
            continue

        p_tr = y_proba_train_by_model.get(key)
        p_te = y_proba_test_by_model.get(key)
        if p_tr is None or p_te is None:
            continue
        
        # 拼接预测概率
        if has_external and y_proba_external_by_model is not None:
            p_ex = y_proba_external_by_model.get(key)
            if p_ex is not None:
                p_all = np.concatenate([
                    np.asarray(p_tr).ravel(),
                    np.asarray(p_te).ravel(),
                    np.asarray(p_ex).ravel()
                ]).astype(float)
            else:
                p_all = np.concatenate([np.asarray(p_tr).ravel(), np.asarray(p_te).ravel()]).astype(float)
        else:
            p_all = np.concatenate([np.asarray(p_tr).ravel(), np.asarray(p_te).ravel()]).astype(float)

        sub = imp_df.copy()
        if "abs_importance" not in sub.columns:
            sub["abs_importance"] = sub["importance"].abs()
        sub = sub.sort_values("abs_importance", ascending=False)
        if top_n is not None:
            sub = sub.head(int(top_n))
        if "feature_idx" not in sub.columns:
            print(f"[FIG] {key}: missing feature_idx; skip variable heatmap.")
            continue

        feat_idx = sub["feature_idx"].astype(int).tolist()
        if "feature_display" in sub.columns:
            feat_labels = sub["feature_display"].astype(str).tolist()
        else:
            feat_labels = _clean_display_feature_names(sub["feature"].astype(str).tolist())

        # ---------- 构建特征值矩阵 ----------
        # 优先使用原始数据（raw_df），保留真实 0/1 值和原始尺度；
        # 仅在无法映射时回退到预处理后矩阵。
        n_feat_tmp = len(feat_idx)
        n_samp_tmp = X_all.shape[0]
        feat_internals = sub["feature"].astype(str).tolist() if "feature" in sub.columns else [""] * n_feat_tmp
        raw_cols_list = list(raw_df.columns) if raw_df is not None else []

        vals = np.full((n_feat_tmp, n_samp_tmp), np.nan, dtype=float)
        for _fi in range(n_feat_tmp):
            mapped = False
            if raw_df is not None and len(raw_df) == n_samp_tmp:
                col_name, cat_val_str, is_cat = _parse_internal_feature_name(
                    feat_internals[_fi], raw_cols_list
                )
                if col_name is not None and col_name in raw_df.columns:
                    raw_v = raw_df[col_name].values
                    if is_cat and cat_val_str is not None:
                        # One-hot 还原：1 = 匹配该类别, 0 = 不匹配
                        target = _convert_cat_val(cat_val_str)
                        nan_m = pd.isna(raw_v)
                        try:
                            matches = (raw_v == target)
                        except TypeError:
                            matches = (raw_v.astype(str) == str(target))
                        row_arr = np.where(matches, 1.0, 0.0)
                        if nan_m.any():
                            row_arr = row_arr.astype(float)
                            row_arr[nan_m] = np.nan
                        vals[_fi] = row_arr
                    else:
                        vals[_fi] = pd.to_numeric(raw_v, errors="coerce").astype(float)
                    mapped = True
            if not mapped:
                col_pp = X_all[:, feat_idx[_fi]]
                vals[_fi] = (
                    col_pp.toarray().ravel()
                    if sp.issparse(col_pp)
                    else np.asarray(col_pp).ravel()
                ).astype(float)

        # 每行 min-max 归一化到 [0,1]
        row_min = np.nanmin(vals, axis=1, keepdims=True)
        row_max = np.nanmax(vals, axis=1, keepdims=True)
        den = np.where((row_max - row_min) == 0, 1.0, (row_max - row_min))
        vals01 = (vals - row_min) / den
        vals01 = np.clip(vals01, 0.0, 1.0)

        # 样本排序：Outcome → Group → 预测概率(降序)
        # Group编码：Training=0, Test=1, External=2
        group_code_all = np.zeros(len(group_all), dtype=int)
        group_code_all[group_all == "Test set"] = 1
        group_code_all[group_all == "External set"] = 2
        outcome_code_all = (y_all == 1).astype(int)  # M0=0, M1=1
        order = np.lexsort((-p_all, group_code_all, outcome_code_all))
        vals01 = vals01[:, order]
        vals_sorted = vals[:, order]
        p_sorted = p_all[order]
        outcome_sorted = outcome_all[order]
        group_sorted = group_all[order]
        group_code_sorted = group_code_all[order]
        outcome_code_sorted = outcome_code_all[order]

        n_feat = int(vals01.shape[0])
        n_samp = int(vals01.shape[1])

        def _fmt_tick(v: float) -> str:
            if not np.isfinite(v):
                return ""
            av = abs(float(v))
            if av >= 100:
                return f"{v:.0f}"
            if av >= 10:
                return f"{v:.1f}"
            return f"{v:.2f}"

        # 示例图配色方案 - 高对比度、差异明显的颜色（参考示例图）
        # 按照示例图风格：棕色系、黄色系、紫色系、粉色系、绿色系、蓝色系等交替
        # 确保相邻颜色对比明显，避免紫色系连续出现
        example_colors = [
            "#C68642",  # 深棕橙
            "#F5C342",  # 金黄
            "#2196F3",  # 亮蓝（原为深紫，改为蓝色增加对比）
            "#E91E63",  # 玫红
            "#795548",  # 棕色
            "#4CAF50",  # 绿色
            "#FF9800",  # 橙色（原为靛紫）
            "#9C27B0",  # 紫色
            "#00BCD4",  # 青色（原为紫色，增加对比）
            "#8BC34A",  # 浅绿（原为蓝色）
            "#FF5722",  # 深橙
            "#3F51B5",  # 靛蓝
            "#CDDC39",  # 黄绿
            "#673AB7",  # 靛紫
            "#009688",  # 深青
            "#E040FB",  # 亮紫
            "#FF6F00",  # 琥珀
            "#1E88E5",  # 亮蓝
            "#43A047",  # 深绿
            "#F06292",  # 粉色
            "#7E57C2",  # 中紫
            "#26A69A",  # 深青
            "#FFA000",  # 深黄
            "#5E35B1",  # 深靛
        ]
        # 扩展颜色列表以覆盖所有特征
        base_colors = []
        for i in range(n_feat):
            base_colors.append(matplotlib.colors.to_rgba(example_colors[i % len(example_colors)]))

        # 分类变量0值用浅灰色（而非白色），与示例图一致
        light_rgba = matplotlib.colors.to_rgba("#F5F5DC")  # 浅米灰色
        rgba = np.zeros((n_feat, n_samp, 4), dtype=float)
        row_cm: list[Any] = []
        row_ticks: list[tuple[list[float], list[str]]] = []
        row_min_max: list[tuple[float, float]] = []

        for i in range(n_feat):
            v_raw = np.asarray(vals_sorted[i, :], dtype=float)
            v01 = np.asarray(vals01[i, :], dtype=float)
            v_raw_finite = v_raw[np.isfinite(v_raw)]
            uniq = np.unique(v_raw_finite) if v_raw_finite.size else np.array([])
            base_rgba = base_colors[i]

            mn = float(np.nanmin(v_raw_finite)) if v_raw_finite.size else 0.0
            mx = float(np.nanmax(v_raw_finite)) if v_raw_finite.size else 1.0
            row_min_max.append((mn, mx))

            # 判断离散/连续
            is_binary_exact = uniq.size <= 2 and uniq.size > 0 and np.all(np.isin(uniq, [0.0, 1.0]))
            is_binary_any2 = uniq.size == 2
            is_binary01 = is_binary_exact or is_binary_any2
            is_small_discrete = (not is_binary01) and (uniq.size > 0) and (uniq.size <= 3)

            if is_binary01:
                if uniq.size >= 2:
                    high_val = uniq[-1]
                else:
                    high_val = 1.0
                mask1 = np.isclose(v_raw, high_val)
                rgba[i, :, :] = light_rgba
                rgba[i, mask1, :] = base_rgba
                row_cm.append(matplotlib.colors.ListedColormap([light_rgba, base_rgba]))
                row_ticks.append(([0, 1], ["0", "1"]))
            elif is_small_discrete:
                levels = np.sort(uniq)
                colors = [light_rgba]
                for t in np.linspace(0.45, 1.0, num=int(levels.size) - 1):
                    c = (
                        light_rgba[0] * (1 - t) + base_rgba[0] * t,
                        light_rgba[1] * (1 - t) + base_rgba[1] * t,
                        light_rgba[2] * (1 - t) + base_rgba[2] * t,
                        1.0,
                    )
                    colors.append(c)
                cm = matplotlib.colors.ListedColormap(colors)
                rgba[i, :, :] = light_rgba
                for j, lv in enumerate(levels):
                    mask = np.isclose(v_raw, lv)
                    rgba[i, mask, :] = colors[min(j, len(colors) - 1)]
                row_cm.append(cm)
                row_ticks.append((list(range(int(levels.size))), [_fmt_tick(lv) for lv in levels]))
            else:
                cm = matplotlib.colors.LinearSegmentedColormap.from_list(
                    "row", [light_rgba, base_rgba]
                )
                rgba[i, :, :] = cm(np.clip(v01, 0.0, 1.0))
                row_cm.append(cm)
                row_ticks.append(([0, 1], [_fmt_tick(mn), _fmt_tick(mx)]))

        # ============ 布局设计（仿照示例图）============
        # legend区: 每个变量一个垂直colorbar，固定每行6个
        legend_ncol = 6  # 固定每行6个图注
        n_legend_items = n_feat + 3  # 特征 + predicted value + outcome + group
        legend_nrows = int(math.ceil(n_legend_items / legend_ncol))

        # 图尺寸 - 调整legend区高度
        fig_w = max(10.0, n_samp / 100.0)
        heatmap_h = 0.28 * n_feat
        annot_h = 0.3 * 3  # 3个注释行（缩小高度）
        legend_h = 1.8 * legend_nrows  # 每行高度（更紧凑）
        fig_h = heatmap_h + annot_h + legend_h + 4.0  # 底部留白

        fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
        # 总布局：热图 | 3个注释行 | 留白间隔 | legend区
        # 注释行高度与变量行一致（都是1.0），hspace=0 使行之间紧密排列
        gs_main = fig.add_gridspec(
            6, 1,
            height_ratios=[n_feat, 1.0, 1.0, 1.0, 1.2, legend_nrows * 3.5 + 2.0],  # 调整间隔行和legend区高度
            hspace=0.0,
            top=0.96, bottom=0.04, left=0.02, right=0.88
        )

        # ============ 主热图 ============
        ax0 = fig.add_subplot(gs_main[0, 0])
        ax0.imshow(rgba, aspect="auto", interpolation="nearest")
        ax0.set_yticks(np.arange(n_feat))
        ax0.set_yticklabels([])  # 左边不显示标签
        ax0.set_xticks([])
        ax0.tick_params(left=False, bottom=False)
        for spine in ax0.spines.values():
            spine.set_visible(False)
        # 添加行间细白色分隔线
        for i in range(1, n_feat):
            ax0.axhline(y=i - 0.5, color="white", linewidth=0.5)

        # 右侧标签 - 字号调大
        ax0_right = ax0.secondary_yaxis("right")
        ax0_right.set_yticks(np.arange(n_feat))
        ax0_right.set_yticklabels(feat_labels, fontsize=FONTSIZE_FEATURE_LABEL)
        ax0_right.tick_params(right=False)
        # 去掉右侧secondary_yaxis的spine（避免黑线）
        for spine in ax0_right.spines.values():
            spine.set_visible(False)

        # ============ RF predicted value（条状显示，非渐变）============
        ax1 = fig.add_subplot(gs_main[1, 0], sharex=ax0)
        # 使用与示例图一致的棕橙色渐变colormap
        pred_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("pred", ["#FFF5E6", "#D2691E"])
        # 将预测概率离散化显示为条状（按样本顺序直接显示，不做插值）
        ax1.imshow(p_sorted.reshape(1, -1), aspect="auto", interpolation="none", cmap=pred_cmap, vmin=0.0, vmax=1.0)
        ax1.set_yticks([])
        ax1.set_xticks([])
        for spine in ax1.spines.values():
            spine.set_visible(False)
        # 顶部添加细白色分隔线
        ax1.axhline(y=-0.5, color="white", linewidth=0.5)
        ax1_right = ax1.secondary_yaxis("right")
        ax1_right.set_yticks([0])
        ax1_right.set_yticklabels([f"{key} predicted value"], fontsize=FONTSIZE_ANNOT_LABEL)
        ax1_right.tick_params(right=False)
        for spine in ax1_right.spines.values():
            spine.set_visible(False)

        # ============ Outcome ============
        ax2 = fig.add_subplot(gs_main[2, 0], sharex=ax0)
        outcome_code = outcome_code_sorted
        outcome_cmap = matplotlib.colors.ListedColormap(["#8BC34A", "#FF9800"])  # M0绿 M1橙
        ax2.imshow(outcome_code.reshape(1, -1), aspect="auto", interpolation="none", cmap=outcome_cmap, vmin=0, vmax=1)
        ax2.set_yticks([])
        ax2.set_xticks([])
        for spine in ax2.spines.values():
            spine.set_visible(False)
        # 顶部添加细白色分隔线
        ax2.axhline(y=-0.5, color="white", linewidth=0.5)
        ax2_right = ax2.secondary_yaxis("right")
        ax2_right.set_yticks([0])
        ax2_right.set_yticklabels(["Outcome"], fontsize=FONTSIZE_ANNOT_LABEL)
        ax2_right.tick_params(right=False)
        for spine in ax2_right.spines.values():
            spine.set_visible(False)

        # ============ Group（支持3个组别：Training/Test/External）============
        ax3 = fig.add_subplot(gs_main[3, 0], sharex=ax0)
        group_code = group_code_sorted
        # 三色：Training紫、Test橙、External蓝绿
        group_cmap = matplotlib.colors.ListedColormap(["#9C27B0", "#FF9800", "#00BCD4"])
        n_groups = 3 if has_external else 2
        ax3.imshow(group_code.reshape(1, -1), aspect="auto", interpolation="none", cmap=group_cmap, vmin=0, vmax=n_groups - 1)
        ax3.set_yticks([])
        ax3.set_xticks([])
        for spine in ax3.spines.values():
            spine.set_visible(False)
        # 顶部添加细白色分隔线
        ax3.axhline(y=-0.5, color="white", linewidth=0.5)
        ax3_right = ax3.secondary_yaxis("right")
        ax3_right.set_yticks([0])
        ax3_right.set_yticklabels(["Group"], fontsize=FONTSIZE_ANNOT_LABEL)
        ax3_right.tick_params(right=False)
        for spine in ax3_right.spines.values():
            spine.set_visible(False)

        # ============ 底部Legend区（垂直colorbar网格）============
        # 添加一个空白间隔行（gs_main[4, 0]），然后是legend区
        gs_legend = gs_main[5, 0].subgridspec(legend_nrows, legend_ncol, wspace=0.18, hspace=0.40)  # 缩小间距

        # 构建legend项列表：特征 + predicted value + outcome + group
        legend_items = []
        for i in range(n_feat):
            ticks = row_ticks[i]
            # 判断类型
            if ticks[1] == ["0", "1"]:
                ltype = "binary"
            elif len(ticks[0]) <= 2:
                ltype = "continuous"
            else:
                ltype = "discrete"
            legend_items.append({
                "type": ltype,
                "label": feat_labels[i],
                "cmap": row_cm[i],
                "ticks": ticks,
                "vmin": row_min_max[i][0],
                "vmax": row_min_max[i][1],
                "base_color": base_colors[i],
            })
        # RF predicted value - 改为连续变量显示（带刻度）
        legend_items.append({
            "type": "continuous",
            "label": f"{key} predicted value",
            "cmap": pred_cmap,
            "ticks": ([0, 0.5, 1], ["0", "0.5", "1"]),
            "vmin": 0, "vmax": 1,
            "base_color": matplotlib.colors.to_rgba("#D2691E"),
        })
        # Outcome
        legend_items.append({
            "type": "categorical",
            "label": "Outcome",
            "colors": ["#8BC34A", "#FF9800"],
            "labels": ["M0", "M1"],
        })
        # Group（根据是否有外部验证集动态调整）
        if has_external:
            legend_items.append({
                "type": "categorical",
                "label": "Group",
                "colors": ["#9C27B0", "#FF9800", "#00BCD4"],
                "labels": ["Training set", "Test set", "External set"],
            })
        else:
            legend_items.append({
                "type": "categorical",
                "label": "Group",
                "colors": ["#9C27B0", "#FF9800"],
                "labels": ["Training set", "Test set"],
            })

        for idx, item in enumerate(legend_items):
            row = idx // legend_ncol
            col = idx % legend_ncol
            if row >= legend_nrows:
                break
            ax_leg = fig.add_subplot(gs_legend[row, col])

            # 统一所有图注的ylim范围，确保同一行的标题在同一水平高度
            unified_ylim = (-0.1, 1.15)
            unified_xlim = (-0.05, 1.3)
            
            # 动态计算 box_w 使色块显示为正方形
            # 公式: box_w = box_h * (ax_height / ax_width) * (xlim_range / ylim_range)
            fig.canvas.draw()  # 需要先绘制以获取准确的 position
            ax_pos = ax_leg.get_position()
            ax_width = ax_pos.width * fig.get_figwidth()   # axes 物理宽度 (inches)
            ax_height = ax_pos.height * fig.get_figheight()  # axes 物理高度 (inches)
            xlim_range = unified_xlim[1] - unified_xlim[0]  # 1.35
            ylim_range = unified_ylim[1] - unified_ylim[0]  # 1.25
            
            box_h = 0.3  # 每个色块的高度（数据坐标）
            # 计算使物理上为正方形的 box_w
            if ax_width > 0:
                box_w = box_h * (ax_height / ax_width) * (xlim_range / ylim_range)
            else:
                box_w = box_h  # fallback

            if item["type"] == "categorical":
                # 离散类别：竖着排列色块（仿照示例图），向上对齐
                n_cats = len(item["colors"])
                top_y = 1.0  # 顶部对齐位置
                for j, (color, label) in enumerate(zip(item["colors"], item["labels"])):
                    y_pos = top_y - (j + 1) * box_h  # 从顶部向下排列
                    rect = plt.Rectangle((0, y_pos), box_w, box_h * 0.9, facecolor=color, edgecolor="none")
                    ax_leg.add_patch(rect)
                    ax_leg.text(box_w + 0.06, y_pos + box_h * 0.45, label, ha="left", va="center", fontsize=FONTSIZE_LEGEND_TEXT)
                ax_leg.set_xlim(unified_xlim)
                ax_leg.set_ylim(unified_ylim)
            elif item["type"] == "binary":
                # 二值变量：竖着排列两个色块，向上对齐
                cm = item["cmap"]
                colors = [cm(0.0), cm(1.0)]
                labels = item["ticks"][1]
                top_y = 1.0  # 顶部对齐位置
                for j, (color, label) in enumerate(zip(colors, labels)):
                    y_pos = top_y - (j + 1) * box_h  # 从顶部向下排列
                    rect = plt.Rectangle((0, y_pos), box_w, box_h * 0.9, facecolor=color, edgecolor="none")
                    ax_leg.add_patch(rect)
                    ax_leg.text(box_w + 0.06, y_pos + box_h * 0.45, label, ha="left", va="center", fontsize=FONTSIZE_LEGEND_TEXT)
                ax_leg.set_xlim(unified_xlim)
                ax_leg.set_ylim(unified_ylim)
            else:
                # 连续变量：垂直colorbar，带刻度线（仿照示例图，更窄）
                gradient = np.linspace(0, 1, 256).reshape(-1, 1)
                ax_leg.imshow(gradient, aspect="auto", cmap=item["cmap"], origin="lower", extent=[0, 0.22, 0, 1])
                # 在colorbar右侧绘制刻度线和标签
                vmin, vmax = item["vmin"], item["vmax"]
                # 生成3-5个刻度
                if vmax - vmin > 0:
                    n_ticks = 4
                    tick_vals = np.linspace(vmin, vmax, n_ticks)
                    tick_positions = np.linspace(0, 1, n_ticks)
                else:
                    tick_vals = [vmin]
                    tick_positions = [0.5]
                # 绘制刻度线和标签
                for tv, tp in zip(tick_vals, tick_positions):
                    ax_leg.plot([0.24, 0.28], [tp, tp], color="0.3", linewidth=0.8)
                    ax_leg.text(0.32, tp, _fmt_tick(tv), ha="left", va="center", fontsize=FONTSIZE_COLORBAR_TICK)
                ax_leg.set_xlim(unified_xlim)
                ax_leg.set_ylim(unified_ylim)

            ax_leg.set_xticks([])
            ax_leg.set_yticks([])
            for spine in ax_leg.spines.values():
                spine.set_visible(False)
            # 标题（变量名）左对齐，用与colorbar相同的颜色
            if item["type"] == "categorical":
                title_color = item["colors"][0]
            else:
                title_color = matplotlib.colors.to_hex(item["base_color"])
            # 使用统一的ylim顶部位置放置标题，确保同一行水平对齐
            ax_leg.text(0, unified_ylim[1] + 0.05, item["label"], fontsize=FONTSIZE_LEGEND_TITLE, fontweight="bold", 
                        color=title_color, ha="left", va="bottom")

        plt.setp(ax0.get_xticklabels(), visible=False)
        out_path = out_dir / f"Variable_heatmap_{key}.png"
        fig.savefig(out_path, dpi=300, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        print(f"  -> {key} 热图已保存: {out_path}")


# ============ 主函数 ============
def main():
    # 默认运行目录
    DEFAULT_RUN_DIR = "ml_results/run_20260206_234650"
    
    parser = argparse.ArgumentParser(description="重新绘制 Variable Heatmap（直接复用原脚本函数）")
    parser.add_argument("--run_dir", default=DEFAULT_RUN_DIR, 
                        help=f"已运行的结果目录 (默认: {DEFAULT_RUN_DIR})")
    parser.add_argument("--target", default="Metastasis", help="因变量列名")
    parser.add_argument("--top_n", type=int, default=20, help="显示的特征数量（默认20，与原脚本一致）")
    parser.add_argument("--models", default=None, help="指定模型（逗号分隔，如 LR,RF,XGB），None表示全部模型")
    parser.add_argument("--output_dir", default=None, help="输出目录（默认为 run_dir/fig_new）")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"找不到结果目录: {run_dir}")

    # 输出目录
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "fig_new"
    _ensure_dir(output_dir)

    # 加载预处理器
    preprocessor_path = run_dir / "artifacts" / "preprocessor.joblib"
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"找不到预处理器: {preprocessor_path}")
    preprocessor = joblib.load(preprocessor_path)

    # 读取训练/测试集划分
    train_xlsx = run_dir / "tables" / "split_train.xlsx"
    test_xlsx = run_dir / "tables" / "split_test.xlsx"
    
    if not (train_xlsx.exists() and test_xlsx.exists()):
        raise FileNotFoundError("找不到训练/测试集划分文件")
    
    df_train = pd.read_excel(train_xlsx)
    df_test = pd.read_excel(test_xlsx)
    
    # 确定要删除的列
    cols_to_drop = [args.target] + [c for c in DEMOGRAPHIC_COLUMNS_EXCLUDE if c in df_train.columns]
    
    y_train = df_train[args.target].astype(int).to_numpy()
    y_test = df_test[args.target].astype(int).to_numpy()
    X_train_df = df_train.drop(columns=[c for c in cols_to_drop if c in df_train.columns])
    X_test_df = df_test.drop(columns=[c for c in cols_to_drop if c in df_test.columns])

    X_train = _safe_float32_matrix(preprocessor.transform(X_train_df))
    X_test = _safe_float32_matrix(preprocessor.transform(X_test_df))

    print(f"训练集: {len(y_train)} 样本")
    print(f"测试集: {len(y_test)} 样本")

    # 检查外部验证集
    external_xlsx = run_dir / "tables" / "split_external.xlsx"
    has_external = external_xlsx.exists()
    X_external = None
    y_external = None
    
    if has_external:
        df_ext = pd.read_excel(external_xlsx)
        y_external = df_ext[args.target].astype(int).to_numpy()
        X_ext_df = df_ext.drop(columns=[c for c in cols_to_drop if c in df_ext.columns])
        # 对齐列
        for col in X_train_df.columns:
            if col not in X_ext_df.columns:
                X_ext_df[col] = np.nan
        X_ext_df = X_ext_df[X_train_df.columns]  # 确保列顺序一致
        X_external = _safe_float32_matrix(preprocessor.transform(X_ext_df))
        print(f"外部验证集: {len(y_external)} 样本")

    # 查找可用模型
    model_files = list((run_dir / "artifacts").glob("model_*.joblib"))
    available_models = [p.stem.replace("model_", "") for p in model_files]
    print(f"可用模型: {available_models}")

    # 筛选要绘制的模型
    if args.models:
        selected_models = [m.strip() for m in args.models.split(",")]
        selected_models = [m for m in selected_models if m in available_models]
    else:
        selected_models = available_models

    if not selected_models:
        print("没有找到要绘制的模型")
        return

    # 加载所有选定模型和特征重要性表
    fitted_pipelines: dict[str, Any] = {}
    importance_tables: dict[str, pd.DataFrame] = {}
    
    for model_key in selected_models:
        print(f"\n[{model_key}] 加载模型和特征重要性表...")
        
        # 加载模型
        model_path = run_dir / "artifacts" / f"model_{model_key}.joblib"
        try:
            model = joblib.load(model_path)
            fitted_pipelines[model_key] = model
        except Exception as e:
            print(f"  警告: 无法加载模型 {model_path}，跳过")
            print(f"  错误信息: {e}")
            continue
        
        # 加载特征重要性表
        imp_path = run_dir / "tables" / f"feature_importance_{model_key}.csv"
        if not imp_path.exists():
            print(f"  警告: 找不到特征重要性表 {imp_path}，跳过")
            continue
        
        imp_df = pd.read_csv(imp_path)
        importance_tables[model_key] = imp_df
        print(f"  -> 已加载")

    if not fitted_pipelines:
        print("没有成功加载任何模型")
        return

    # 计算预测概率
    print("\n计算预测概率...")
    y_proba_train_by_model: dict[str, np.ndarray] = {}
    y_proba_test_by_model: dict[str, np.ndarray] = {}
    y_proba_external_by_model: dict[str, np.ndarray] = {}

    for model_key, model in fitted_pipelines.items():
        try:
            p_train = model.predict_proba(X_train)[:, 1]
            p_test = model.predict_proba(X_test)[:, 1]
            y_proba_train_by_model[model_key] = p_train
            y_proba_test_by_model[model_key] = p_test
            
            if has_external and X_external is not None:
                p_external = model.predict_proba(X_external)[:, 1]
                y_proba_external_by_model[model_key] = p_external
            
            print(f"  [{model_key}] 预测完成")
        except Exception as e:
            print(f"  [{model_key}] 预测失败: {e}")
            # 从字典中移除
            if model_key in importance_tables:
                del importance_tables[model_key]

    # 过滤掉预测失败的模型
    valid_models = set(y_proba_train_by_model.keys()) & set(importance_tables.keys())
    importance_tables = {k: v for k, v in importance_tables.items() if k in valid_models}

    if not importance_tables:
        print("没有有效的模型可以绘图")
        return

    # 拼接原始数据（与 X_train/X_test/X_external 同序），用于热图显示
    raw_dfs = [X_train_df, X_test_df]
    if has_external:
        raw_dfs.append(X_ext_df)
    raw_df_concat = pd.concat(raw_dfs, ignore_index=True)

    # 调用绘图函数
    print(f"\n开始绘制热图...")
    print(f"输出目录: {output_dir}")
    
    save_variable_heatmaps_by_model(
        out_dir=output_dir,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        y_proba_train_by_model=y_proba_train_by_model,
        y_proba_test_by_model=y_proba_test_by_model,
        importance_tables=importance_tables,
        top_n=args.top_n,
        X_external=X_external if has_external else None,
        y_external=y_external if has_external else None,
        y_proba_external_by_model=y_proba_external_by_model if has_external else None,
        raw_df=raw_df_concat,
    )

    print(f"\n完成！输出目录: {output_dir}")


if __name__ == "__main__":
    main()
