#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
重绘 SHAP Waterfall 图的独立脚本 - 直接复用原始 _run_shap 逻辑。

使用方法:
    1. 从模型重新计算 SHAP 并绑图:
        python redraw_shap_waterfall.py --run_dir ml_results/run_20260206_234650 --model LR
        python redraw_shap_waterfall.py --run_dir ml_results/run_20260206_234650  # 重绘所有模型

    2. 从已保存的 CSV 数据文件直接绑图 (需要先执行上述代码一次后，速度快):
        # 使用默认样本 (默认随机抽样 20 个样本)
        python redraw_shap_waterfall.py --from_data ml_results/run_20260206_234650/shap_new/LR/shap_waterfall_data.csv

        # 指定多个样本索引 (逗号分隔或空格分隔)
        python redraw_shap_waterfall.py --from_data ml_results/run_20260206_234650/shap_new/LR/shap_waterfall_data.csv --sample_idx 0,5,10,15,20
        python redraw_shap_waterfall.py --from_data ml_results/run_20260206_234650/shap_new/LR/shap_waterfall_data.csv --sample_idx 0 1 2 3 4

        # 指定范围 (如 0-19 表示第0到19个样本)
        python redraw_shap_waterfall.py --from_data ml_results/run_20260206_234650/shap_new/LR/shap_waterfall_data.csv --sample_idx 0-19

        # 指定输出目录
        python redraw_shap_waterfall.py --from_data ml_results/run_20260206_234650/shap_new/LR/shap_waterfall_data.csv --sample_idx 0-4 --output_dir ./output

    3. 查看 CSV 文件中可用的样本数量:
        打开 shap_waterfall_data.csv，第一行注释 "# n_samples: 200" 表示有 200 个样本可选 (sample_idx: 0-199)
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin

matplotlib.use("Agg")

# 将当前目录添加到路径，以便导入原始模块
sys.path.insert(0, str(Path(__file__).parent))


# ============== 自定义类（与原代码一致，用于反序列化模型） ==============

class ColumnSelector(TransformerMixin, BaseEstimator):
    """特征列选择器，与原代码保持一致。"""
    def __init__(self, indices: np.ndarray):
        self.indices = np.asarray(indices, dtype=int)

    def fit(self, X: Any, y: Any = None):
        return self

    def transform(self, X: Any):
        if sp.issparse(X):
            return X[:, self.indices]
        return np.asarray(X)[:, self.indices]


class ToDense(TransformerMixin, BaseEstimator):
    """将稀疏矩阵转换为稠密矩阵。"""
    def fit(self, X: Any, y: Any = None):
        return self

    def transform(self, X: Any):
        if sp.issparse(X):
            return X.toarray()
        return np.asarray(X)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ============== 与 12 脚本一致的清洗/映射逻辑 ==============


DEMOGRAPHIC_COLUMNS_EXCLUDE: list[str] = ["Sex"]


def normalize_categorical_series(s: pd.Series) -> pd.Series:
    """统一规范化分类变量的取值，避免重复水平。

    典型问题：同一类别被写成 0 / 0.0 / ' 0 '，导致 OneHotEncoder 视为不同类别。
    规则：
    - 去首尾空白
    - 将 '/', 空串, 'nan' 等视为缺失
    - 对可解析为数值的类别：整数 -> '0'/'1'；非整数 -> 使用 %.10g 规范化
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


def normalize_categorical_columns(df: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    cols = [c for c in categorical_cols if c in df.columns]
    if not cols:
        return df
    df2 = df.copy()
    for c in cols:
        df2[c] = normalize_categorical_series(df2[c])
    return df2


def _clean_display_feature_names(raw_names: list[str]) -> list[str]:
    """用于图中展示的特征名清理与映射（对齐 12 脚本风格）。"""
    DISPLAY_OVERRIDES: dict[str, str] = {
        "T stage T3-T4": "T stage(T3-T4)",
        "T stage T1-T2": "T stage(T1-T2)",
        "N stage 0": "N stage(N0)",
        "N stage 1": "N stage(N1)",
        "N stage 2": "N stage(N2)",
        "Differentiation grade G1": "Differentiation grade(G1)",
        "Differentiation grade G2": "Differentiation grade(G2)",
        "Differentiation grade G3": "Differentiation grade(G3)",
        "Vascular invasion 0": "Vascular invasion(No)",
        "Vascular invasion 1": "Vascular invasion(Yes)",
        "Perineural invasion 0": "Perineural invasion(No)",
        "Perineural invasion 1": "Perineural invasion(Yes)",
        "Carcinoma nodule 0": "Carcinoma nodule(None)",
        "Carcinoma nodule 1": "Carcinoma nodule(1-3)",
        "Carcinoma nodule 2": "Carcinoma nodule(≥4)",
        "MLH1 1": "MLH1(+)",
        "MLH1 0": "MLH1(-)",
        "MSH2 1": "MSH2(+)",
        "MSH2 0": "MSH2(-)",
        "PMS2 0": "PMS2(-)",
        "PMS2 1": "PMS2(+)",
        "MSH6 1": "MSH6(+)",
        "MSH6 0": "MSH6(-)",
        "Colonic obstruction 0": "Colonic obstruction(No)",
        "Colonic obstruction 1": "Colonic obstruction(Yes)",
        "KRAS mutant 0": "KRAS mutant(No)",
        "KRAS mutant 1": "KRAS mutant(Yes)",
        "BRAF mutant 0": "BRAF mutant(No)",
        "BRAF mutant 1": "BRAF mutant(Yes)",
        "NRAS mutant 0": "NRAS mutant(No)",
        "NRAS mutant 1": "NRAS mutant(Yes)",
        "MSI-H 0": "MSS",
        "MSI-H 1": "MSI-H",
        "mGPS 0": "mGPS: 0",
        "mGPS 1": "mGPS: 1",
        "mGPS 2": "mGPS: 2",
        "Diabetes 1": "Diabetes(Yes)",
        "Diabetes 0": "Diabetes(No)",
        "Hypertension 1": "Hypertension(Yes)",
        "Hypertension 0": "Hypertension(No)",
        "Family history 0": "Family history(No)",
        "Family history 1": "Family history(Yes)",
    }

    UNIT_OVERRIDES: dict[str, str] = {
        "CA199": "CA199(U/mL)",
        "CEA": "CEA(ng/ml)",
        "CA242": "CA242(U/mL)",
        "CA724": "CA724(IU/mL)",
        "Albumin": "Albumin(g/L)",
        "LDH": "LDH(U/L)",
        "Prealbumin": "Prealbumin(mg/L)",
        "Total protein": "Total protein(g/L)",
        "Age": "Age(years)",
        "BMI": "BMI(kg/m²)",
        "Tumor size": "Tumor size(cm)",
        "Tumor volume": "Tumor volume(cm^3)",
        "Ki67": "Ki67(%)",
    }

    cleaned: list[str] = []
    for n in raw_names:
        s = str(n)
        if s.startswith("num__"):
            s = s[5:]
        elif s.startswith("cat__"):
            s = s[5:]
        elif s.startswith("num_"):
            s = s[4:]
        elif s.startswith("cat_"):
            s = s[4:]

        s = s.replace("_", " ")
        s = " ".join(s.split())
        # one-hot 后缀可能来自 float（如 0.0/1.0/2.0），归一化成整数以命中映射
        s = re.sub(r"(?<!\d)(-?\d+)\.0+(?!\d)", r"\1", s)

        s = DISPLAY_OVERRIDES.get(s, s)
        s = UNIT_OVERRIDES.get(s, s)
        cleaned.append(s)

    # 去重：重复项追加编号后缀
    seen: dict[str, int] = {}
    final: list[str] = []
    for s in cleaned:
        cnt = seen.get(s, 0) + 1
        seen[s] = cnt
        final.append(s if cnt == 1 else f"{s} ({cnt})")
    return final


def _shap_explanation_with_feature_names(expl: Any, feature_names: list[str]) -> Any:
    """Return a SHAP Explanation with replaced feature_names."""
    import shap
    return shap.Explanation(
        values=expl.values,
        base_values=expl.base_values,
        data=getattr(expl, "data", None),
        feature_names=list(feature_names),
    )

def _draw_waterfall(
    shap_exp: Any,
    output_path: Path,
    max_display: int = 15,
) -> None:
    """绘制 SHAP waterfall 图的核心函数。"""
    import shap
    
    try:
        shap.plots.waterfall(shap_exp, max_display=max_display, show=False)
        
        # 获取当前 figure
        fig = plt.gcf()
        fig.set_size_inches(7.8, 10.0)
        fig.patch.set_facecolor("white")

        # Default SHAP colors
        default_pos_color = "#ff0051"
        default_neg_color = "#008bfb"
        default_inner_text_color = "#ffffff"  # 箭头内部文字默认白色
        # Custom colors
        positive_color = "#F6D03D"
        negative_color = "#A52B61"
        edge_color = "#242424"  # 边框颜色
        positive_text_color = "#000000"  # 正向文字颜色：黑色
        negative_text_color = "#ffffff"  # 负向文字颜色：白色
        
        # 修改箭头颜色
        for fc in plt.gcf().get_children():
            for fcc in fc.get_children():
                if isinstance(fcc, matplotlib.patches.FancyArrow):
                    if matplotlib.colors.to_hex(fcc.get_facecolor()) == default_pos_color:
                        fcc.set_facecolor(positive_color)
                        fcc.set_edgecolor(edge_color)
                    elif matplotlib.colors.to_hex(fcc.get_facecolor()) == default_neg_color:
                        fcc.set_facecolor(negative_color)
                        fcc.set_edgecolor(edge_color)
        
        # 处理文字颜色
        for fc in plt.gcf().get_children():
            for fcc in fc.get_children():
                if isinstance(fcc, plt.Text):
                    text_color = matplotlib.colors.to_hex(fcc.get_color())
                    text_content = fcc.get_text().strip()
                    # 框外文字（原本带颜色的）
                    if text_color == default_pos_color:
                        fcc.set_color(positive_text_color)
                    elif text_color == default_neg_color:
                        fcc.set_color(positive_text_color)
                    # 框内文字（原本是白色的）- 根据文字内容判断正负
                    elif text_color == default_inner_text_color:
                        try:
                            val = float(text_content.replace("+", "").replace(" ", ""))
                            if val >= 0:
                                fcc.set_color(positive_text_color)
                            else:
                                fcc.set_color(negative_text_color)
                        except ValueError:
                            pass
        
        ax2 = plt.gca()
        ax2.set_facecolor("white")
        ax2.tick_params(axis="y", labelsize=7)
        ax2.tick_params(axis="x", labelsize=8)
        ax2.set_xlabel("SHAP value", fontsize=8)
        # fig.subplots_adjust(left=0.45, right=0.95, top=0.94, bottom=0.06)
        fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
        print(f"  已保存: {output_path}")
    except Exception as e:
        print(f"  绘制 waterfall 失败: {e}")
        import traceback
        traceback.print_exc()


def load_data(run_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """加载训练/测试数据（优先使用 12 脚本导出的 split_train/split_test）。"""
    meta_path = run_dir / "run_meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    target_col = meta.get("target", "Metastasis")
    categorical_cols = list(meta.get("categorical_cols_used", []) or [])

    preprocessor = joblib.load(run_dir / "artifacts" / "preprocessor.joblib")

    split_train = run_dir / "tables" / "split_train.xlsx"
    split_test = run_dir / "tables" / "split_test.xlsx"
    if split_train.exists() and split_test.exists():
        df_train = pd.read_excel(split_train)
        df_test = pd.read_excel(split_test)
        drop_cols = [c for c in [target_col, *DEMOGRAPHIC_COLUMNS_EXCLUDE] if c in df_train.columns]
        X_train_df = df_train.drop(columns=drop_cols)
        drop_cols_te = [c for c in [target_col, *DEMOGRAPHIC_COLUMNS_EXCLUDE] if c in df_test.columns]
        X_test_df = df_test.drop(columns=drop_cols_te)
    else:
        input_path = Path(meta["input"])
        df = pd.read_excel(input_path)
        drop_cols = [c for c in [target_col, *DEMOGRAPHIC_COLUMNS_EXCLUDE] if c in df.columns]
        X_df = df.drop(columns=drop_cols)

        # 注意：这里的切分仅用于兜底（可能与训练时的随机切分不一致）
        n_train = int(meta.get("n_train", int(len(df) * 0.7)))
        X_train_df = X_df.iloc[:n_train].copy()
        X_test_df = X_df.iloc[n_train:].copy()

    # 关键：在 transform 前做类别值规范化（需与训练时一致，避免 0/0.0/空格0 重复水平）
    if categorical_cols:
        X_train_df = normalize_categorical_columns(X_train_df, categorical_cols)
        X_test_df = normalize_categorical_columns(X_test_df, categorical_cols)

    X_train = preprocessor.transform(X_train_df)
    X_test = preprocessor.transform(X_test_df)

    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names = [f"f{i}" for i in range(X_train.shape[1])]

    return X_train, X_test, feature_names


def run_shap_for_model(
    run_dir: Path,
    model_key: str,
    output_dir_name: str = "shap_new",
    sample_idx_input: list[str] | None = None,
    max_display: int = 15,
    shap_background: int = 100,
    shap_samples: int = 200,
    random_state: int = 42,
) -> None:
    """为指定模型运行 SHAP 分析 - 复用原始 _run_shap 的逻辑。"""
    import shap

    print(f"[SHAP] {model_key} ...")

    # 加载模型
    model_path = run_dir / "artifacts" / f"model_{model_key}.joblib"
    if not model_path.exists():
        print(f"  模型文件不存在: {model_path}")
        return

    pipe = joblib.load(model_path)

    # 加载数据
    X_train, X_test, feature_names = load_data(run_dir)

    # 创建输出目录
    model_dir = run_dir / output_dir_name / model_key
    _ensure_dir(model_dir)

    # 从 pipeline 中提取特征索引
    select_step = pipe.named_steps.get("select")
    if select_step is None:
        print("  Pipeline 中没有 select 步骤，使用所有特征")
        feat_idx = np.arange(X_train.shape[1])
    else:
        feat_idx = select_step.indices

    # 准备数据 - 与原始代码完全一致
    rng = np.random.default_rng(random_state)
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    bg_idx = rng.choice(n_train, size=min(shap_background, n_train), replace=False)
    ex_idx = rng.choice(n_test, size=min(shap_samples, n_test), replace=False)

    # 取选择后的特征并转成 dense
    X_bg = X_train[bg_idx][:, feat_idx]
    X_ex = X_test[ex_idx][:, feat_idx]
    if sp.issparse(X_bg):
        X_bg = X_bg.toarray()
    if sp.issparse(X_ex):
        X_ex = X_ex.toarray()
    X_bg = np.asarray(X_bg, dtype=np.float32)
    X_ex = np.asarray(X_ex, dtype=np.float32)

    # 同步 pipeline 的后处理（to_dense / scaler），但跳过 select
    steps = list(pipe.named_steps.items())
    steps_wo_select = [(n, s) for n, s in steps if n != "select"]

    def _transform_no_select(Xa: np.ndarray) -> np.ndarray:
        Xo: Any = Xa
        for name, step in steps_wo_select:
            if name == "model":
                break
            Xo = step.transform(Xo)
        return np.asarray(Xo)

    X_bg_t = _transform_no_select(X_bg)
    X_ex_t = _transform_no_select(X_ex)

    model = pipe.named_steps["model"]

    def f(Xa: np.ndarray) -> np.ndarray:
        return model.predict_proba(Xa)[:, 1]

    # feature names
    names = feature_names if feature_names else [f"f{i}" for i in range(X_train.shape[1])]
    sel_names = [names[i] for i in feat_idx]
    sel_names_disp = _clean_display_feature_names(sel_names)

    print(f"  计算 SHAP 值 (背景样本: {X_bg_t.shape[0]}, 解释样本: {X_ex_t.shape[0]})...")
    explainer = shap.Explainer(f, X_bg_t, feature_names=sel_names)
    shap_values = explainer(X_ex_t)

    n_samples = len(shap_values)

    # 保存所有样本的 CSV 数据，方便后续调整 sample_idx
    # 格式: sample_idx, feature_name, shap_value, feature_value
    all_rows = []
    for sidx in range(n_samples):
        for fidx, fname in enumerate(sel_names_disp):
            all_rows.append({
                "sample_idx": sidx,
                "feature_name": fname,
                "shap_value": shap_values.values[sidx, fidx],
                "feature_value": X_ex_t[sidx, fidx],
            })
    waterfall_data = pd.DataFrame(all_rows)
    
    # 获取 base_value（可能是标量或数组）
    if hasattr(shap_values.base_values, '__len__'):
        base_values_list = [float(bv) for bv in shap_values.base_values]
    else:
        base_values_list = [float(shap_values.base_values)] * n_samples
    
    csv_path = model_dir / "shap_waterfall_data.csv"
    with open(csv_path, "w", encoding="utf-8-sig") as f:
        f.write(f"# n_samples: {n_samples}\n")
        f.write(f"# n_features: {len(sel_names_disp)}\n")
        f.write(f"# base_values: {base_values_list[0]:.6f}\n")  # 通常所有样本的 base_value 相同
        f.write(f"# feature_names: {','.join(sel_names_disp)}\n")
        waterfall_data.to_csv(f, index=False)
    print(f"  已保存 waterfall 数据 ({n_samples} 个样本): {csv_path}")

    # 解析样本索引（默认前20个）
    sample_indices = parse_sample_indices(sample_idx_input, n_samples)
    
    # 绘制 waterfall 图
    print(f"  绘制 {len(sample_indices)} 张 waterfall 图: sample_idx = {sample_indices}")
    for sidx in sample_indices:
        output_path = model_dir / f"shap_waterfall_{sidx}.png"
        sv0 = _shap_explanation_with_feature_names(shap_values[sidx], sel_names_disp)
        _draw_waterfall(sv0, output_path, max_display)


def parse_sample_indices(
    sample_idx_input: list[str] | None,
    max_samples: int,
    random_state: int = 42,
) -> list[int]:
    """解析样本索引输入，支持多种格式：
    - 单个数字: ['0'] -> [0]
    - 逗号分隔: ['0,1,2'] -> [0, 1, 2]
    - 空格分隔: ['0', '1', '2'] -> [0, 1, 2]
    - 范围: ['0-9'] -> [0, 1, 2, ..., 9]
    - 混合: ['0-4', '10', '15,20'] -> [0, 1, 2, 3, 4, 10, 15, 20]
    """
    if sample_idx_input is None:
        # 默认随机抽样 20 个样本；样本不足 20 时全部返回
        if max_samples <= 0:
            return [0]
        rng = np.random.default_rng(random_state)
        sampled = rng.choice(max_samples, size=min(20, max_samples), replace=False)
        return sorted(int(i) for i in sampled.tolist())
    
    indices = set()
    for item in sample_idx_input:
        # 处理逗号分隔
        parts = item.split(",")
        for part in parts:
            part = part.strip()
            if not part:
                continue
            # 检查是否是范围格式 (如 0-9)
            if "-" in part and not part.startswith("-"):
                try:
                    start, end = part.split("-", 1)
                    start, end = int(start.strip()), int(end.strip())
                    for i in range(start, end + 1):
                        if 0 <= i < max_samples:
                            indices.add(i)
                except ValueError:
                    print(f"  警告: 无法解析范围 '{part}'")
            else:
                try:
                    idx = int(part)
                    if 0 <= idx < max_samples:
                        indices.add(idx)
                    else:
                        print(f"  警告: sample_idx={idx} 超出范围 (0-{max_samples-1})")
                except ValueError:
                    print(f"  警告: 无法解析索引 '{part}'")
    
    return sorted(indices) if indices else [0]


def draw_waterfall_from_csv(
    data_path: Path,
    sample_indices: list[int] = None,
    max_display: int = 15,
    output_dir: Path = None,
    random_state: int = 42,
) -> None:
    """从已保存的 CSV 数据文件直接绘制 SHAP waterfall 图（支持多个样本）。"""
    import shap

    print(f"[SHAP Waterfall] 从数据文件加载: {data_path}")

    # 读取文件头部的元信息
    base_value = None
    n_samples_total = None
    with open(data_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            if line.startswith("# base_values"):
                base_value = float(line.split(":")[-1].strip())
            elif line.startswith("# n_samples"):
                n_samples_total = int(line.split(":")[-1].strip())
            elif not line.startswith("#"):
                break
    
    # 从 CSV 加载数据（跳过注释行）
    df = pd.read_csv(data_path, comment="#", encoding="utf-8-sig")
    
    # 获取可用样本数
    available_samples = sorted(df["sample_idx"].unique())
    max_samples = len(available_samples)
    print(f"  可用样本数: {max_samples}")
    
    # 如果未指定样本索引，使用默认（随机抽样 20 个）
    if sample_indices is None:
        rng = np.random.default_rng(random_state)
        sample_indices = sorted(
            int(i) for i in rng.choice(available_samples, size=min(20, max_samples), replace=False).tolist()
        )
    
    # 过滤有效的样本索引
    valid_indices = [idx for idx in sample_indices if idx in available_samples]
    if not valid_indices:
        print(f"  错误: 没有有效的样本索引")
        return
    
    print(f"  将绘制 {len(valid_indices)} 张图: sample_idx = {valid_indices}")
    
    # 确定输出目录
    save_dir = output_dir if output_dir is not None else data_path.parent
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 为每个样本绘制 waterfall 图
    for sample_idx in valid_indices:
        # 筛选指定样本的数据
        sample_df = df[df["sample_idx"] == sample_idx]
        shap_values_arr = sample_df["shap_value"].values
        
        print(f"  绘制样本 {sample_idx}: base_value={base_value:.4f}, f(x)={base_value + shap_values_arr.sum():.4f}")

        # 构建 SHAP Explanation 对象
        # 对历史 CSV：如 feature_name 仍是 'Carcinoma nodule 2' 形式，这里也做一次映射
        disp_names = _clean_display_feature_names(sample_df["feature_name"].astype(str).tolist())
        shap_exp = shap.Explanation(
            values=shap_values_arr,
            base_values=base_value,
            data=sample_df["feature_value"].values,
            feature_names=disp_names,
        )

        # 绘图
        output_path = save_dir / f"shap_waterfall_{sample_idx}.png"
        _draw_waterfall(shap_exp, output_path, max_display)


def main():
    # 默认运行目录
    DEFAULT_RUN_DIR = "ml_results/run_20260206_234650"
    
    parser = argparse.ArgumentParser(description="重绘 SHAP Waterfall 图 - 复用原始 _run_shap 逻辑")
    parser.add_argument("--run_dir", type=str, default=DEFAULT_RUN_DIR, 
                        help=f"运行结果目录路径 (默认: {DEFAULT_RUN_DIR})")
    parser.add_argument("--model", type=str, default=None, help="模型名称 (如 LR, RF, XGB)，不指定则重绘所有模型")
    parser.add_argument("--sample_idx", type=str, nargs="*", default=None,
                        help="要解释的样本索引，支持多种格式: 单个(0)、逗号分隔(0,1,2)、范围(0-9)、混合(0-4,10,15)。默认随机输出20个样本")
    parser.add_argument("--max_display", type=int, default=15, help="显示的最大特征数 (默认: 15)")
    parser.add_argument("--shap_background", type=int, default=100, help="SHAP 背景样本数 (默认: 100)")
    parser.add_argument("--shap_samples", type=int, default=200, help="SHAP 解释样本数 (默认: 200)")
    parser.add_argument("--random_state", type=int, default=42, help="随机种子 (默认: 42)")
    parser.add_argument("--output_dir", type=str, default="shap_new", help="输出目录名称 (默认: shap_new)")
    parser.add_argument("--from_data", type=str, default=None, help="从已保存的 CSV 数据文件直接绘图")

    args = parser.parse_args()

    # 模式1: 从保存的数据文件直接绘图
    if args.from_data:
        data_path = Path(args.from_data)
        if not data_path.exists():
            print(f"错误: 数据文件不存在 {data_path}")
            return
        
        # 先读取 CSV 获取样本总数
        n_samples_total = 200  # 默认值
        with open(data_path, "r", encoding="utf-8-sig") as f:
            for line in f:
                if line.startswith("# n_samples"):
                    n_samples_total = int(line.split(":")[-1].strip())
                    break
        
        # 解析样本索引
        sample_indices = parse_sample_indices(args.sample_idx, n_samples_total, random_state=args.random_state)
        
        out_dir = Path(args.output_dir) if args.output_dir and args.output_dir != "shap_new" else None
        draw_waterfall_from_csv(
            data_path=data_path,
            sample_indices=sample_indices,
            max_display=args.max_display,
            output_dir=out_dir,
            random_state=args.random_state,
        )
        print("\n完成!")
        return

    # 模式2: 重新计算 SHAP 并绘图（使用默认或指定的 run_dir）
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"错误: 目录不存在 {run_dir}")
        return

    # 获取可用的模型列表
    artifacts_dir = run_dir / "artifacts"
    available_models = []
    for f in artifacts_dir.glob("model_*.joblib"):
        model_name = f.stem.replace("model_", "")
        available_models.append(model_name)

    if not available_models:
        print(f"错误: 在 {artifacts_dir} 中没有找到模型文件")
        return

    print(f"可用模型: {available_models}")

    # 确定要处理的模型
    if args.model:
        if args.model not in available_models:
            print(f"错误: 模型 '{args.model}' 不存在。可用模型: {available_models}")
            return
        models_to_process = [args.model]
    else:
        models_to_process = available_models

    # 处理每个模型
    for model_key in models_to_process:
        try:
            run_shap_for_model(
                run_dir=run_dir,
                model_key=model_key,
                output_dir_name=args.output_dir,
                sample_idx_input=args.sample_idx,
                max_display=args.max_display,
                shap_background=args.shap_background,
                shap_samples=args.shap_samples,
                random_state=args.random_state,
            )
        except Exception as e:
            print(f"  处理模型 {model_key} 时出错: {e}")
            import traceback
            traceback.print_exc()

    print("\n完成!")


if __name__ == "__main__":
    main()
