#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Independent script to redraw SHAP Waterfall plots - directly reuses original _run_shap logic.

Usage:
    1. Recalculate SHAP from the model and generate plots:
        python redraw_shap_waterfall.py --run_dir ml_results/run_20260206_234650 --model LR
        python redraw_shap_waterfall.py --run_dir ml_results/run_20260206_234650  # Redraw for all models

    2. Direct plotting from saved CSV data files (Requires executing the above command once first, fast):
        # Using default samples (defaults to 20 randomly sampled instances)
        python redraw_shap_waterfall.py --from_data ml_results/run_20260206_234650/shap_new/LR/shap_waterfall_data.csv

        # Specify multiple sample indices (comma-separated or space-separated)
        python redraw_shap_waterfall.py --from_data ml_results/run_20260206_234650/shap_new/LR/shap_waterfall_data.csv --sample_idx 0,5,10,15,20
        python redraw_shap_waterfall.py --from_data ml_results/run_20260206_234650/shap_new/LR/shap_waterfall_data.csv --sample_idx 0 1 2 3 4

        # Specify a range (e.g., 0-19 denotes sample 0 to sample 19)
        python redraw_shap_waterfall.py --from_data ml_results/run_20260206_234650/shap_new/LR/shap_waterfall_data.csv --sample_idx 0-19

        # Specify output directory
        python redraw_shap_waterfall.py --from_data ml_results/run_20260206_234650/shap_new/LR/shap_waterfall_data.csv --sample_idx 0-4 --output_dir ./output

    3. Check the number of available samples in the CSV file:
        Open shap_waterfall_data.csv, the first comment line "# n_samples: 200" indicates 200 samples are available (sample_idx: 0-199)
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

# Add current directory to path to import original modules
sys.path.insert(0, str(Path(__file__).parent))


# ============== Custom Classes (Consistent with original code, used for model deserialization) ==============

class ColumnSelector(TransformerMixin, BaseEstimator):
    """Feature column selector, consistent with the original code."""
    def __init__(self, indices: np.ndarray):
        self.indices = np.asarray(indices, dtype=int)

    def fit(self, X: Any, y: Any = None):
        return self

    def transform(self, X: Any):
        if sp.issparse(X):
            return X[:, self.indices]
        return np.asarray(X)[:, self.indices]


class ToDense(TransformerMixin, BaseEstimator):
    """Converts a sparse matrix to a dense matrix."""
    def fit(self, X: Any, y: Any = None):
        return self

    def transform(self, X: Any):
        if sp.issparse(X):
            return X.toarray()
        return np.asarray(X)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ============== Cleaning/Mapping Logic Consistent with Script 12 ==============


DEMOGRAPHIC_COLUMNS_EXCLUDE: list[str] = ["Sex"]


def normalize_categorical_series(s: pd.Series) -> pd.Series:
    """Standardizes and normalizes categorical variable values to avoid duplicate levels.

    Typical issue: The same category is written as 0 / 0.0 / ' 0 ', causing OneHotEncoder to treat them as distinct categories.
    Rules:
    - Strip leading and trailing whitespaces
    - Treat '/', empty strings, 'nan', etc., as missing values
    - For categories parsable as numeric: integers -> '0'/'1'; non-integers -> normalized using %.10g
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
    """Feature name cleaning and mapping for figure display (aligned with Script 12 style)."""
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
        # One-hot suffix might come from float (e.g., 0.0/1.0/2.0), normalize to int to match key mappings
        s = re.sub(r"(?<!\d)(-?\d+)\.0+(?!\d)", r"\1", s)

        s = DISPLAY_OVERRIDES.get(s, s)
        s = UNIT_OVERRIDES.get(s, s)
        cleaned.append(s)

    # Deduplication: Append numerical suffixes to repeated items
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
    """Core function to plot the SHAP waterfall figure."""
    import shap
    
    try:
        shap.plots.waterfall(shap_exp, max_display=max_display, show=False)
        
        # Get current figure
        fig = plt.gcf()
        fig.set_size_inches(7.8, 10.0)
        fig.patch.set_facecolor("white")

        # Default SHAP colors
        default_pos_color = "#ff0051"
        default_neg_color = "#008bfb"
        default_inner_text_color = "#ffffff"  # Inner arrow text defaults to white
        # Custom colors
        positive_color = "#F6D03D"
        negative_color = "#A52B61"
        edge_color = "#242424"  # Edge color
        positive_text_color = "#000000"  # Positive text color: black
        negative_text_color = "#ffffff"  # Negative text color: white
        
        # Modify arrow colors
        for fc in plt.gcf().get_children():
            for fcc in fc.get_children():
                if isinstance(fcc, matplotlib.patches.FancyArrow):
                    if matplotlib.colors.to_hex(fcc.get_facecolor()) == default_pos_color:
                        fcc.set_facecolor(positive_color)
                        fcc.set_edgecolor(edge_color)
                    elif matplotlib.colors.to_hex(fcc.get_facecolor()) == default_neg_color:
                        fcc.set_facecolor(negative_color)
                        fcc.set_edgecolor(edge_color)
        
        # Handle text colors
        for fc in plt.gcf().get_children():
            for fcc in fc.get_children():
                if isinstance(fcc, plt.Text):
                    text_color = matplotlib.colors.to_hex(fcc.get_color())
                    text_content = fcc.get_text().strip()
                    # Texts outside the box (originally colored)
                    if text_color == default_pos_color:
                        fcc.set_color(positive_text_color)
                    elif text_color == default_neg_color:
                        fcc.set_color(positive_text_color)
                    # Texts inside the box (originally white) - determine positive/negative by text content
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
        print(f"  Saved: {output_path}")
    except Exception as e:
        print(f"  Failed to plot waterfall: {e}")
        import traceback
        traceback.print_exc()


def load_data(run_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Loads training/testing data (prioritizes split_train/split_test exported by Script 12)."""
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

        # Note: This split is only used as a fallback (might not align with the random split during training)
        n_train = int(meta.get("n_train", int(len(df) * 0.7)))
        X_train_df = X_df.iloc[:n_train].copy()
        X_test_df = X_df.iloc[n_train:].copy()

    # Crucial: Perform categorical feature normalization before calling transform
    # (Must match training behavior to prevent 0/0.0/space-0 duplicates)
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
    """Runs SHAP analysis for a specific model - reuses the original _run_shap logic."""
    import shap

    print(f"[SHAP] {model_key} ...")

    # Load model
    model_path = run_dir / "artifacts" / f"model_{model_key}.joblib"
    if not model_path.exists():
        print(f"  Model file does not exist: {model_path}")
        return

    pipe = joblib.load(model_path)

    # Load data
    X_train, X_test, feature_names = load_data(run_dir)

    # Create output directory
    model_dir = run_dir / output_dir_name / model_key
    _ensure_dir(model_dir)

    # Extract feature indices from pipeline
    select_step = pipe.named_steps.get("select")
    if select_step is None:
        print("  No 'select' step found in pipeline, using all features")
        feat_idx = np.arange(X_train.shape[1])
    else:
        feat_idx = select_step.indices

    # Prepare data - exactly identical to original code
    rng = np.random.default_rng(random_state)
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    bg_idx = rng.choice(n_train, size=min(shap_background, n_train), replace=False)
    ex_idx = rng.choice(n_test, size=min(shap_samples, n_test), replace=False)

    # Subset chosen features and convert to dense
    X_bg = X_train[bg_idx][:, feat_idx]
    X_ex = X_test[ex_idx][:, feat_idx]
    if sp.issparse(X_bg):
        X_bg = X_bg.toarray()
    if sp.issparse(X_ex):
        X_ex = X_ex.toarray()
    X_bg = np.asarray(X_bg, dtype=np.float32)
    X_ex = np.asarray(X_ex, dtype=np.float32)

    # Synchronize pipeline post-processing (to_dense / scaler), but skip select step
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

    # Feature names
    names = feature_names if feature_names else [f"f{i}" for i in range(X_train.shape[1])]
    sel_names = [names[i] for i in feat_idx]
    sel_names_disp = _clean_display_feature_names(sel_names)

    print(f"  Calculating SHAP values (Background samples: {X_bg_t.shape[0]}, Explaining samples: {X_ex_t.shape[0]})...")
    explainer = shap.Explainer(f, X_bg_t, feature_names=sel_names)
    shap_values = explainer(X_ex_t)

    n_samples = len(shap_values)

    # Save CSV data for all samples to easily tune sample_idx later
    # Format: sample_idx, feature_name, shap_value, feature_value
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
    
    # Get base_value (can be scalar or array)
    if hasattr(shap_values.base_values, '__len__'):
        base_values_list = [float(bv) for bv in shap_values.base_values]
    else:
        base_values_list = [float(shap_values.base_values)] * n_samples
    
    csv_path = model_dir / "shap_waterfall_data.csv"
    with open(csv_path, "w", encoding="utf-8-sig") as f:
        f.write(f"# n_samples: {n_samples}\n")
        f.write(f"# n_features: {len(sel_names_disp)}\n")
        f.write(f"# base_values: {base_values_list[0]:.6f}\n")  # Usually base_value remains identical across all instances
        f.write(f"# feature_names: {','.join(sel_names_disp)}\n")
        waterfall_data.to_csv(f, index=False)
    print(f"  Saved waterfall data ({n_samples} samples): {csv_path}")

    # Parse sample indices (defaults to first 20 instances)
    sample_indices = parse_sample_indices(sample_idx_input, n_samples)
    
    # Plot waterfall figures
    print(f"  Plotting {len(sample_indices)} waterfall figures: sample_idx = {sample_indices}")
    for sidx in sample_indices:
        output_path = model_dir / f"shap_waterfall_{sidx}.png"
        sv0 = _shap_explanation_with_feature_names(shap_values[sidx], sel_names_disp)
        _draw_waterfall(sv0, output_path, max_display)


def parse_sample_indices(
    sample_idx_input: list[str] | None,
    max_samples: int,
    random_state: int = 42,
) -> list[int]:
    """Parses sample index input, supporting various formats:
    - Single digit: ['0'] -> [0]
    - Comma-separated: ['0,1,2'] -> [0, 1, 2]
    - Space-separated: ['0', '1', '2'] -> [0, 1, 2]
    - Range: ['0-9'] -> [0, 1, 2, ..., 9]
    - Mixed: ['0-4', '10', '15,20'] -> [0, 1, 2, 3, 4, 10, 15, 20]
    """
    if sample_idx_input is None:
        # Default to randomly sampling 20 instances; return all instances if total count < 20
        if max_samples <= 0:
            return [0]
        rng = np.random.default_rng(random_state)
        sampled = rng.choice(max_samples, size=min(20, max_samples), replace=False)
        return sorted(int(i) for i in sampled.tolist())
    
    indices = set()
    for item in sample_idx_input:
        # Handle comma-separated strings
        parts = item.split(",")
        for part in parts:
            part = part.strip()
            if not part:
                continue
            # Check for range format (e.g., 0-9)
            if "-" in part and not part.startswith("-"):
                try:
                    start, end = part.split("-", 1)
                    start, end = int(start.strip()), int(end.strip())
                    for i in range(start, end + 1):
                        if 0 <= i < max_samples:
                            indices.add(i)
                except ValueError:
                    print(f"  Warning: Could not parse range '{part}'")
            else:
                try:
                    idx = int(part)
                    if 0 <= idx < max_samples:
                        indices.add(idx)
                    else:
                        print(f"  Warning: sample_idx={idx} out of range (0-{max_samples-1})")
                except ValueError:
                    print(f"  Warning: Could not parse index '{part}'")
    
    return sorted(indices) if indices else [0]


def draw_waterfall_from_csv(
    data_path: Path,
    sample_indices: list[int] = None,
    max_display: int = 15,
    output_dir: Path = None,
    random_state: int = 42,
) -> None:
    """Directly plots SHAP waterfall figures from a saved CSV file (supports multiple samples)."""
    import shap

    print(f"[SHAP Waterfall] Loading from data file: {data_path}")

    # Read metadata header lines from file
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
    
    # Load data from CSV (skipping comment rows)
    df = pd.read_csv(data_path, comment="#", encoding="utf-8-sig")
    
    # Get available sample count
    available_samples = sorted(df["sample_idx"].unique())
    max_samples = len(available_samples)
    print(f"  Available samples count: {max_samples}")
    
    # If sample indices are omitted, use default (randomly sample 20)
    if sample_indices is None:
        rng = np.random.default_rng(random_state)
        sample_indices = sorted(
            int(i) for i in rng.choice(available_samples, size=min(20, max_samples), replace=False).tolist()
        )
    
    # Filter valid sample indices
    valid_indices = [idx for idx in sample_indices if idx in available_samples]
    if not valid_indices:
        print(f"  Error: No valid sample indices found")
        return
    
    print(f"  Will plot {len(valid_indices)} figures: sample_idx = {valid_indices}")
    
    # Determine output directory
    save_dir = output_dir if output_dir is not None else data_path.parent
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot waterfall figure for each instance
    for sample_idx in valid_indices:
        # Filter data for specific sample index
        sample_df = df[df["sample_idx"] == sample_idx]
        shap_values_arr = sample_df["shap_value"].values
        
        print(f"  Plotting sample {sample_idx}: base_value={base_value:.4f}, f(x)={base_value + shap_values_arr.sum():.4f}")

        # Reconstruct SHAP Explanation object
        # For legacy CSV files: If feature_name is still structured like 'Carcinoma nodule 2', remap it here
        disp_names = _clean_display_feature_names(sample_df["feature_name"].astype(str).tolist())
        shap_exp = shap.Explanation(
            values=shap_values_arr,
            base_values=base_value,
            data=sample_df["feature_value"].values,
            feature_names=disp_names,
        )

        # Plot
        output_path = save_dir / f"shap_waterfall_{sample_idx}.png"
        _draw_waterfall(shap_exp, output_path, max_display)


def main():
    # Default execution directory
    DEFAULT_RUN_DIR = "ml_results/run_20260206_234650"
    
    parser = argparse.ArgumentParser(description="Redraw SHAP Waterfall plots - reuses original _run_shap logic")
    parser.add_argument("--run_dir", type=str, default=DEFAULT_RUN_DIR, 
                        help=f"Path to run results directory (Default: {DEFAULT_RUN_DIR})")
    parser.add_argument("--model", type=str, default=None, help="Model name (e.g., LR, RF, XGB). Redraws for all models if omitted")
    parser.add_argument("--sample_idx", type=str, nargs="*", default=None,
                        help="Sample indices to explain. Supports multiple formats: single index (0), comma-separated (0,1,2), range (0-9), mixed (0-4,10,15). Defaults to outputting 20 randomly sampled items")
    parser.add_argument("--max_display", type=int, default=15, help="Maximum number of features displayed (Default: 15)")
    parser.add_argument("--shap_background", type=int, default=100, help="SHAP background sample count (Default: 100)")
    parser.add_argument("--shap_samples", type=int, default=200, help="SHAP evaluation sample count (Default: 200)")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed state (Default: 42)")
    parser.add_argument("--output_dir", type=str, default="shap_new", help="Output folder name (Default: shap_new)")
    parser.add_argument("--from_data", type=str, default=None, help="Directly plot from an already saved CSV data file")

    args = parser.parse_args()

    # Mode 1: Direct plotting from a saved data file
    if args.from_data:
        data_path = Path(args.from_data)
        if not data_path.exists():
            print(f"Error: Data file does not exist {data_path}")
            return
        
        # Read CSV first to extract total sample count
        n_samples_total = 200  # Fallback default value
        with open(data_path, "r", encoding="utf-8-sig") as f:
            for line in f:
                if line.startswith("# n_samples"):
                    n_samples_total = int(line.split(":")[-1].strip())
                    break
        
        # Parse sample indices
        sample_indices = parse_sample_indices(args.sample_idx, n_samples_total, random_state=args.random_state)
        
        out_dir = Path(args.output_dir) if args.output_dir and args.output_dir != "shap_new" else None
        draw_waterfall_from_csv(
            data_path=data_path,
            sample_indices=sample_indices,
            max_display=args.max_display,
            output_dir=out_dir,
            random_state=args.random_state,
        )
        print("\nDone!")
        return

    # Mode 2: Recompute SHAP values and generate figures (uses default or explicit run_dir)
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: Directory does not exist {run_dir}")
        return

    # Retrieve list of available models
    artifacts_dir = run_dir / "artifacts"
    available_models = []
    for f in artifacts_dir.glob("model_*.joblib"):
        model_name = f.stem.replace("model_", "")
        available_models.append(model_name)

    if not available_models:
        print(f"Error: No model files found inside {artifacts_dir}")
        return

    print(f"Available models: {available_models}")

    # Determine targeted models to process
    if args.model:
        if args.model not in available_models:
            print(f"Error: Model '{args.model}' does not exist. Available models: {available_models}")
            return
        models_to_process = [args.model]
    else:
        models_to_process = available_models

    # Process each selected model
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
            print(f"  Error encountered processing model {model_key}: {e}")
            import traceback
            traceback.print_exc()

    print("\nDone!")


if __name__ == "__main__":
    main()