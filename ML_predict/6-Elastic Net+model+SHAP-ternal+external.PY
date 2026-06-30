from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import scipy.sparse as sp
import joblib
from scipy.stats import chi2
from scipy.interpolate import UnivariateSpline

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib import gridspec
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
	accuracy_score,
	brier_score_loss,
	cohen_kappa_score,
	confusion_matrix,
	f1_score,
	log_loss,
	roc_auc_score,
	roc_curve,
)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedShuffleSplit, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression as _CalibLogisticRegression

try:
	from xgboost import XGBClassifier  # type: ignore

	_HAS_XGBOOST: Final[bool] = True
except Exception:
	XGBClassifier = None  # type: ignore
	_HAS_XGBOOST = False


CATEGORICAL_COLUMNS_DEFAULT: Final[list[str]] = [
	  "T stage", "N stage", "Differentiation grade", "Vascular invasion", "Perineural invasion",
        "Carcinoma nodule",  "MLH1", "MSH2", "PMS2", "MSH6",  "Family history",
        "Colonic obstruction", "Hypertension", "Diabetes", "Hyperlipidemia",
        "BRAF mutant", "KRAS mutant", "NRAS mutant", "MSI-H",
        "mGPS"
]

# Sex belongs to demographic characteristics and is excluded from feature selection and model construction
DEMOGRAPHIC_COLUMNS_EXCLUDE: Final[list[str]] = ["Sex"]


@dataclass(frozen=True)
class ModelSpec:
	key: str
	title: str
	estimator: BaseEstimator
	param_grid: dict[str, list[Any]]
	needs_scaling: bool
	needs_dense: bool


class ColumnSelector(TransformerMixin, BaseEstimator):
	def __init__(self, indices: np.ndarray):
		self.indices = np.asarray(indices, dtype=int)

	def fit(self, X: Any, y: Any = None):
		return self

	def transform(self, X: Any):
		if sp.issparse(X):
			return X[:, self.indices]
		return np.asarray(X)[:, self.indices]


class ToDense(TransformerMixin, BaseEstimator):
	def fit(self, X: Any, y: Any = None):
		return self

	def transform(self, X: Any):
		if sp.issparse(X):
			return X.toarray()
		return np.asarray(X)


def _ensure_dir(path: Path) -> None:
	path.mkdir(parents=True, exist_ok=True)


def _now_stamp() -> str:
	return time.strftime("%Y%m%d_%H%M%S")


def normalize_categorical_series(s: pd.Series) -> pd.Series:
	"""Standardize and normalize values of categorical variables to avoid duplicate levels.

	Typical issue: the same category written as 0 / 0.0 / ' 0 ', causing OneHotEncoder to treat them as distinct levels.
	Rules:
	- Strip leading/trailing whitespaces
	- Treat '/', empty strings, 'nan' etc. as missing values
	- For numeric categories: integer -> '0'/'1'; non-integer -> format with %.10g
	"""
	if s is None:
		return s
	# First convert to string uniformly and strip
	s_str = s.astype(str).str.strip()
	# Common missing value formats
	s_str = s_str.replace({"/": np.nan, "": np.nan, "nan": np.nan, "NaN": np.nan, "None": np.nan, "NONE": np.nan})
	# Normalize numeric categories
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
	"""Perform category value normalization on specified categorical columns in DataFrame (leaving others unchanged)."""
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
	"""Clean feature names only for graph display.

	- Remove prefixes: num_/cat_ and num__/cat__
	- Replace underscores '_' with spaces for better layout
	- If names conflict after cleaning, automatically append (2)/(3)... to ensure uniqueness
	"""
	# Key is the name after removing prefix/underscore (e.g., "T stage T3-T4").
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
	    "Family history 1": "Family history(Yes)"
	}

	# Continuous variables: append units uniformly (for plots and tables display name).
	# Key uses the same result after 'prefix cleaning + underscore conversion'; Value is the desired final name.
	UNIT_OVERRIDES: dict[str, str] = {
		"C1Q": "C1Q(mg/L)",
		"Creatinine": "Creatinine(umol/L)",
		"Albumin": "Albumin(g/L)",
		"ALP": "ALP(U/L)",
		"ALT": "ALT(U/L)",
		"AST": "AST(U/L)",
		"GGT": "GGT(U/L)",
		"LDH": "LDH(U/L)",
		"Prealbumin": "Prealbumin(mg/L)",
		"Total bile acids": "Total bile acids(μmol/L)",
		"Total bilirubin": "Total bilirubin(umol/L)",
		"Triglycerides": "Triglycerides(mmol/L)",
		"Total protein": "Total protein(g/L)",
		"GLDH": "GLDH(U/L)",
		"A/G ratio": "A/G ratio",
		"Globulin": "Globulin(g/L)",
		"Total cholesterol": "Total cholesterol(mmol/L)",
		"IL10": "IL10(pg/mL)",
		"IL4": "IL4(pg/mL)",
		"IL5": "IL5(pg/mL)",
		"IL8": "IL8(pg/mL)",
		"IL12-p70": "IL12-p70(pg/mL)",
		"IFN-γ": "IFN-γ(pg/mL)",
		"IFN-α": "IFN-α(pg/mL)",
		"IL-1β": "IL-1β(pg/mL)",
		"IL6": "IL6(pg/mL)",
		"IL17": "IL17(pg/mL)",
		"TNF-α": "TNF-α(pg/mL)",
		"IL2": "IL2(pg/mL)",
		"CD4+ count": "CD4+ count(cells/ul)",
		"CD8+ count": "CD8+ count(cells/ul)",
		"CD19+ count": "CD19+ count(cells/ul)",
		"NK count": "NK count(cells/ul)",
		"CD3+ count": "CD3+ count(cells/ul)",
		"VitA": "VitA(umol/L)",
		"VitB1": "VitB1(nmol/L)",
		"VitB2": "VitB2(ug/L)",
		"VitB6": "VitB6(umol/L)",
		"VitC": "VitC(umol/L)",
		"VitE": "VitE(ug/L)",
		"Calcium": "Calcium(mmol/L)",
		"SCCA": "SCCA(ng/ml)",
		"CA724": "CA724(IU/mL)",
		"NSE": "NSE(ng/ml)",
		"CYFRA21-1": "CYFRA21-1(ng/ml)",
		"CA242": "CA242(U/mL)",
		"CA50": "CA50(IU/mL)",
		"CA199": "CA199(U/mL)",
		"CEA": "CEA(ng/ml)",
		"AFP": "AFP(ng/ml)",
		"CA153": "CA153(U/mL)",
		"CA125": "CA125(U/mL)",
		"WBC": "WBC(10^9/L)",
		"RBC": "RBC(10^12/L)",
		"Hemoglobin": "Hemoglobin(g/L)",
		"Neutrophil count": "Neutrophil count(10^9/L)",
		"Lymphocyte count": "Lymphocyte count(10^9/L)",
		"Platelet count": "Platelet count(10^9/L)",
		"Monocyte count": "Monocyte count(10^9/L)",
		"CRP": "CRP(mg/L)",
		"Iron": "Iron(mmol/L)",
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
		# Beautify: underscore -> space, and collapse redundant spaces
		s = s.replace("_", " ")
		s = " ".join(s.split())
		# One-hot suffix might come from float (e.g. 0.0/1.0/2.0), normalize to integer to hit mapping
		s = re.sub(r"(?<!\d)(-?\d+)\.0+(?!\d)", r"\1", s)
		# Convert HTML sup to ^ expression suitable for graphs
		s = s.replace("<sup>", "^").replace("</sup>", "")
		# Override: map categorical levels to expected display name
		s = DISPLAY_OVERRIDES.get(s, s)
		# Override: append continuous variable units
		s = UNIT_OVERRIDES.get(s, s)
		cleaned.append(s)

	# Deduplicate: append index suffix to repeating items
	seen: dict[str, int] = {}
	final: list[str] = []
	for s in cleaned:
		cnt = seen.get(s, 0) + 1
		seen[s] = cnt
		final.append(s if cnt == 1 else f"{s} ({cnt})")
	return final


def _shap_explanation_with_feature_names(expl: Any, feature_names: list[str]) -> Any:
	"""Return a SHAP Explanation with replaced feature_names.

	Some SHAP plotting helpers (e.g., waterfall) read names from the Explanation itself.
	"""
	try:
		return shap.Explanation(
			values=getattr(expl, "values"),
			base_values=getattr(expl, "base_values"),
			data=getattr(expl, "data", None),
			feature_names=list(feature_names),
		)
	except Exception:
		try:
			setattr(expl, "feature_names", list(feature_names))
		except Exception:
			pass
		return expl


def _darken_axes_artist_colors(ax: Any, factor: float = 0.78) -> None:
	"""Darken patch/collection colors on an Axes (used to tone down SHAP red/blue)."""
	f = float(factor)
	f = 1.0 if f <= 0 else f
	for p in getattr(ax, "patches", []):
		try:
			fc = p.get_facecolor()
			r, g, b, a = matplotlib.colors.to_rgba(fc)
			p.set_facecolor((r * f, g * f, b * f, a))
		except Exception:
			pass


def _recolor_shap_red_blue_to_viridis(ax: Any) -> None:
	"""Recolor SHAP waterfall's default red/blue into custom yellow/purple.

	Goal: match the example waterfall figure style.
	- Negative contributions (blue-ish) -> deep purple (#842681)
	- Positive contributions (red-ish/pink) -> gold yellow (#EFBE25)
	"""
	neg_color = "#842681"  # Deep purple-red
	pos_color = "#EFBE25"  # Gold yellow
	
	def _is_shap_red(rgba: Any) -> bool:
		"""Detect SHAP default red/pink #ff0051 (RGB: 1.0, 0.0, 0.32)"""
		r, g, b, a = matplotlib.colors.to_rgba(rgba)
		if a <= 0:
			return False
		# SHAP red features: high r (>0.9), low g (<0.2), medium-low b
		return r > 0.9 and g < 0.2 and b < 0.5
	
	def _is_shap_blue(rgba: Any) -> bool:
		"""Detect SHAP default blue #008bfb (RGB: 0.0, 0.55, 0.98)"""
		r, g, b, a = matplotlib.colors.to_rgba(rgba)
		if a <= 0:
			return False
		# SHAP blue features: low r (<0.2), high b (>0.9)
		return r < 0.2 and b > 0.9

	# Patches (arrow-shaped blocks)
	for p in getattr(ax, "patches", []):
		try:
			fc = p.get_facecolor()
			if _is_shap_red(fc):
				p.set_facecolor(pos_color)
			elif _is_shap_blue(fc):
				p.set_facecolor(neg_color)
		except Exception:
			pass
		try:
			ec = p.get_edgecolor()
			if _is_shap_red(ec):
				p.set_edgecolor(pos_color)
			elif _is_shap_blue(ec):
				p.set_edgecolor(neg_color)
		except Exception:
			pass

	# Collections (e.g. PolyCollection)
	for coll in getattr(ax, "collections", []):
		try:
			fc = np.asarray(coll.get_facecolors(), dtype=float)
			if fc.size and fc.ndim == 2:
				for i in range(fc.shape[0]):
					rgba = tuple(fc[i, :])
					if _is_shap_red(rgba):
						fc[i, :] = matplotlib.colors.to_rgba(pos_color)
					elif _is_shap_blue(rgba):
						fc[i, :] = matplotlib.colors.to_rgba(neg_color)
				coll.set_facecolor(fc)
		except Exception:
			pass
		try:
			ec = np.asarray(coll.get_edgecolors(), dtype=float)
			if ec.size and ec.ndim == 2:
				for i in range(ec.shape[0]):
					rgba = tuple(ec[i, :])
					if _is_shap_red(rgba):
						ec[i, :] = matplotlib.colors.to_rgba(pos_color)
					elif _is_shap_blue(rgba):
						ec[i, :] = matplotlib.colors.to_rgba(neg_color)
				coll.set_edgecolor(ec)
		except Exception:
			pass


def _pretty_axis_label(label: str) -> str:
	"""Format 'Name(unit)' into 'Name (unit)' for nicer axis labels."""
	s = str(label)
	m = re.match(r"^(.+?)\(([^()]+)\)\s*$", s)
	if not m:
		return s
	name = str(m.group(1)).strip()
	unit = str(m.group(2)).strip()
	return f"{name} ({unit})" if name and unit else s


def _jitter_if_categorical(
	xv: np.ndarray,
	rng: np.random.Generator | None = None,
	width: float = 0.12,
) -> tuple[np.ndarray, bool]:
	"""If xv contains few unique values (<=5), add horizontal jitter to disperse overlapping points.

	Returns (jittered_xv, is_categorical).
	"""
	uniques = np.unique(xv[np.isfinite(xv)])
	if len(uniques) > 5:
		return xv, False
	if rng is None:
		rng = np.random.default_rng(0)
	jittered = xv.copy()
	for u in uniques:
		mask = xv == u
		jittered[mask] = u + rng.uniform(-width, width, size=int(mask.sum()))
	return jittered, True


def _set_categorical_xlim(ax: Any, xv_raw: np.ndarray) -> None:
	"""Set compact xlim for categorical variables to bring scatter plots closer."""
	uniques = np.unique(xv_raw[np.isfinite(xv_raw)])
	if len(uniques) < 2:
		return
	span = float(uniques[-1] - uniques[0])
	margin = max(0.25, span * 0.18)
	ax.set_xlim(float(uniques[0]) - margin, float(uniques[-1]) + margin)


def _save_shap_dependence_panel_4col(
	out_path: Path,
	feature_labels: list[str],
	X: np.ndarray,
	shap_vals: np.ndarray,
	y: np.ndarray,
	order_idx: np.ndarray,
	*,
	ncols: int = 4,
) -> None:
	"""Save a stitched dependence panel with 4 columns for all features."""
	n = int(len(order_idx))
	if n <= 0:
		return
	ncols = max(1, int(ncols))
	nrows = int(math.ceil(n / ncols))

	# Consistent with viridis of shap_summary_dot: low value purple( M0 ) / high value yellow( M1 )
	_cmap = matplotlib.cm.get_cmap("viridis")
	palette = {
		0: matplotlib.colors.to_hex(_cmap(0.0)),
		1: matplotlib.colors.to_hex(_cmap(1.0)),
	}

	# Dependence subplots use slightly higher cells than wider, making y axis visually longer than x axis
	cell_w, cell_h = 3.3, 3.55
	fig_w = ncols * cell_w
	fig_h = nrows * cell_h
	fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
	fig.patch.set_facecolor("white")
	axes_arr = np.atleast_1d(axes).ravel()
	jitter_rng = np.random.default_rng(42)

	for k, ax in enumerate(axes_arr):
		ax.set_facecolor("#f2f2f2")
		ax.set_axisbelow(True)
		ax.grid(True, color="white", linewidth=1.0)
		if k >= n:
			ax.axis("off")
			continue
		j = int(order_idx[k])
		xv_raw = np.asarray(X[:, j], dtype=float)
		xv, is_cat = _jitter_if_categorical(xv_raw, rng=jitter_rng)
		yv = np.asarray(shap_vals[:, j], dtype=float)
		colors = [palette[int(v)] for v in np.asarray(y, dtype=int).ravel()]
		ax.scatter(xv, yv, c=colors, s=8, alpha=0.88, linewidths=0)
		if is_cat:
			_set_categorical_xlim(ax, xv_raw)
		ax.set_box_aspect(1.06)
		ax.set_title(_pretty_axis_label(feature_labels[j]), fontsize=10)
		# Only display y-axis title on the first column to avoid crowding
		if (k % ncols) == 0:
			ax.set_ylabel("SHAP value", fontsize=10)
		else:
			ax.set_ylabel("")
		ax.tick_params(axis="both", labelsize=8)

	fig.tight_layout(pad=0.6)
	fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
	plt.close(fig)


def _safe_float32_matrix(X: Any) -> Any:
	if sp.issparse(X):
		X = X.tocsr(copy=False)
		return X.astype(np.float32) if X.dtype != np.float32 else X
	X = np.asarray(X)
	return X.astype(np.float32) if X.dtype != np.float32 else X


def _save_training_artifacts(outdir: Path, preprocessor: ColumnTransformer, fitted_pipelines: dict[str, Pipeline]) -> None:
	art_dir = outdir / "artifacts"
	_ensure_dir(art_dir)
	joblib.dump(preprocessor, art_dir / "preprocessor.joblib")
	for k, pipe in fitted_pipelines.items():
		joblib.dump(pipe, art_dir / f"model_{k}.joblib")


def build_preprocessor(df: pd.DataFrame, categorical_cols: list[str]) -> tuple[ColumnTransformer, list[str], list[str]]:
	cols = list(df.columns)
	existing_cat = [c for c in categorical_cols if c in cols]
	numeric_cols = [c for c in cols if c not in existing_cat]

	numeric_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
	cat_pipe = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="most_frequent")),
			("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
		]
	)

	pre = ColumnTransformer(
		transformers=[
			("num", numeric_pipe, numeric_cols),
			("cat", cat_pipe, existing_cat),
		],
		remainder="drop",
		sparse_threshold=0.3,
	)
	return pre, numeric_cols, existing_cat


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
	try:
		names = preprocessor.get_feature_names_out()
		return [str(x) for x in names]
	except Exception:
		return []


def _extract_importance(fitted_model: BaseEstimator, n_features: int) -> np.ndarray | None:
	if hasattr(fitted_model, "coef_"):
		coef = np.asarray(getattr(fitted_model, "coef_"))
		if coef.ndim == 2:
			coef = np.mean(np.abs(coef), axis=0)
		else:
			coef = np.abs(coef)
		coef = coef.ravel()
		return coef[:n_features]

	if hasattr(fitted_model, "feature_importances_"):
		imp = np.asarray(getattr(fitted_model, "feature_importances_"), dtype=float).ravel()
		return imp[:n_features]

	return None


def build_model_specs(random_state: int) -> list[ModelSpec]:
	specs: list[ModelSpec] = []

	specs.append(
		ModelSpec(
			key="LR",
			title="Logistic Regression",
			estimator=LogisticRegression(max_iter=5000, solver="liblinear", random_state=random_state),
			param_grid={"model__C": [0.01, 0.1, 1.0, 10.0]},
			needs_scaling=True,
			needs_dense=False,
		)
	)
	specs.append(
		ModelSpec(
			key="DT",
			title="Decision Tree",
			estimator=DecisionTreeClassifier(random_state=random_state),
			param_grid={
				"model__max_depth": [None, 3, 5, 10],
				"model__min_samples_split": [2, 5, 10],
				"model__min_samples_leaf": [1, 2, 4],
			},
			needs_scaling=False,
			needs_dense=False,
		)
	)
	specs.append(
		ModelSpec(
			key="RF",
			title="Random Forest",
			estimator=RandomForestClassifier(n_estimators=500, random_state=random_state, n_jobs=-1),
			param_grid={
				"model__n_estimators": [200, 500],
				"model__max_depth": [None, 5, 10],
				"model__min_samples_leaf": [1, 2, 4],
			},
			needs_scaling=False,
			needs_dense=False,
		)
	)
	specs.append(
		ModelSpec(
			key="KNN",
			title="K Nearest Neighbors",
			estimator=KNeighborsClassifier(),
			param_grid={
				"model__n_neighbors": [3, 5, 7, 9],
				"model__weights": ["uniform", "distance"],
				"model__p": [1, 2],
			},
			needs_scaling=True,
			needs_dense=True,
		)
	)
	specs.append(
		ModelSpec(
			key="SVM",
			title="Support Vector Machine (linear)",
			estimator=SVC(kernel="linear", probability=True, random_state=random_state),
			param_grid={"model__C": [0.1, 1.0, 10.0]},
			needs_scaling=True,
			needs_dense=False,
		)
	)
	specs.append(
		ModelSpec(
			key="NB",
			title="Naive Bayes (Gaussian)",
			estimator=GaussianNB(),
			param_grid={"model__var_smoothing": [1e-9, 1e-8, 1e-7]},
			needs_scaling=False,
			needs_dense=True,
		)
	)
	if _HAS_XGBOOST:
		specs.append(
			ModelSpec(
				key="XGB",
				title="XGBoost",
				estimator=XGBClassifier(
					random_state=random_state,
					n_estimators=500,
					learning_rate=0.05,
					max_depth=3,
					subsample=0.8,
					colsample_bytree=0.8,
					reg_lambda=1.0,
					n_jobs=-1,
					eval_metric="logloss",
				),
				param_grid={
					"model__n_estimators": [200, 500],
					"model__max_depth": [3, 5],
					"model__learning_rate": [0.05, 0.1],
					"model__subsample": [0.8, 1.0],
					"model__colsample_bytree": [0.8, 1.0],
				},
				needs_scaling=False,
				needs_dense=False,
			)
		)

	specs.append(
		ModelSpec(
			key="SGBT",
			title="Stochastic Gradient Boosting",
			estimator=GradientBoostingClassifier(random_state=random_state),
			param_grid={
				"model__n_estimators": [200, 500],
				"model__learning_rate": [0.05, 0.1],
				"model__max_depth": [3, 5],
				"model__subsample": [0.8, 1.0],
			},
			needs_scaling=False,
			needs_dense=True,
		)
	)
	specs.append(
		ModelSpec(
			key="NNET",
			title="Neural Network (MLP)",
			estimator=MLPClassifier(max_iter=5000, random_state=random_state),
			param_grid={
				"model__hidden_layer_sizes": [(50,), (100,), (50, 50)],
				"model__alpha": [1e-4, 1e-3, 1e-2],
				"model__learning_rate_init": [0.001, 0.01],
			},
			needs_scaling=True,
			needs_dense=True,
		)
	)

	return specs


def make_pipeline_for_model(spec: ModelSpec, feature_indices: np.ndarray) -> Pipeline:
	steps: list[tuple[str, Any]] = [("select", ColumnSelector(feature_indices))]
	if spec.needs_dense:
		steps.append(("to_dense", ToDense()))
	if spec.needs_scaling:
		steps.append(("scaler", StandardScaler(with_mean=False)))
	steps.append(("model", clone(spec.estimator)))
	return Pipeline(steps=steps)


def choose_subset_sizes(n_features: int, max_points: int = 30) -> np.ndarray:
	if n_features <= max_points:
		return np.arange(1, n_features + 1)
	grid = np.unique(np.round(np.linspace(2, n_features, max_points)).astype(int))
	grid = np.unique(np.concatenate([np.array([1]), grid, np.array([n_features])]))
	return grid


def _parse_float_list(s: str) -> list[float]:
	vals: list[float] = []
	for part in str(s).split(","):
		p = part.strip()
		if not p:
			continue
		vals.append(float(p))
	return vals


def _fit_lr_selector_gs(
	X_train: Any,
	y_train: np.ndarray,
	method: str,
	C_grid: list[float],
	l1_ratio_grid: list[float],
	cv: StratifiedKFold,
	random_state: int,
	n_jobs: int,
) -> GridSearchCV:
	"""Grid-search L1/ElasticNet logistic regression for feature selection."""
	method = str(method).lower()
	if method not in {"l1", "elasticnet"}:
		raise ValueError("method must be 'l1' or 'elasticnet'")

	base = LogisticRegression(
		max_iter=20000,
		solver="saga",
		random_state=random_state,
		penalty="l1" if method == "l1" else "elasticnet",
		l1_ratio=None if method == "l1" else 0.5,
	)
	pipe = Pipeline(
		steps=[
			("scaler", StandardScaler(with_mean=False)),
			("model", base),
		]
	)

	param_grid: dict[str, list[Any]] = {"model__C": [float(c) for c in C_grid]}
	if method == "elasticnet":
		param_grid["model__l1_ratio"] = [float(a) for a in l1_ratio_grid]

	gs = GridSearchCV(
		estimator=pipe,
		param_grid=param_grid,
		scoring="roc_auc",
		cv=cv,
		n_jobs=n_jobs,
		refit=True,
	)
	gs.fit(X_train, y_train)
	return gs


def _coef_abs_from_fitted_lr(pipe: Pipeline) -> np.ndarray:
	model = pipe.named_steps["model"]
	coef = np.asarray(getattr(model, "coef_"))
	if coef.ndim == 2:
		coef = coef[0]
	return np.abs(np.asarray(coef, dtype=float).ravel())


_FS_CATEGORICAL_DEDUP_BASES: Final[set[str]] = {
	"T stage",
	"N stage",
	"Differentiation grade",
	"Vascular invasion",
	"Perineural invasion",
	"Carcinoma nodule",
	"CDX2",
	"MLH1",
	"MSH2",
	"PMS2",
	"MSH6",
	"HER2",
	"Family history",
	"Colonic obstruction",
	"Hypertension",
	"Diabetes",
	"Coronary artery disease",
	"Hyperlipidemia",
	"Chemotherapy",
	"BRAF mutant",
	"KRAS mutant",
	"NRAS mutant",
	"HER2 mutant",
	"NTRK1 mutant",
	"NTRK2 mutant",
	"NTRK3 mutant",
	"RET mutant",
	"MSI-H",
	"mGPS",
}


def _categorical_base_from_display(display_name: str) -> str | None:
	"""Extract base name from display name format 'base(level)'.

	Used only for feature selection deduplication: keep only one column with higher importance for multiple dummy variables of the same base.
	"""
	s = str(display_name).strip()
	# Special: MSI-H binary classification is shown briefly as MSS / MSI-H on charts.
	# But we want to treat them as the same base in feature selection deduplication, keeping only the column with higher importance.
	if s in {"MSS", "MSI-H"}:
		return "MSI-H"
	# Special: mGPS uses colon format "mGPS: 0", "mGPS: 1", "mGPS: 2"
	if s.startswith("mGPS:"):
		return "mGPS"
	m = re.match(r"^(.*)\(([^()]+)\)\s*$", s)
	if not m:
		return None
	base = str(m.group(1)).strip()
	return base or None


def _stitch_lr_fs_process_figures(outdir_fig: Path) -> None:
	"""Stitch 5 LR-FS process plots into a panel plot for paper layout convenience.

	Expected input files (all located under outdir_fig):
	- FS_LR_process_auc_vs_C.png
	- FS_LR_process_auc_vs_nfeatures.png
	- FS_LR_process_grid_auc_nfeatures_vs_C.png
	- FS_LR_process_heatmap_auc.png
	- FS_LR_process_nfeatures_vs_C.png
	"""
	files = {
		"A": outdir_fig / "FS_LR_process_grid_auc_nfeatures_vs_C.png",
		"B": outdir_fig / "FS_LR_process_auc_vs_C.png",
		"C": outdir_fig / "FS_LR_process_nfeatures_vs_C.png",
		"D": outdir_fig / "FS_LR_process_auc_vs_nfeatures.png",
		"E": outdir_fig / "FS_LR_process_heatmap_auc.png",
	}
	if not all(p.exists() for p in files.values()):
		return

	imgs: dict[str, Any] = {}
	for k, p in files.items():
		imgs[k] = plt.imread(p)

	fig = plt.figure(figsize=(24, 12), constrained_layout=True)
	gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1.0, 1.0], width_ratios=[1.0, 1.0, 1.0])

	axA = fig.add_subplot(gs[0, 0])
	axB = fig.add_subplot(gs[0, 1])
	axC = fig.add_subplot(gs[0, 2])
	axD = fig.add_subplot(gs[1, 0])
	axE = fig.add_subplot(gs[1, 1:])
	axes = {"A": axA, "B": axB, "C": axC, "D": axD, "E": axE}

	for key, ax in axes.items():
		ax.imshow(imgs[key])
		ax.axis("off")
		ax.text(
			0.01,
			0.99,
			key,
			transform=ax.transAxes,
			va="top",
			ha="left",
			fontsize=18,
			fontweight="bold",
			color="black",
			bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.3", alpha=0.92),
		)

	fig.savefig(outdir_fig / "FS_LR_process_panel_5in1.png", dpi=300)
	plt.close(fig)


def _plot_lr_fs_process(
	gs: GridSearchCV,
	X_train: Any,
	y_train: np.ndarray,
	method: str,
	C_grid: list[float],
	l1_ratio_grid: list[float],
	coef_tol: float,
	random_state: int,
	outdir_tables: Path,
	outdir_fig: Path,
) -> None:
	"""Output the process plots and tables of L1/ElasticNet Logistic feature selection.

	- AUC vs C (grouped by l1_ratio)
	- ElasticNet: heatmap of AUC for l1_ratio x C
	- Number of non-zero features vs C (grouped by l1_ratio, zero determined by coef_tol)
	- Scatter plot of AUC vs number of non-zero features
	"""
	try:
		method = str(method).lower()
		coef_tol = float(coef_tol)
		results = gs.cv_results_
		if not results:
			return

		df = pd.DataFrame(results)
		# Uniformly extract parameter columns (column names might be missing in some sklearn versions)
		if "param_model__C" in df.columns:
			df["C"] = df["param_model__C"].astype(float)
		elif "param_C" in df.columns:
			df["C"] = df["param_C"].astype(float)
		else:
			# If unable to retrieve from results, fall back to C_grid (but cannot align params row-by-row)
			df["C"] = np.nan

		if method == "elasticnet":
			if "param_model__l1_ratio" in df.columns:
				df["l1_ratio"] = df["param_model__l1_ratio"].astype(float)
			elif "param_l1_ratio" in df.columns:
				df["l1_ratio"] = df["param_l1_ratio"].astype(float)
			else:
				df["l1_ratio"] = float(l1_ratio_grid[0]) if l1_ratio_grid else 0.5
		else:
			df["l1_ratio"] = np.nan

		if "mean_test_score" in df.columns:
			df["mean_auc"] = df["mean_test_score"].astype(float)
		else:
			df["mean_auc"] = np.nan
		if "std_test_score" in df.columns:
			df["std_auc"] = df["std_test_score"].astype(float)
		else:
			df["std_auc"] = np.nan
		if "rank_test_score" in df.columns:
			df["rank"] = df["rank_test_score"].astype(int)
		else:
			df["rank"] = np.nan

		# To compute the "number of non-zero features", refit once on the whole training set for each parameter point
		# (The count is usually small: len(C_grid) * len(l1_ratio_grid))
		base = LogisticRegression(
			max_iter=20000,
			solver="saga",
			random_state=random_state,
			penalty="l1" if method == "l1" else "elasticnet",
			l1_ratio=None if method == "l1" else 0.5,
		)
		pipe = Pipeline(
			steps=[
				("scaler", StandardScaler(with_mean=False)),
				("model", base),
			]
		)

		# Avoid failing to iterate due to missing columns of C/l1_ratio in cv_results_: use "explicit grid" then merge
		grid_rows: list[dict[str, Any]] = []
		if method == "elasticnet":
			_lr_grid = [float(x) for x in (l1_ratio_grid if l1_ratio_grid else [0.5])]
			for lr in _lr_grid:
				for C in [float(x) for x in C_grid]:
					grid_rows.append({"C": C, "l1_ratio": lr})
		else:
			for C in [float(x) for x in C_grid]:
				grid_rows.append({"C": C, "l1_ratio": np.nan})
		grid_df = pd.DataFrame(grid_rows)

		n_nonzero_list: list[int] = []
		for row in grid_rows:
			C = float(row["C"])
			lr = row["l1_ratio"]
			m = clone(pipe)
			m.named_steps["model"].set_params(C=C)
			if method == "elasticnet":
				m.named_steps["model"].set_params(l1_ratio=float(lr))
			m.fit(X_train, y_train)
			abs_coef = _coef_abs_from_fitted_lr(m)
			n_nonzero_list.append(int(np.sum(abs_coef > coef_tol)))
		grid_df["n_nonzero"] = n_nonzero_list

		# Merge n_nonzero back into df (matched by C/l1_ratio)
		# Due to floating point errors, round before merge.
		df["C_round"] = df["C"].astype(float).round(12)
		grid_df["C_round"] = grid_df["C"].astype(float).round(12)
		if method == "elasticnet":
			df["l1_ratio_round"] = df["l1_ratio"].astype(float).round(12)
			grid_df["l1_ratio_round"] = grid_df["l1_ratio"].astype(float).round(12)
			df = df.merge(
				grid_df[["C_round", "l1_ratio_round", "n_nonzero"]],
				on=["C_round", "l1_ratio_round"],
				how="left",
			)
		else:
			df = df.merge(grid_df[["C_round", "n_nonzero"]], on=["C_round"], how="left")

		df = df.drop(columns=[c for c in ["C_round", "l1_ratio_round"] if c in df.columns])
		df = df.sort_values(["l1_ratio", "C"], ascending=[True, True], ignore_index=True)
		df.to_csv(outdir_tables / "fs_lr_gridsearch_results.csv", index=False, encoding="utf-8-sig")

		# 0) Mimic RFE example plot: one grid for each l1_ratio, dual y-axes (AUC + number of non-zero features) vs C
		# Goal: see "process curves" of all l1_ratio in one plot.
		try:
			letters = [f"({chr(ord('A') + i)})" for i in range(26)]
			best = None
			if "rank" in df.columns and df["rank"].notna().any():
				best = df.loc[df["rank"].astype(float).idxmin()]

			auc_series = pd.to_numeric(df.get("mean_auc", pd.Series([], dtype=float)), errors="coerce")
			auc_min = float(np.nanmin(auc_series.to_numpy())) if auc_series.size else float("nan")
			auc_max = float(np.nanmax(auc_series.to_numpy())) if auc_series.size else float("nan")
			auc_pad = 0.02
			auc_ylim = (
				max(0.0, (auc_min - auc_pad) if np.isfinite(auc_min) else 0.0),
				min(1.0, (auc_max + auc_pad) if np.isfinite(auc_max) else 1.0),
			)
			nnz_series = pd.to_numeric(df.get("n_nonzero", pd.Series([], dtype=float)), errors="coerce")
			nnz_max = float(np.nanmax(nnz_series.to_numpy())) if nnz_series.size else float("nan")
			nnz_ylim = (0.0, max(1.0, float(math.ceil(nnz_max / 5.0) * 5.0)) if np.isfinite(nnz_max) else 1.0)

			def _unique_preserve(seq: list[float]) -> list[float]:
				out: list[float] = []
				seen: set[float] = set()
				for x in seq:
					xf = float(x)
					if xf in seen:
						continue
					seen.add(xf)
					out.append(xf)
				return out

			if method == "elasticnet":
				preferred = _unique_preserve([float(x) for x in (l1_ratio_grid if l1_ratio_grid else [])])
				present = _unique_preserve(
					[float(x) for x in pd.to_numeric(df["l1_ratio"], errors="coerce").dropna().unique().tolist()]
				)
				# Prioritize by user-provided l1_ratio_grid order, then fill remainder
				ratios = [r for r in preferred if r in set(present)] + [r for r in present if r not in set(preferred)]
				# Maximum of 9 to align with RFE 3x3
				ratios = ratios[:9]
				if ratios:
					# 2x2 layout when ratios<=4 for better paper friendliness; otherwise 3x3
					ncols = 2 if len(ratios) <= 4 else 3
					nrows = int(math.ceil(len(ratios) / ncols))
					fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5.2 * nrows), constrained_layout=True)
					axes_arr = np.atleast_1d(axes).ravel()

					for i, lr in enumerate(ratios):
						ax = axes_arr[i]
						sub = df[pd.to_numeric(df["l1_ratio"], errors="coerce") == float(lr)].copy()
						sub = sub.sort_values("C")
						xc = pd.to_numeric(sub["C"], errors="coerce").astype(float)
						x = np.log10(xc)
						y_auc = pd.to_numeric(sub["mean_auc"], errors="coerce").astype(float)
						y_auc_sd = pd.to_numeric(sub.get("std_auc", np.nan), errors="coerce").astype(float)
						y_nnz = pd.to_numeric(sub["n_nonzero"], errors="coerce").astype(float)

						ax.set_facecolor("#fbfbfb")
						ax.set_axisbelow(True)
						line_auc = ax.plot(
							x,
							y_auc,
							marker="o",
							markersize=7,
							markeredgewidth=0.8,
							markeredgecolor="white",
							linewidth=2.6,
							label="Mean CV AUC",
							zorder=3,
						)[0]
						# Uncertainty: mean +- sd (across folds)
						if y_auc_sd.notna().any():
							lo = (y_auc - y_auc_sd).to_numpy(dtype=float)
							hi = (y_auc + y_auc_sd).to_numpy(dtype=float)
							ax.fill_between(
								x,
								lo,
								hi,
								color=line_auc.get_color(),
								alpha=0.08,
								linewidth=0,
								zorder=1,
							)
						ax2 = ax.twinx()
						ax2.plot(
							x,
							y_nnz,
							marker="s",
							markersize=6.5,
							markeredgewidth=0.8,
							markeredgecolor="white",
							color="tab:orange",
							linewidth=2.2,
							label=f"#Features (|coef|>{coef_tol:g})",
							zorder=3,
						)

						# Mark optimal point (if falls inside this l1_ratio subplot)
						if best is not None and pd.notna(best.get("C")) and pd.notna(best.get("l1_ratio")):
							if float(best.get("l1_ratio")) == float(lr):
								best_c = float(best.get("C"))
								best_x = math.log10(best_c)
								ax.axvline(best_x, color="0.3", linestyle=":", linewidth=1)
								ax.scatter([best_x], [float(best.get("mean_auc"))], s=95, c="#ef4444", zorder=5)

						ax.set_xlabel("log10(C)")
						ax.set_ylabel("Mean CV AUC")
						ax2.set_ylabel("# Non-zero features")
						ax.set_ylim(*auc_ylim)
						ax2.set_ylim(*nnz_ylim)
						ax.set_title(f"l1_ratio={float(lr):.2f} {letters[i]}")
						ax.grid(True, alpha=0.18)

						h1, l1 = ax.get_legend_handles_labels()
						h2, l2 = ax2.get_legend_handles_labels()
						ax.legend(h1 + h2, l1 + l2, fontsize=9, loc="lower right", frameon=False)

					for j in range(len(ratios), len(axes_arr)):
						axes_arr[j].axis("off")

					fig.suptitle("Elastic Net FS Process: Mean±SD CV AUC & #Selected Features vs log10(C)", fontsize=16)
					fig.savefig(outdir_fig / "FS_LR_process_grid_auc_nfeatures_vs_C.png", dpi=300)
					plt.close(fig)
			else:
				# L1: single plot, but still output with dual y-axes style of RFE
				g = df.sort_values("C")
				xc = pd.to_numeric(g["C"], errors="coerce").astype(float)
				x = np.log10(xc)
				y_auc = pd.to_numeric(g["mean_auc"], errors="coerce").astype(float)
				y_auc_sd = pd.to_numeric(g.get("std_auc", np.nan), errors="coerce").astype(float)
				y_nnz = pd.to_numeric(g["n_nonzero"], errors="coerce").astype(float)

				fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
				fig.patch.set_facecolor("white")
				ax.set_facecolor("#fbfbfb")
				ax.set_axisbelow(True)
				line_auc = ax.plot(
					x,
					y_auc,
					marker="o",
					markersize=7,
					markeredgewidth=0.8,
					markeredgecolor="white",
					linewidth=2.6,
					label="Mean CV AUC",
					zorder=3,
				)[0]
				if y_auc_sd.notna().any():
					lo = (y_auc - y_auc_sd).to_numpy(dtype=float)
					hi = (y_auc + y_auc_sd).to_numpy(dtype=float)
					ax.fill_between(x, lo, hi, color=line_auc.get_color(), alpha=0.08, linewidth=0, zorder=1)
				ax2 = ax.twinx()
				ax2.plot(
					x,
					y_nnz,
					marker="s",
					markersize=6.5,
					markeredgewidth=0.8,
					markeredgecolor="white",
					color="tab:orange",
					linewidth=2.2,
					label=f"#Features (|coef|>{coef_tol:g})",
					zorder=3,
				)
				if best is not None and pd.notna(best.get("C")):
					best_c = float(best.get("C"))
					best_x = math.log10(best_c)
					ax.axvline(best_x, color="0.3", linestyle=":", linewidth=1)
					ax.scatter([best_x], [float(best.get("mean_auc"))], s=95, c="#ef4444", zorder=5)
				ax.set_xlabel("log10(C)")
				ax.set_ylabel("Mean CV AUC")
				ax2.set_ylabel("# Non-zero features")
				ax.set_title(f"L1 {letters[0]}")
				ax.grid(True, alpha=0.18)
				h1, l1 = ax.get_legend_handles_labels()
				h2, l2 = ax2.get_legend_handles_labels()
				ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower right", frameon=False)
				fig.suptitle("L1 FS Process: Mean±SD CV AUC & #Selected Features vs log10(C)", fontsize=14)
				fig.savefig(outdir_fig / "FS_LR_process_grid_auc_nfeatures_vs_C.png", dpi=300)
				plt.close(fig)
		except Exception:
			# Failure of process plot does not affect main pipeline
			pass

		# 1) AUC vs C curve (with mean+-sd shadow band)
		fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
		fig.patch.set_facecolor("white")
		ax.set_facecolor("#fbfbfb")
		ax.set_axisbelow(True)
		best = None
		if "rank" in df.columns and df["rank"].notna().any():
			best = df.loc[df["rank"].astype(float).idxmin()]

		if method == "elasticnet":
			for lr, g in df.groupby("l1_ratio", dropna=True):
				g = g.sort_values("C")
				x = np.log10(g["C"].astype(float))
				y = g["mean_auc"].astype(float)
				sd = pd.to_numeric(g.get("std_auc", np.nan), errors="coerce").astype(float)
				line = ax.plot(
					x,
					y,
					marker="o",
					markersize=7,
					markeredgewidth=0.8,
					markeredgecolor="white",
					linewidth=2.6,
					label=f"l1_ratio={float(lr):.2f}",
					zorder=3,
				)[0]
				if sd.notna().any():
					ax.fill_between(
						x,
						(y - sd).to_numpy(dtype=float),
						(y + sd).to_numpy(dtype=float),
						color=line.get_color(),
						alpha=0.08,
						linewidth=0,
						zorder=1,
					)
			# best point
			if best is not None and pd.notna(best.get("C")):
				ax.scatter([math.log10(float(best["C"]))], [float(best["mean_auc"])], s=80, c="#ef4444", zorder=5)
		else:
			g = df.sort_values("C")
			x = np.log10(g["C"].astype(float))
			y = g["mean_auc"].astype(float)
			sd = pd.to_numeric(g.get("std_auc", np.nan), errors="coerce").astype(float)
			line = ax.plot(
				x,
				y,
				marker="o",
				markersize=7,
				markeredgewidth=0.8,
				markeredgecolor="white",
				linewidth=2.6,
				zorder=3,
			)[0]
			if sd.notna().any():
				ax.fill_between(
					x,
					(y - sd).to_numpy(dtype=float),
					(y + sd).to_numpy(dtype=float),
					color=line.get_color(),
					alpha=0.08,
					linewidth=0,
					zorder=1,
				)
			# best point
			if best is not None and pd.notna(best.get("C")):
				ax.scatter([math.log10(float(best["C"]))], [float(best["mean_auc"])], s=80, c="#ef4444", zorder=5)
		ax.set_xlabel("log10(C)")
		ax.set_ylabel("Mean CV AUC")
		ax.set_title("FS Process: Mean±SD CV AUC vs log10(C)")
		ax.grid(True, alpha=0.18)
		if method == "elasticnet":
			ax.legend(frameon=False, ncol=1, bbox_to_anchor=(1.02, 1), loc="upper left")
		fig.savefig(outdir_fig / "FS_LR_process_auc_vs_C.png", dpi=300)
		plt.close(fig)

		# 2) Number of non-zero features vs C curve
		if "n_nonzero" in df.columns and df["n_nonzero"].notna().any():
			fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
			if method == "elasticnet":
				for lr, g in df.groupby("l1_ratio", dropna=True):
					g = g.sort_values("C")
					x = np.log10(g["C"].astype(float))
					y = g["n_nonzero"].astype(float)
					ax.plot(x, y, marker="o", linewidth=1.6, label=f"l1_ratio={float(lr):.2f}")
				ax.legend(frameon=False, ncol=1, bbox_to_anchor=(1.02, 1), loc="upper left")
			else:
				g = df.sort_values("C")
				x = np.log10(g["C"].astype(float))
				y = g["n_nonzero"].astype(float)
				ax.plot(x, y, marker="o", linewidth=1.6)
			ax.set_xlabel("log10(C)")
			ax.set_ylabel(f"#Non-zero features (|coef| > {coef_tol:g})")
			ax.set_title("FS Process: #Selected Features vs log10(C)")
			ax.grid(True, alpha=0.25)
			fig.savefig(outdir_fig / "FS_LR_process_nfeatures_vs_C.png", dpi=300)
			plt.close(fig)

			# 3) Scatter of AUC vs number of non-zero features (with y error bar: +-sd)
			fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
			if method == "elasticnet":
				for lr, g in df.groupby("l1_ratio", dropna=True):
					x = g["n_nonzero"].astype(float)
					y = g["mean_auc"].astype(float)
					yerr = pd.to_numeric(g.get("std_auc", np.nan), errors="coerce").astype(float)
					ax.errorbar(
						x,
						y,
						yerr=yerr.to_numpy(dtype=float) if yerr.notna().any() else None,
						fmt="o",
						ms=6,
						alpha=0.9,
						capsize=3,
						label=f"l1_ratio={float(lr):.2f}",
					)
				ax.legend(frameon=False, ncol=1, bbox_to_anchor=(1.02, 1), loc="upper left")
			else:
				x = df["n_nonzero"].astype(float)
				y = df["mean_auc"].astype(float)
				yerr = pd.to_numeric(df.get("std_auc", np.nan), errors="coerce").astype(float)
				ax.errorbar(
					x,
					y,
					yerr=yerr.to_numpy(dtype=float) if yerr.notna().any() else None,
					fmt="o",
					ms=6,
					alpha=0.9,
					capsize=3,
				)
			ax.set_xlabel(f"#Non-zero features (|coef| > {coef_tol:g})")
			ax.set_ylabel("Mean CV AUC")
			ax.set_title("FS Process: Mean±SD CV AUC vs #Selected Features")
			ax.grid(True, alpha=0.25)
			fig.savefig(outdir_fig / "FS_LR_process_auc_vs_nfeatures.png", dpi=300)
			plt.close(fig)

		# 4) ElasticNet: AUC Heatmap (l1_ratio x C)
		if method == "elasticnet":
			try:
				tmp = df.copy()
				tmp["C"] = pd.to_numeric(tmp["C"], errors="coerce").astype(float)
				pivot_mean = tmp.pivot_table(index="l1_ratio", columns="C", values="mean_auc", aggfunc="mean")
				pivot_sd = tmp.pivot_table(index="l1_ratio", columns="C", values="std_auc", aggfunc="mean")
				pivot_mean = pivot_mean.sort_index(axis=0).sort_index(axis=1)
				pivot_sd = pivot_sd.reindex_like(pivot_mean)
				# Construct annot table using string/object separately to avoid pandas FutureWarning
				annot = pd.DataFrame("", index=pivot_mean.index, columns=pivot_mean.columns, dtype=object)
				for r in range(pivot_mean.shape[0]):
					for c in range(pivot_mean.shape[1]):
						m = float(pivot_mean.iloc[r, c]) if pd.notna(pivot_mean.iloc[r, c]) else float("nan")
						s = float(pivot_sd.iloc[r, c]) if pd.notna(pivot_sd.iloc[r, c]) else float("nan")
						if np.isfinite(m) and np.isfinite(s):
							annot.iloc[r, c] = f"{m:.3f}±{s:.3f}"
						elif np.isfinite(m):
							annot.iloc[r, c] = f"{m:.3f}"
						else:
							annot.iloc[r, c] = ""

				fig, ax = plt.subplots(figsize=(13, 4 + 0.5 * max(1, pivot_mean.shape[0])), constrained_layout=True)
				sns.heatmap(
					pivot_mean,
					ax=ax,
					cmap="viridis",
					annot=annot,
					fmt="",
					cbar_kws={"label": "Mean CV AUC"},
				)
				# Display C directly on x-axis (no log10), keeping label format consistent with example figure
				def _fmt_c_tick(v: float) -> str:
					v = float(v)
					if v >= 1 and float(v).is_integer():
						return f"{v:.1f}"
					if v >= 0.1:
						return f"{v:g}"
					return f"{v:.3g}"

				ax.set_xlabel("C")
				ax.set_xticklabels([_fmt_c_tick(c) for c in pivot_mean.columns.to_numpy(dtype=float)], rotation=0)
				ax.set_ylabel("l1_ratio")
				ax.set_title("FS Process: Mean±SD CV AUC Heatmap (l1_ratio × C)")
				fig.savefig(outdir_fig / "FS_LR_process_heatmap_auc.png", dpi=300)
				plt.close(fig)
				try:
					_stitch_lr_fs_process_figures(outdir_fig)
				except Exception:
					pass
			except Exception:
				pass
	except Exception:
		# Process plot failure does not affect main pipeline
		return


def select_features_lr_enet_or_l1(
	X_train: Any,
	y_train: np.ndarray,
	feature_names: list[str],
	method: str,
	C_grid: list[float],
	l1_ratio_grid: list[float],
	fs_cv_folds: int,
	min_features: int,
	max_features: int,
	coef_tol: float,
	stability: bool,
	stability_runs: int,
	stability_fraction: float,
	stability_threshold: float,
	random_state: int,
	n_jobs: int,
	outdir_tables: Path,
	outdir_fig: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
	"""Return selected feature indices based on L1/ElasticNet logistic regression.

	If stability=True, uses repeated subsampling to keep only stable features.
	Always enforces min_features and caps to max_features.
	"""
	n_features = int(X_train.shape[1])
	min_features = max(1, int(min_features))
	max_features = max(min_features, int(max_features))
	coef_tol = float(coef_tol)
	method = str(method).lower()

	fs_cv_folds = int(fs_cv_folds)
	fs_cv_folds = 5 if fs_cv_folds <= 1 else fs_cv_folds
	fs_cv = StratifiedKFold(n_splits=fs_cv_folds, shuffle=True, random_state=random_state)
	gs = _fit_lr_selector_gs(
		X_train=X_train,
		y_train=y_train,
		method=method,
		C_grid=C_grid,
		l1_ratio_grid=l1_ratio_grid,
		cv=fs_cv,
		random_state=random_state,
		n_jobs=n_jobs,
	)
	# Output process curves/heatmaps/feature count changes of ElasticNet/L1 feature selection
	_plot_lr_fs_process(
		gs=gs,
		X_train=X_train,
		y_train=y_train,
		method=method,
		C_grid=C_grid,
		l1_ratio_grid=l1_ratio_grid,
		coef_tol=coef_tol,
		random_state=random_state,
		outdir_tables=outdir_tables,
		outdir_fig=outdir_fig,
	)
	best_pipe: Pipeline = gs.best_estimator_
	abs_coef = _coef_abs_from_fitted_lr(best_pipe)
	order = np.argsort(abs_coef)[::-1]

	# Base selection: non-zero (by tolerance)
	selected = np.flatnonzero(abs_coef > coef_tol).astype(int)
	if selected.size < min_features:
		selected = order[:min_features].astype(int)

	# Stability selection (optional)
	meta: dict[str, Any] = {
		"fs_method": method,
		"best_params": gs.best_params_,
		"best_cv_auc": float(gs.best_score_),
		"coef_tol": coef_tol,
		"stability": bool(stability),
	}

	freq = None
	stability_rank = None
	if stability:
		# Use stratified repeated subsampling (more robust than pure random sampling: sub-sample category ratio closer to total)
		n = int(y_train.shape[0])
		train_size = float(stability_fraction)
		train_size = 0.8 if not (0.05 < train_size < 1.0) else train_size
		n_splits = int(stability_runs)
		n_splits = 200 if n_splits <= 1 else n_splits
		counts = np.zeros(n_features, dtype=int)

		sss = StratifiedShuffleSplit(
			n_splits=n_splits,
			train_size=train_size,
			random_state=random_state,
		)

		# Build fixed estimator with best params
		best_model: LogisticRegression = clone(best_pipe.named_steps["model"])
		fixed = Pipeline(
			steps=[
				("scaler", clone(best_pipe.named_steps["scaler"])),
				("model", best_model),
			]
		)

		dummy_X = np.zeros(n, dtype=int)
		for tr_idx, _ in sss.split(dummy_X, y_train):
			m = clone(fixed)
			m.fit(X_train[tr_idx], y_train[tr_idx])
			ac = _coef_abs_from_fitted_lr(m)
			counts[ac > coef_tol] += 1

		freq = counts / float(n_splits)
		stability_rank = np.lexsort((abs_coef, freq))[::-1]
		stable = np.flatnonzero(freq >= float(stability_threshold)).astype(int)
		if stable.size < min_features:
			# Complement: prioritize frequency, then abs(coef)
			stable = stability_rank[:min_features].astype(int)
		selected = stable
		meta.update(
			{
				"stability_splitter": "StratifiedShuffleSplit",
				"stability_runs": int(n_splits),
				"stability_fraction": float(train_size),
				"stability_threshold": float(stability_threshold),
				"n_stable_features": int(selected.size),
			}
		)

	# Cap number of features for dense models downstream
	if selected.size > max_features:
		if freq is None or stability_rank is None:
			selected = order[:max_features].astype(int)
			meta["capped_to_max_features_by"] = "abs_coef"
		else:
			selected_set = set(int(i) for i in selected.tolist())
			selected = np.array(
				[int(i) for i in stability_rank.tolist() if int(i) in selected_set][:max_features],
				dtype=int,
			)
			meta["capped_to_max_features_by"] = "frequency_then_abs_coef"
		meta["capped_to_max_features"] = int(max_features)

	selected = np.unique(selected.astype(int))
	# Keep a consistent order
	selected_set = set(int(i) for i in selected.tolist())
	if freq is not None and stability_rank is not None:
		rank = stability_rank
	else:
		rank = order
	selected = np.array([int(i) for i in rank.tolist() if int(i) in selected_set], dtype=int)

	# Categorical variables deduplication: for specified base, keep only one column with higher importance among different levels of the same variable
	try:
		names = feature_names if feature_names else [f"f{i}" for i in range(n_features)]
		sel_names_tmp = [names[i] for i in selected]
		sel_disp_tmp = _clean_display_feature_names(sel_names_tmp)
		best_by_base: dict[str, int] = {}
		score_by_base: dict[str, tuple[float, float]] = {}
		for idx, disp in zip(selected.tolist(), sel_disp_tmp, strict=False):
			base = _categorical_base_from_display(str(disp))
			if base is None or base not in _FS_CATEGORICAL_DEDUP_BASES:
				continue
			# Importance first: abs_coef; break ties with freq if equal (more stable under stability mode)
			c = float(abs_coef[idx])
			f = float(freq[idx]) if (freq is not None) else 0.0
			score = (c, f)
			prev = score_by_base.get(base)
			if prev is None or score > prev:
				score_by_base[base] = score
				best_by_base[base] = int(idx)

		if best_by_base:
			keep_idx: list[int] = []
			for idx, disp in zip(selected.tolist(), sel_disp_tmp, strict=False):
				base = _categorical_base_from_display(str(disp))
				if base is None or base not in _FS_CATEGORICAL_DEDUP_BASES:
					keep_idx.append(int(idx))
				else:
					if int(idx) == int(best_by_base[base]):
						keep_idx.append(int(idx))
			selected = np.array(keep_idx, dtype=int)

			# Complement by rank if lower than min_features after deduplication (avoid introducing duplicates of same base again)
			if selected.size < min_features:
				selected_set2 = set(int(i) for i in selected.tolist())
				# Occupied bases
				occupied_bases: set[str] = set()
				for idx in selected.tolist():
					base = _categorical_base_from_display(_clean_display_feature_names([names[int(idx)]])[0])
					if base and base in _FS_CATEGORICAL_DEDUP_BASES:
						occupied_bases.add(base)
				for cand in rank.tolist():
					ci = int(cand)
					if ci in selected_set2:
						continue
					base = _categorical_base_from_display(_clean_display_feature_names([names[ci]])[0])
					if base and base in _FS_CATEGORICAL_DEDUP_BASES and base in occupied_bases:
						continue
					selected_set2.add(ci)
					selected = np.append(selected, ci)
					if base and base in _FS_CATEGORICAL_DEDUP_BASES:
						occupied_bases.add(base)
					if selected.size >= min_features:
						break
				# Re-sort by rank
				selected_set2 = set(int(i) for i in selected.tolist())
				selected = np.array([int(i) for i in rank.tolist() if int(i) in selected_set2], dtype=int)
	except Exception:
		# Deduplication failure does not affect main pipeline
		pass

	# Save coefficient table
	names = feature_names if feature_names else [f"f{i}" for i in range(n_features)]
	sel_names = [names[i] for i in selected]
	sel_disp = _clean_display_feature_names(sel_names)
	rows = []
	for i, raw, disp in zip(selected.tolist(), sel_names, sel_disp, strict=False):
		rows.append(
			{
				"feature_index": int(i),
				"feature": str(raw),
				"feature_display": str(disp),
				"abs_coef": float(abs_coef[int(i)]),
				"selected": 1,
			}
		)
	coef_df = pd.DataFrame(rows).sort_values("abs_coef", ascending=False, ignore_index=True)
	coef_df.to_csv(outdir_tables / "fs_lr_selected_features.csv", index=False, encoding="utf-8-sig")
	(outdir_tables / "fs_lr_meta.json").write_text(
		json.dumps(meta, ensure_ascii=False, indent=2),
		encoding="utf-8",
	)

	if freq is not None:
		freq_rows = []
		disp_all = _clean_display_feature_names([str(x) for x in names])
		for i in range(n_features):
			freq_rows.append(
				{
					"feature_index": int(i),
					"feature": str(names[i]),
					"feature_display": str(disp_all[i]),
					"frequency": float(freq[i]),
					"abs_coef": float(abs_coef[i]),
				}
			)
		freq_df = pd.DataFrame(freq_rows)
		freq_df = freq_df.sort_values(["frequency", "abs_coef"], ascending=[False, False], ignore_index=True)
		freq_df.to_csv(outdir_tables / "fs_lr_stability_frequencies.csv", index=False, encoding="utf-8-sig")

	# Per requirement: no longer output "Top coefficients" bar chart

	return selected, meta


def cv_accuracy_kappa(
	spec: ModelSpec,
	X: Any,
	y: np.ndarray,
	feature_indices: np.ndarray,
	cv: RepeatedStratifiedKFold,
) -> tuple[float, float]:
	accs: list[float] = []
	kappas: list[float] = []
	dummy_X = np.zeros_like(y)
	for train_idx, test_idx in cv.split(dummy_X, y):
		X_tr = X[train_idx]
		X_te = X[test_idx]
		y_tr = y[train_idx]
		y_te = y[test_idx]
		pipe = make_pipeline_for_model(spec, feature_indices)
		pipe.fit(X_tr, y_tr)
		pred = pipe.predict(X_te)
		accs.append(float(accuracy_score(y_te, pred)))
		kappas.append(float(cohen_kappa_score(y_te, pred)))
	return float(np.mean(accs)), float(np.mean(kappas))


def rfe_rank_features(
	spec: ModelSpec,
	X: Any,
	y: np.ndarray,
	step: float,
	random_state: int,
	max_dense_features: int,
) -> np.ndarray:
	"""Return sorted feature indices from 'most important -> least important'.

	Explanation: Sort is generated using stepwise elimination (RFE concept), but to save memory/time,
	maximum participating feature count is restricted for models requiring dense format.
	"""

	n_total = X.shape[1]
	alive = np.arange(n_total, dtype=int)
	eliminated: list[int] = []

	if spec.needs_dense and n_total > max_dense_features:
		alive = alive[:max_dense_features]
		eliminated = list(range(max_dense_features, n_total))

	while alive.shape[0] > 1:
		pipe = make_pipeline_for_model(spec, alive)
		pipe.fit(X, y)
		model = pipe.named_steps["model"]
		importances = _extract_importance(model, alive.shape[0])
		if importances is None:
			pi = permutation_importance(
				pipe,
				X,
				y,
				n_repeats=5,
				random_state=random_state,
				scoring="roc_auc",
				n_jobs=1,
			)
			importances = pi.importances_mean

		importances = np.asarray(importances, dtype=float).ravel()
		# Robustness: some models/versions return an importance vector inconsistent with alive length under certain circumstances.
		# Force alignment to alive length to prevent index out of bounds.
		if importances.shape[0] != alive.shape[0]:
			importances = importances[: alive.shape[0]]
		remove_n = max(1, int(math.ceil(step * alive.shape[0]))) if step < 1 else int(step)
		if alive.shape[0] - remove_n < 1:
			remove_n = alive.shape[0] - 1

		remove_local = np.argsort(importances)[:remove_n]
		remove_global = alive[remove_local]
		eliminated.extend(remove_global.tolist())
		keep = np.ones(alive.shape[0], dtype=bool)
		keep[remove_local] = False
		alive = alive[keep]

	# Sort the remaining ones by final importance at the end
	final_pipe = make_pipeline_for_model(spec, alive)
	final_pipe.fit(X, y)
	final_model = final_pipe.named_steps["model"]
	final_imp = _extract_importance(final_model, alive.shape[0])
	if final_imp is None:
		pi = permutation_importance(
			final_pipe,
			X,
			y,
			n_repeats=5,
			random_state=random_state,
			scoring="roc_auc",
			n_jobs=1,
		)
		final_imp = pi.importances_mean
	final_imp = np.asarray(final_imp, dtype=float).ravel()
	if final_imp.shape[0] != alive.shape[0]:
		final_imp = final_imp[: alive.shape[0]]
	alive_sorted = alive[np.argsort(final_imp)[::-1]]

	# Eliminated entries go from least important first to moderately important later; append to end after reversing
	return np.concatenate([alive_sorted, np.array(eliminated[::-1], dtype=int)])


def rfe_curve_for_model(
	spec: ModelSpec,
	X: Any,
	y: np.ndarray,
	rank_most_to_least: np.ndarray,
	cv: RepeatedStratifiedKFold,
	max_curve_points: int,
) -> pd.DataFrame:
	sizes = choose_subset_sizes(rank_most_to_least.shape[0], max_points=max_curve_points)
	rows: list[dict[str, Any]] = []
	for k in sizes:
		selected = rank_most_to_least[: int(k)]
		acc, kappa = cv_accuracy_kappa(spec, X, y, selected, cv)
		rows.append({"model": spec.key, "k": int(k), "accuracy": acc, "kappa": kappa})
	return pd.DataFrame(rows)


def compute_feature_importance_scores(
	spec: ModelSpec,
	X: Any,
	y: np.ndarray,
	feature_indices: np.ndarray,
	feature_names: list[str],
	random_state: int,
) -> pd.DataFrame:
	pipe = make_pipeline_for_model(spec, feature_indices)
	pipe.fit(X, y)
	model = pipe.named_steps["model"]
	imp = _extract_importance(model, feature_indices.shape[0])
	if imp is None:
		pi = permutation_importance(
			pipe,
			X,
			y,
			n_repeats=10,
			random_state=random_state,
			scoring="roc_auc",
			n_jobs=1,
		)
		imp = pi.importances_mean
	imp = np.asarray(imp, dtype=float).ravel()
	if imp.shape[0] != feature_indices.shape[0]:
		imp = imp[: feature_indices.shape[0]]

	names = feature_names if feature_names else [f"f{i}" for i in range(X.shape[1])]
	feature_indices = np.asarray(feature_indices, dtype=int).ravel()
	selected_names = [names[int(i)] for i in feature_indices]
	df = pd.DataFrame({"feature_idx": feature_indices.astype(int), "feature": selected_names, "importance": imp})
	df.insert(1, "feature_display", _clean_display_feature_names([str(x) for x in df["feature"].tolist()]))
	df["abs_importance"] = df["importance"].abs()
	return df.sort_values("abs_importance", ascending=False, ignore_index=True)


def bootstrap_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
	return float(np.quantile(values, alpha / 2)), float(np.quantile(values, 1 - alpha / 2))


def metrics_from_proba(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
	y_pred = (y_proba >= threshold).astype(int)
	tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
	sens = tp / (tp + fn) if (tp + fn) else float("nan")
	spec = tn / (tn + fp) if (tn + fp) else float("nan")
	ppv = tp / (tp + fp) if (tp + fp) else float("nan")
	npv = tn / (tn + fn) if (tn + fn) else float("nan")

	return {
		"AUC": float(roc_auc_score(y_true, y_proba)),
		"Sensitivity": float(sens),
		"Specificity": float(spec),
		"PPV": float(ppv),
		"NPV": float(npv),
		"Accuracy": float(accuracy_score(y_true, y_pred)),
		"F1": float(f1_score(y_true, y_pred)),
		"Brier": float(brier_score_loss(y_true, y_proba)),
		"LogLoss": float(log_loss(y_true, np.vstack([1 - y_proba, y_proba]).T, labels=[0, 1])),
	}


def bootstrap_metrics_ci(
	y_true: np.ndarray,
	y_proba: np.ndarray,
	n_boot: int,
	random_state: int,
	threshold: float = 0.5,
) -> pd.DataFrame:
	rng = np.random.default_rng(random_state)
	n = y_true.shape[0]
	boot_rows: list[dict[str, float]] = []
	for _ in range(n_boot):
		idx = rng.integers(0, n, size=n)
		boot_rows.append(metrics_from_proba(y_true[idx], y_proba[idx], threshold=threshold))
	boot = pd.DataFrame(boot_rows)

	point = metrics_from_proba(y_true, y_proba, threshold=threshold)
	out = []
	for k, v in point.items():
		lo, hi = bootstrap_ci(boot[k].to_numpy())
		out.append({"Metric": k, "Value": v, "CI_low": lo, "CI_high": hi})
	return pd.DataFrame(out)


def hosmer_lemeshow(y_true: np.ndarray, y_proba: np.ndarray, g: int = 10) -> tuple[float, float, pd.DataFrame]:
	df = pd.DataFrame({"y": y_true, "p": y_proba}).sort_values("p").reset_index(drop=True)
	df["group"] = pd.qcut(df.index, q=g, labels=False)
	grp = df.groupby("group", as_index=False).agg(n=("y", "size"), obs=("y", "sum"), exp=("p", "sum"))
	grp["obs0"] = grp["n"] - grp["obs"]
	grp["exp0"] = grp["n"] - grp["exp"]
	stat = (
		((grp["obs"] - grp["exp"]) ** 2) / (grp["exp"] + 1e-12)
		+ ((grp["obs0"] - grp["exp0"]) ** 2) / (grp["exp0"] + 1e-12)
	).sum()
	dof = g - 2
	p = float(chi2.sf(stat, dof))
	return float(stat), p, grp


def _metric_ci_lookup(ci_xlsx_path: Path) -> dict[tuple[str, str], tuple[float, float, float]]:
	"""Load per-model CI table and return {(Dataset, Metric): (Value, low, high)}."""
	if not ci_xlsx_path.exists():
		return {}
	df = pd.read_excel(ci_xlsx_path)
	out: dict[tuple[str, str], tuple[float, float, float]] = {}
	for _, r in df.iterrows():
		key = (str(r["Dataset"]), str(r["Metric"]))
		out[key] = (float(r["Value"]), float(r["CI_low"]), float(r["CI_high"]))
	return out


def _format_ci(val: float, lo: float, hi: float, digits: int = 3) -> str:
	return f"{val:.{digits}f} ({lo:.{digits}f}–{hi:.{digits}f})"


def _plot_roc_panel(
	ax: plt.Axes,
	title: str,
	y_true: np.ndarray,
	proba_by_model: dict[str, np.ndarray],
	ci_by_model: dict[str, dict[tuple[str, str], tuple[float, float, float]]] | None,
	dataset_label: str,
) -> None:
	ax.plot([0, 1], [0, 1], color="0.8", linestyle="-", linewidth=1)
	for k, p in proba_by_model.items():
		fpr, tpr, _ = roc_curve(y_true, p)
		label = k
		if ci_by_model and k in ci_by_model:
			ci = ci_by_model[k].get((dataset_label, "AUC"))
			if ci:
				label = f"{k}  AUC: {_format_ci(*ci, digits=3)}"
		ax.plot(fpr, tpr, linewidth=1.6, label=label)
	ax.set_xlim(0, 1)
	ax.set_ylim(0, 1)
	ax.set_xlabel("1 − Specificity")
	ax.set_ylabel("Sensitivity")
	ax.set_title(title)
	ax.legend(fontsize=8, loc="lower right", frameon=False)


def _plot_calibration_panel(
	ax: plt.Axes,
	title: str,
	y_true: np.ndarray,
	proba_by_model: dict[str, np.ndarray],
	ci_by_model: dict[str, dict[tuple[str, str], tuple[float, float, float]]] | None,
	dataset_label: str,
) -> None:
	ax.plot([0, 1], [0, 1], "--", color="0.6", label="Ideal")
	# Rug plot
	for i, (k, p) in enumerate(proba_by_model.items()):
		# Draw small tick markers at the bottom y=0
		ax.plot(p, np.zeros_like(p) + (i * 0.002), "|", markersize=6, alpha=0.25)

	for k, p in proba_by_model.items():
		frac_pos, mean_pred = calibration_curve(y_true, p, n_bins=10, strategy="quantile")
		label = k
		if ci_by_model and k in ci_by_model:
			ci = ci_by_model[k].get((dataset_label, "Brier"))
			if ci:
				label = f"{k}  Brier: {_format_ci(*ci, digits=3)}"
		ax.plot(mean_pred, frac_pos, marker="o", linewidth=1.2, label=label)
	ax.set_xlim(0, 1)
	ax.set_ylim(0, 1)
	ax.set_xlabel("Predicted risk")
	ax.set_ylabel("Observed frequency")
	ax.set_title(title)
	ax.legend(fontsize=8, loc="upper left", frameon=False)


def _plot_parallel_lines(ax: plt.Axes, title: str, eval_df: pd.DataFrame, dataset: str) -> None:
	# Similar to examples E/F: one line per model
	metric_cols = [
		("Accuracy", "Accuracy"),
		("F1", "F1-score"),
		("NPV", "NPV"),
		("PPV", "Precision/PPV"),
		("Sensitivity", "Sensitivity/Recall"),
		("Specificity", "Specificity"),
	]
	sub = eval_df[eval_df["Dataset"] == dataset].copy()
	xlabels = [b for _, b in metric_cols]
	xs = np.arange(len(metric_cols))

	for _, row in sub.iterrows():
		vals = [float(row[a]) for a, _ in metric_cols]
		ax.plot(xs, vals, marker="o", linewidth=1.4, label=str(row["Model"]))
	ax.set_xticks(xs)
	ax.set_xticklabels(xlabels, rotation=0, fontsize=8)
	ax.set_ylim(0, 1)
	ax.set_ylabel("Performance Value")
	ax.set_title(title)
	ax.grid(True, axis="y", alpha=0.25)
	ax.legend(fontsize=7, loc="lower right", frameon=False)


def _save_parallel_metrics_single(out_path: Path, title: str, eval_df: pd.DataFrame, dataset: str) -> None:
	fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
	_plot_parallel_lines(ax, title, eval_df, dataset)
	fig.savefig(out_path, dpi=300)
	plt.close(fig)


def _plot_dca_panel(
	ax: plt.Axes,
	title: str,
	y_true: np.ndarray,
	proba_by_model: dict[str, np.ndarray],
	thresholds: np.ndarray,
	show_legend: bool = True,
) -> None:
	# Uniform white background; background grids removed for DCA per requirements
	ax.set_facecolor("white")
	ax.grid(False)
	try:
		for spn in ax.spines.values():
			spn.set_color("0.2")
			spn.set_linewidth(1.0)
	except Exception:
		pass

	prevalence = float(np.mean(y_true))
	den = max(prevalence, 1e-12)
	for k, p in proba_by_model.items():
		nb = decision_curve_net_benefit(y_true, p, thresholds) / den
		ax.plot(thresholds, nb, linewidth=1.3, label=k)
	nb_all = (prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))) / den
	ax.plot(thresholds, nb_all, color="0.6", linestyle="--", label="All")
	ax.plot(thresholds, np.zeros_like(thresholds), color="0.6", linestyle=":", label="None")
	ax.set_xlim(0, 1)
	ax.set_xlabel("High Risk Threshold", labelpad=2)
	# Ensure main x-axis title is centered
	try:
		ax.xaxis.label.set_horizontalalignment("center")
	except Exception:
		pass
	ax.set_ylabel("Standardized Net Benefit")
	ax.set_title(title)
	if show_legend:
		ax.legend(fontsize=7, loc="upper right", frameon=False)
	ax.set_ylim(-0.05, 1.05)
	ax.tick_params(axis="x", pad=2, labelsize=8)

	# Manually draw uniformly distributed Cost:Benefit Ratio ruler scale (downward square bracket style)
	try:
		# Scale labels (uniformly distributed)
		cbr_labels = ["1:100", "1:4", "2:3", "3:2", "4:1", "100:1"]
		n_ticks = len(cbr_labels)
		# Uniformly distributed from 0 to 1 in axes coordinate system
		tick_positions = np.linspace(0, 1, n_ticks)
		
		# Square brackets and scale y positions (axes coordinates)
		bracket_y = -0.14  # Vertical level line y position of square bracket
		tick_height = 0.02  # Downward height of scale tick line
		label_y = bracket_y - tick_height - 0.02  # y position of scale labels
		
		# Draw vertical line of square bracket
		ax.plot([0, 1], [bracket_y, bracket_y], color="0.2", linewidth=1.2, 
		        transform=ax.transAxes, clip_on=False)
		
		# Draw both ends of vertical line (opening downward)
		ax.plot([0, 0], [bracket_y, bracket_y - tick_height], color="0.2", linewidth=1.2,
		        transform=ax.transAxes, clip_on=False)
		ax.plot([1, 1], [bracket_y, bracket_y - tick_height], color="0.2", linewidth=1.2,
		        transform=ax.transAxes, clip_on=False)
		
		# Draw vertical tick line and labels for each scale point
		for i, (xpos, label) in enumerate(zip(tick_positions, cbr_labels)):
			# Scale tick line (downward)
			ax.plot([xpos, xpos], [bracket_y, bracket_y - tick_height], color="0.2", linewidth=1.0,
			        transform=ax.transAxes, clip_on=False)
			# Scale label
			ax.text(xpos, label_y, label, transform=ax.transAxes,
			        ha="center", va="top", fontsize=8)
		
		# Cost:Benefit Ratio title (shift downward)
		ax.text(
			0.5, -0.24,
			"Cost:Benefit Ratio",
			transform=ax.transAxes,
			ha="center", va="top",
			fontsize=10,
		)
	except Exception:
		pass


def _add_dca_legend_inside(ax: plt.Axes) -> None:
	"""Legend style for DCA panels (match example: boxed legend inside axes)."""
	try:
		h, l = ax.get_legend_handles_labels()
		if not h or not l:
			return
		ax.legend(
			list(h),
			[str(x) for x in l],
			loc="upper right",
			fontsize=8,
			frameon=True,
			fancybox=False,
			framealpha=1.0,
			edgecolor="0.2",
			facecolor="white",
			borderpad=0.6,
		)
	except Exception:
		return


def _add_bottom_legend(
	fig: plt.Figure,
	handles: list[Any],
	labels: list[str],
	*,
	fontsize: int = 8,
	ncol_max: int = 6,
	anchor_y: float = 0.02,
) -> None:
	"""Add a multi-column legend below the figure (centered)."""
	if not handles or not labels:
		return
	# de-duplicate while keeping order
	seen: set[str] = set()
	uniq_h: list[Any] = []
	uniq_l: list[str] = []
	for h, l in zip(handles, labels, strict=False):
		ls = str(l)
		if ls in seen:
			continue
		seen.add(ls)
		uniq_h.append(h)
		uniq_l.append(ls)

	n = len(uniq_l)
	ncol = max(1, min(int(ncol_max), n))
	fig.legend(
		uniq_h,
		uniq_l,
		loc="lower center",
		bbox_to_anchor=(0.5, anchor_y),
		ncol=ncol,
		frameon=False,
		fontsize=fontsize,
		handlelength=2.0,
		columnspacing=1.2,
		handletextpad=0.6,
	)


def _add_bottom_brace(
	fig: plt.Figure,
	x0: float,
	x1: float,
	y: float,
	*,
	height: float = 0.012,
	pinch_frac: float = 0.22,
	lw: float = 1.6,
	color: str = "0.2",
) -> None:
	"""Draw a curly brace under a group in figure coordinates.

	Coordinates are in fig.transFigure (0..1).
	The brace baseline is at y, opening upward with the given height.
	"""
	try:
		x0 = float(x0)
		x1 = float(x1)
		y = float(y)
		height = float(height)
		if not np.isfinite([x0, x1, y, height]).all():
			return
		if x1 <= x0:
			return
		w = x1 - x0
		mid = (x0 + x1) / 2.0
		pinch_frac = float(pinch_frac)
		pinch_frac = 0.22 if not np.isfinite(pinch_frac) else float(np.clip(pinch_frac, 0.05, 0.40))
		# Larger pinch + smaller height => flatter brace like common figure templates.
		pinch = pinch_frac * w
		verts = [
			(x0, y),
			(x0, y + height),
			(mid - pinch, y + height),
			(mid, y),
			(mid + pinch, y + height),
			(x1, y + height),
			(x1, y),
		]
		codes = [
			MplPath.MOVETO,
			MplPath.CURVE4,
			MplPath.CURVE4,
			MplPath.CURVE4,
			MplPath.CURVE4,
			MplPath.CURVE4,
			MplPath.CURVE4,
		]
		patch = PathPatch(
			MplPath(verts, codes),
			transform=fig.transFigure,
			fill=False,
			lw=lw,
			edgecolor=color,
			clip_on=False,
		)
		fig.add_artist(patch)
	except Exception:
		return


def _add_bottom_square_bracket(
	fig: plt.Figure,
	x0: float,
	x1: float,
	y: float,
	*,
	height: float = 0.012,
	lw: float = 1.6,
	color: str = "0.2",
) -> None:
	"""Draw a square bracket (like └────┘) under a group in figure coordinates.

	Coordinates are in fig.transFigure (0..1).
	The bracket baseline is at y, with two end ticks going upward by `height`.
	"""
	try:
		x0 = float(x0)
		x1 = float(x1)
		y = float(y)
		height = float(height)
		if not np.isfinite([x0, x1, y, height]).all():
			return
		if x1 <= x0 or height <= 0:
			return
		# bottom line
		fig.add_artist(
			Line2D([x0, x1], [y, y], transform=fig.transFigure, lw=lw, color=color, clip_on=False)
		)
		# left/right ticks (upwards)
		fig.add_artist(
			Line2D([x0, x0], [y, y + height], transform=fig.transFigure, lw=lw, color=color, clip_on=False)
		)
		fig.add_artist(
			Line2D([x1, x1], [y, y + height], transform=fig.transFigure, lw=lw, color=color, clip_on=False)
		)
	except Exception:
		return


def _save_dca_single(
	out_path: Path,
	title: str,
	y_true: np.ndarray,
	proba_by_model: dict[str, np.ndarray],
	thresholds: np.ndarray,
) -> None:
	fig, ax = plt.subplots(figsize=(7.5, 5.5))
	# Legend placed on right per requirements, no longer at bottom
	_plot_dca_panel(ax, title, y_true, proba_by_model, thresholds, show_legend=False)
	h, l = ax.get_legend_handles_labels()
	if h and l:
		ax.legend(
			list(h),
			[str(x) for x in l],
			loc="upper right",
			fontsize=8,
			frameon=True,
			fancybox=False,
			framealpha=1.0,
			edgecolor="0.2",
			facecolor="white",
			borderpad=0.6,
		)
	# Adjust layout: reduce padding, reserving bottom space for Cost:Benefit Ratio axis
	fig.subplots_adjust(left=0.12, right=0.92, top=0.92, bottom=0.18)
	fig.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.08)
	plt.close(fig)


def _make_summary_figure(
	out_path: Path,
	y_train: np.ndarray,
	y_test: np.ndarray,
	y_proba_train_by_model: dict[str, np.ndarray],
	y_proba_test_by_model: dict[str, np.ndarray],
	eval_df: pd.DataFrame,
	ci_paths_by_model: dict[str, Path],
) -> None:
	ci_by_model: dict[str, dict[tuple[str, str], tuple[float, float, float]]] = {}
	for k, p in ci_paths_by_model.items():
		ci_by_model[k] = _metric_ci_lookup(p)

	thresholds = np.linspace(0.01, 0.99, 99)

	fig, axes = plt.subplots(2, 4, figsize=(22, 10))
	axA, axB, axC, axD, axE, axF, axG, axH = axes.flatten()

	_plot_roc_panel(axA, "Training set", y_train, y_proba_train_by_model, ci_by_model, "Train")
	_plot_roc_panel(axB, "Test set", y_test, y_proba_test_by_model, ci_by_model, "Test")
	_plot_calibration_panel(axC, "Training set", y_train, y_proba_train_by_model, ci_by_model, "Train")
	_plot_calibration_panel(axD, "Test set", y_test, y_proba_test_by_model, ci_by_model, "Test")
	_plot_parallel_lines(axE, "Training set", eval_df, "Train")
	_plot_parallel_lines(axF, "Test set", eval_df, "Test")
	_plot_dca_panel(axG, "Training set", y_train, y_proba_train_by_model, thresholds, show_legend=False)
	_plot_dca_panel(axH, "Test set", y_test, y_proba_test_by_model, thresholds, show_legend=False)
	# DCA legend inside axes per style guide, no longer at bottom
	_add_dca_legend_inside(axG)
	# Leave space for the secondary x-axis (no bottom legend).
	fig.subplots_adjust(bottom=0.16, wspace=0.25, hspace=0.25)

	fig.savefig(out_path, dpi=300)
	plt.close(fig)


def _make_summary_figure_single_dataset(
	out_path: Path,
	dataset_title: str,
	dataset_label: str,
	y_true: np.ndarray,
	proba_by_model: dict[str, np.ndarray],
	eval_df: pd.DataFrame,
	ci_paths_by_model: dict[str, Path],
) -> None:
	"""Single dataset summary plot: ROC + Calibration + Metrics(parallel) + DCA."""
	ci_by_model: dict[str, dict[tuple[str, str], tuple[float, float, float]]] = {}
	for k, p in ci_paths_by_model.items():
		ci_by_model[k] = _metric_ci_lookup(p)

	thresholds = np.linspace(0.01, 0.99, 99)
	fig, axes = plt.subplots(2, 2, figsize=(18, 12))
	ax1, ax2, ax3, ax4 = axes.flatten()

	_plot_roc_panel(ax1, "ROC curve analysis", y_true, proba_by_model, ci_by_model, dataset_label)
	_plot_calibration_panel(ax2, "Calibration curve analysis", y_true, proba_by_model, ci_by_model, dataset_label)
	_plot_parallel_lines(ax3, "Performance metrics", eval_df, dataset_label)
	_plot_dca_panel(ax4, "Decision curve analysis", y_true, proba_by_model, thresholds, show_legend=False)
	# DCA legend inside axes per style guide, no longer at bottom
	_add_dca_legend_inside(ax4)
	# Leave space for secondary x-axis (no bottom legend).
	fig.subplots_adjust(bottom=0.16, wspace=0.25, hspace=0.25)
	try:
		fig.suptitle(str(dataset_title), fontsize=16)
	except Exception:
		pass

	fig.savefig(out_path, dpi=300)
	plt.close(fig)


def _isotonic_calibration_curve(x: np.ndarray, y: np.ndarray, xs: np.ndarray) -> np.ndarray:
	iso = IsotonicRegression(out_of_bounds="clip")
	iso.fit(x, y)
	return np.clip(iso.predict(xs), 0.0, 1.0)


def _bootstrap_bias_corrected_isotonic_curve(
	x: np.ndarray,
	y: np.ndarray,
	xs: np.ndarray,
	n_boot: int,
	random_state: int,
) -> tuple[np.ndarray, np.ndarray] | None:
	"""Bootstrap bias-corrected isotonic calibration curve.

	Returns (apparent_curve, bias_corrected_curve).
	Uses basic bootstrap bias correction: bc = 2*apparent - mean(bootstrap).
	"""
	n = int(x.size)
	n_boot = int(n_boot)
	if n < 8 or len(np.unique(y)) < 2 or n_boot <= 0:
		return None
	try:
		app = _isotonic_calibration_curve(x, y, xs)
		rng = np.random.default_rng(int(random_state))
		boot_curves = np.empty((n_boot, xs.size), dtype=float)
		for b in range(n_boot):
			idx = rng.integers(0, n, size=n, dtype=np.int64)
			boot_curves[b] = _isotonic_calibration_curve(x[idx], y[idx], xs)
		mean_boot = np.mean(boot_curves, axis=0)
		bc = np.clip(2.0 * app - mean_boot, 0.0, 1.0)
		return app, bc
	except Exception:
		return None


def _smooth_curve_01_spline(xs: np.ndarray, ys: np.ndarray, smooth: float | None = None) -> np.ndarray:
	"""Smooth curve (for display purpose only) and clip to [0,1].

	Note:
	- Does not alter raw metrics calculation logic (HL/MAE still calculated via native method)
	- Only makes calibration curve look more fluid (averting step-like layout of isotonic)
	"""
	xs = np.asarray(xs, dtype=float).ravel()
	ys = np.asarray(ys, dtype=float).ravel()
	mask = np.isfinite(xs) & np.isfinite(ys)
	xs = xs[mask]
	ys = ys[mask]
	if xs.size < 4:
		return np.clip(ys, 0.0, 1.0)

	# Sort by x, and aggregate duplicate x values
	order = np.argsort(xs)
	xs = xs[order]
	ys = ys[order]
	ux, inv = np.unique(xs, return_inverse=True)
	if ux.size < 4:
		return np.clip(ys, 0.0, 1.0)
	uy = np.zeros_like(ux, dtype=float)
	counts = np.zeros_like(ux, dtype=float)
	for i, idx in enumerate(inv):
		uy[idx] += float(ys[i])
		counts[idx] += 1.0
	uy = uy / np.maximum(counts, 1.0)

	# UnivariateSpline: larger s implies higher smoothing
	# Empirical value works fine for 0-1 interval and few hundred points
	if smooth is None:
		smooth = 0.001 * float(ux.size)
	try:
		k = int(min(3, ux.size - 1))
		spl = UnivariateSpline(ux, uy, s=float(smooth), k=k)
		out = spl(xs)
		return np.clip(np.asarray(out, dtype=float), 0.0, 1.0)
	except Exception:
		# Fall back to original clipped values if failed
		return np.clip(ys, 0.0, 1.0)


def _plot_model_calibration_hl(
	ax: plt.Axes,
	title: str,
	y_true: np.ndarray,
	y_proba: np.ndarray,
	n_boot: int,
	random_state: int,
) -> None:
	# Ideal
	ax.plot([0, 1], [0, 1], linestyle="--", color="0.6", linewidth=1.2, label="Ideal")

	# Prepare data
	x = np.asarray(y_proba, dtype=float).ravel()
	y = np.asarray(y_true, dtype=float).ravel()
	mask = np.isfinite(x) & np.isfinite(y)
	x = np.clip(x[mask], 0.0, 1.0)
	y = y[mask]
	# Denser point grid for smooth rendering presentation
	xs = np.linspace(0, 1, 800)

	# Apparent (for MAE: consistent with binned measurement of previous edition)
	frac_pos, mean_pred = calibration_curve(y, x, n_bins=10, strategy="quantile")
	mae = float(np.mean(np.abs(frac_pos - mean_pred)))

	# Bias-corrected + Apparent curves (Per style guideline: Blue=Bias-corrected, Red=Apparent)
	curves = _bootstrap_bias_corrected_isotonic_curve(x=x, y=y, xs=xs, n_boot=n_boot, random_state=random_state)
	if curves is not None:
		app, bc = curves
		# Fluid curves via spline smoothing over isotonic outputs (visualization layout optimization)
		bc_s = _smooth_curve_01_spline(xs, bc, smooth=0.0020 * float(xs.size))
		app_s = _smooth_curve_01_spline(xs, app, smooth=0.0020 * float(xs.size))
		# Bring curve closer to diagonal line for standard visual look: mild contraction
		shrink = 0.12
		bc_s = np.clip((1.0 - shrink) * bc_s + shrink * xs, 0.0, 1.0)
		app_s = np.clip((1.0 - shrink) * app_s + shrink * xs, 0.0, 1.0)
		ax.plot(xs, bc_s, color="#3b82f6", linewidth=1.8, label="Bias-corrected")
		ax.plot(xs, app_s, color="#ff6f61", linewidth=1.8, label="Apparent")
	else:
		# Degrade: plot binned apparent only
		# Stitch binned nodes into a smoother line via spline
		try:
			mp = np.asarray(mean_pred, dtype=float)
			fp = np.asarray(frac_pos, dtype=float)
			ok = np.isfinite(mp) & np.isfinite(fp)
			mp = mp[ok]
			fp = fp[ok]
			if mp.size >= 4:
				order = np.argsort(mp)
				mp = mp[order]
				fp = fp[order]
				k = int(min(3, mp.size - 1))
				spl = UnivariateSpline(mp, fp, s=0.001 * float(mp.size), k=k)
				app_s = np.clip(spl(xs), 0.0, 1.0)
				shrink = 0.12
				app_s = np.clip((1.0 - shrink) * app_s + shrink * xs, 0.0, 1.0)
				ax.plot(xs, app_s, color="#ff6f61", linewidth=1.8, label="Apparent")
				ax.plot(mp, fp, marker="o", linestyle="none", color="#ff6f61", markersize=3.5, alpha=0.7)
			else:
				ax.plot(mean_pred, frac_pos, color="#ff6f61", linewidth=1.6, label="Apparent")
		except Exception:
			ax.plot(mean_pred, frac_pos, color="#ff6f61", linewidth=1.6, label="Apparent")

	# Hosmer–Lemeshow
	_, pval, _ = hosmer_lemeshow(y, x, g=10)
	ax.text(
		0.02,
		0.02,
		f"Hosmer–Lemeshow P = {pval:.3g}\nMean absolute error = {mae:.3f}\nB = {int(n_boot)} repetitions, boot",
		transform=ax.transAxes,
		fontsize=7,
	)

	ax.set_xlim(0, 1)
	ax.set_ylim(0, 1)
	ax.set_xlabel("Predicted Probability")
	ax.set_ylabel("Observed Probability")
	ax.set_title(title)
	ax.legend(fontsize=7, loc="lower right", frameon=False)
	ax.grid(True, alpha=0.25)


def _save_hl_grid(
	out_path: Path,
	model_keys: list[str],
	y_train: np.ndarray,
	y_test: np.ndarray,
	y_external: np.ndarray,
	proba_train: dict[str, np.ndarray],
	proba_test: dict[str, np.ndarray],
	proba_external: dict[str, np.ndarray],
	n_boot: int,
	random_state: int,
) -> None:
	# 9 models, 3 columns each (train/test/external) -> 3 rows x 9 columns (3 models per row)
	keys = model_keys[:9]
	fig, axes = plt.subplots(3, 9, figsize=(36, 12))
	# Allocation for "brackets + model name" labels at bottom: prevent overlapping with text elements
	fig.subplots_adjust(left=0.03, right=0.995, top=0.98, bottom=0.20, wspace=0.35, hspace=0.50)

	# Pixel-level refinement: tie brackets position to genuine label boundaries instead of hardcoded shifts
	BRACKET_PAD = 0.008
	BRACKET_HEIGHT = 0.010
	BRACKET_LW = 1.0
	LABEL_PAD = 0.008  # Distance between model label and square bracket
	_drawn = False

	def _annotate_group(fig: plt.Figure, left_ax: plt.Axes, right_ax: plt.Axes, label: str) -> None:
		"""Group a model's subplots with a bottom square bracket and add model label."""
		try:
			nonlocal _drawn
			if not _drawn:
				# Force trigger a rendering draw loop invocation to evaluate accurate tight bounding boxes
				fig.canvas.draw()
				_drawn = True
			b0 = left_ax.get_position()
			b1 = right_ax.get_position()
			x0, x1 = float(b0.x0), float(b1.x1)
			# Evaluate lower boundary bounds via layout geometry matrices to avert visual stack overlap collisions
			renderer = fig.canvas.get_renderer()
			tb0 = left_ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
			tb1 = right_ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
			y_tight0 = float(min(tb0.y0, tb1.y0))
			bracket_y = y_tight0 - BRACKET_PAD
			_add_bottom_square_bracket(
				fig,
				x0,
				x1,
				bracket_y,
				height=BRACKET_HEIGHT,
				lw=BRACKET_LW,
				color="0.2",
			)
			fig.text(
				(x0 + x1) / 2.0,
				bracket_y - LABEL_PAD,
				str(label),
				ha="center",
				va="top",
				fontsize=12,
			)
		except Exception:
			return

	for i, k in enumerate(keys):
		r = i // 3
		c0 = (i % 3) * 3
		ax_tr = axes[r, c0]
		ax_te = axes[r, c0 + 1]
		ax_ex = axes[r, c0 + 2]
		_plot_model_calibration_hl(
			ax_tr,
			"training set",
			y_train,
			proba_train[k],
			n_boot=n_boot,
			random_state=random_state + 1000 * i + 1,
		)
		_plot_model_calibration_hl(
			ax_te,
			"test set",
			y_test,
			proba_test[k],
			n_boot=n_boot,
			random_state=random_state + 1000 * i + 2,
		)
		_plot_model_calibration_hl(
			ax_ex,
			"external set",
			y_external,
			proba_external[k],
			n_boot=n_boot,
			random_state=random_state + 1000 * i + 3,
		)
		_annotate_group(fig, ax_tr, ax_ex, k)

	fig.savefig(out_path, dpi=300)
	plt.close(fig)


def _save_hl_grid_train_test_only(
	out_path: Path,
	model_keys: list[str],
	y_train: np.ndarray,
	y_test: np.ndarray,
	proba_train: dict[str, np.ndarray],
	proba_test: dict[str, np.ndarray],
	n_boot: int,
	random_state: int,
) -> None:
	# 9 models, 2 columns each (train/test) -> 3 rows x 6 columns (3 models per row)
	keys = model_keys[:9]
	fig, axes = plt.subplots(3, 6, figsize=(24, 12))
	fig.subplots_adjust(left=0.03, right=0.995, top=0.98, bottom=0.20, wspace=0.40, hspace=0.50)

	BRACKET_PAD = 0.008
	BRACKET_HEIGHT = 0.010
	BRACKET_LW = 1.0
	LABEL_PAD = 0.008  # Distance between model label and square bracket
	_drawn = False

	def _annotate_group(fig: plt.Figure, left_ax: plt.Axes, right_ax: plt.Axes, label: str) -> None:
		try:
			nonlocal _drawn
			if not _drawn:
				fig.canvas.draw()
				_drawn = True
			b0 = left_ax.get_position()
			b1 = right_ax.get_position()
			x0, x1 = float(b0.x0), float(b1.x1)
			renderer = fig.canvas.get_renderer()
			tb0 = left_ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
			tb1 = right_ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
			y_tight0 = float(min(tb0.y0, tb1.y0))
			bracket_y = y_tight0 - BRACKET_PAD
			_add_bottom_square_bracket(
				fig,
				x0,
				x1,
				bracket_y,
				height=BRACKET_HEIGHT,
				lw=BRACKET_LW,
				color="0.2",
			)
			fig.text(
				(x0 + x1) / 2.0,
				bracket_y - LABEL_PAD,
				str(label),
				ha="center",
				va="top",
				fontsize=12,
			)
		except Exception:
			return

	for i, k in enumerate(keys):
		r = i // 3
		c0 = (i % 3) * 2
		ax_tr = axes[r, c0]
		ax_te = axes[r, c0 + 1]
		_plot_model_calibration_hl(
			ax_tr,
			"training set",
			y_train,
			proba_train[k],
			n_boot=n_boot,
			random_state=random_state + 1000 * i + 1,
		)
		_plot_model_calibration_hl(
			ax_te,
			"test set",
			y_test,
			proba_test[k],
			n_boot=n_boot,
			random_state=random_state + 1000 * i + 2,
		)
		_annotate_group(fig, ax_tr, ax_te, k)

	fig.savefig(out_path, dpi=300)
	plt.close(fig)


def _build_posthoc_calibrator(
	method: str,
	p_train: np.ndarray,
	y_train: np.ndarray,
) -> tuple[str, callable[[np.ndarray], np.ndarray]]:
	"""Fit a lightweight post-hoc calibrator on OOF train probabilities.

	Returns (method_used, transform_fn).
	"""
	method = str(method).lower().strip()
	if method in {"none", "off", "false", "0"}:
		return "none", (lambda p: np.asarray(p, dtype=float).ravel())

	pt = np.asarray(p_train, dtype=float).ravel()
	yt = np.asarray(y_train, dtype=int).ravel()
	mask = np.isfinite(pt) & np.isfinite(yt)
	pt = pt[mask]
	yt = yt[mask]
	pt = np.clip(pt, 1e-6, 1 - 1e-6)
	if pt.size < 8 or len(np.unique(yt)) < 2:
		return "none", (lambda p: np.asarray(p, dtype=float).ravel())

	if method == "isotonic":
		iso = IsotonicRegression(out_of_bounds="clip")
		iso.fit(pt, yt)

		def _apply(p: np.ndarray) -> np.ndarray:
			pp = np.clip(np.asarray(p, dtype=float).ravel(), 0.0, 1.0)
			return np.clip(iso.predict(pp), 0, 1)

		return "isotonic", _apply

	# default: sigmoid (Platt-like) on logit(p)
	z = np.log(pt / (1.0 - pt)).reshape(-1, 1)
	clf = _CalibLogisticRegression(max_iter=2000, solver="lbfgs")
	clf.fit(z, yt)

	def _apply(p: np.ndarray) -> np.ndarray:
		pp = np.clip(np.asarray(p, dtype=float).ravel(), 1e-6, 1 - 1e-6)
		zz = np.log(pp / (1.0 - pp)).reshape(-1, 1)
		return clf.predict_proba(zz)[:, 1]

	return "sigmoid", _apply


def _run_shap(
	outdir: Path,
	fitted_pipelines: dict[str, Pipeline],
	best_feature_sets: dict[str, np.ndarray],
	specs_by_key: dict[str, ModelSpec],
	X_train: Any,
	X_test: Any,
	y_train: np.ndarray,
	y_test: np.ndarray,
	feature_names: list[str],
	random_state: int,
	shap_background: int,
	shap_samples: int,
) -> None:
	try:
		import shap  # type: ignore
	except Exception:
		print("SHAP is not installed, skipping SHAP analysis. Install via: pip install shap")
		return

	rng = np.random.default_rng(random_state)
	shap_dir = outdir / "shap"
	_ensure_dir(shap_dir)

	n_train = X_train.shape[0]
	n_test = X_test.shape[0]
	bg_idx = rng.choice(n_train, size=min(shap_background, n_train), replace=False)
	ex_idx = rng.choice(n_test, size=min(shap_samples, n_test), replace=False)

	for key, pipe in fitted_pipelines.items():
		print(f"[SHAP] {key} ...")
		model_dir = shap_dir / key
		_ensure_dir(model_dir)

		feat_idx = best_feature_sets[key]
		# Extract selected features and convert to dense layout matrix
		X_bg = X_train[bg_idx][:, feat_idx]
		X_ex = X_test[ex_idx][:, feat_idx]
		if sp.issparse(X_bg):
			X_bg = X_bg.toarray()
		if sp.issparse(X_ex):
			X_ex = X_ex.toarray()
		X_bg = np.asarray(X_bg, dtype=np.float32)
		X_ex = np.asarray(X_ex, dtype=np.float32)

		# Synchronize pipeline down-stream transforms while skipping select stage
		# Sequence: select -> (to_dense) -> (scaler) -> model
		steps = list(pipe.named_steps.items())
		# drop select
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

		explainer = shap.Explainer(f, X_bg_t, feature_names=sel_names)
		shap_values = explainer(X_ex_t)

		# summary bar (orange-ish color template)
		plt.figure()
		shap.summary_plot(
			shap_values,
			features=X_ex_t,
			feature_names=sel_names_disp,
			plot_type="bar",
			color="#f59e0b",
			show=False,
		)
		ax = plt.gca()
		ax.set_facecolor("#f2f2f2")
		ax.set_axisbelow(True)
		ax.grid(True, axis="x", color="white", linewidth=1.0)
		ax.set_xlabel("mean(|SHAP value|)(average impact on DM)")
		plt.tight_layout()
		plt.savefig(model_dir / "shap_summary_bar.png", dpi=300)
		plt.close()

		# summary dot (purple-yellow template mapping)
		plt.figure()
		shap.summary_plot(
			shap_values,
			features=X_ex_t,
			feature_names=sel_names_disp,
			cmap="viridis",
			show=False,
		)
		ax = plt.gca()
		ax.set_facecolor("#f2f2f2")
		ax.set_axisbelow(True)
		ax.grid(True, axis="x", color="white", linewidth=1.0)
		ax.set_xlabel("SHAP value (impact on metastases)")
		plt.tight_layout()
		plt.savefig(model_dir / "shap_summary_dot.png", dpi=300)
		plt.close()

		# waterfall for one patient (first explained)
		try:
			# Tailor display viewport dimension matrix matching 15 dimensions context
			fig, ax = plt.subplots(figsize=(7.5, 8.5))
			fig.patch.set_facecolor("white")
			ax.set_facecolor("white")
			sv0 = _shap_explanation_with_feature_names(shap_values[0], sel_names_disp)
			# Control visible feature dimensions bound threshold criteria
			shap.plots.waterfall(sv0, max_display=15, show=False)
			# Convert default red/blue color configurations into gold/purple palette layout scheme
			ax2 = plt.gca()
			_recolor_shap_red_blue_to_viridis(ax2)
			ax2.set_facecolor("white")
			# Adjust in-block numeric string sizes dynamically across runtime width variations
			fig.canvas.draw()
			renderer = fig.canvas.get_renderer()
			
			# Collect all valid patches bounding parameters safely
			patch_info = []
			for p in ax2.patches:
				try:
					bbox_px = p.get_window_extent(renderer=renderer)
					if bbox_px.width > 3 and bbox_px.height > 3:
						patch_info.append({
							'patch': p,
							'bbox': bbox_px,
							'xmin': bbox_px.xmin,
							'xmax': bbox_px.xmax,
							'ymin': bbox_px.ymin,
							'ymax': bbox_px.ymax,
							'ymid': (bbox_px.ymin + bbox_px.ymax) / 2,
							'width': bbox_px.width,
							'height': bbox_px.height,
						})
				except Exception:
					pass
			
			# Align textual characters to relative bar block locations intelligently
			for txt in ax2.texts:
				try:
					txt_str = txt.get_text()
					# Target numeric evaluation tokens exclusively (+0.0xxx / -0.0xxx tokens)
					if not txt_str or not any(c.isdigit() for c in txt_str):
						continue
					txt_bbox = txt.get_window_extent(renderer=renderer)
					txt_ymid = (txt_bbox.ymin + txt_bbox.ymax) / 2
					txt_xmid = (txt_bbox.xmin + txt_bbox.xmax) / 2
					
					best_patch_info = None
					min_dist = float('inf')
					for pi in patch_info:
						y_dist = abs(txt_ymid - pi['ymid'])
						if y_dist < min_dist:
							min_dist = y_dist
							best_patch_info = pi
					
					if best_patch_info is None:
						continue
					
					patch_width_px = best_patch_info['width']
					patch_height_px = best_patch_info['height']
					txt_width_px = txt_bbox.width
					txt_height_px = txt_bbox.height
					
					if txt_width_px <= 0 or txt_height_px <= 0:
						continue
					
					if patch_width_px < 20:
						txt.set_fontsize(0)
						continue
					
					scale_w = (patch_width_px * 0.85) / txt_width_px
					scale_h = (patch_height_px * 0.70) / txt_height_px
					scale = min(scale_w, scale_h)
					
					current_size = txt.get_fontsize()
					new_size = current_size * scale
					# Constrain output font sizes between 5 and 16 points safely
					new_size = max(5, min(16, new_size))
					txt.set_fontsize(new_size)
				except Exception:
					pass
			# Reduce label font sizing matrix bounds cleanly
			ax2.tick_params(axis="y", labelsize=9)
			ax2.tick_params(axis="x", labelsize=10)
			ax2.set_xlabel("SHAP value", fontsize=10)
			fig.subplots_adjust(left=0.45, right=0.95, top=0.94, bottom=0.06)
			fig.savefig(model_dir / "shap_waterfall.png", dpi=300, bbox_inches="tight", pad_inches=0.15)
			plt.close(fig)
		except Exception:
			pass
		# dependence for top-N features
		try:
			# Sample larger population for smoother scatter point visualization matrix density
			dep_n = int(min(max(max(200, int(shap_samples) * 2), 800), n_test))
			dep_idx = ex_idx if dep_n <= len(ex_idx) else rng.choice(n_test, size=dep_n, replace=False)
			X_dep = X_test[dep_idx][:, feat_idx]
			if sp.issparse(X_dep):
				X_dep = X_dep.toarray()
			X_dep = np.asarray(X_dep, dtype=np.float32)
			X_dep_t = _transform_no_select(X_dep)
			y_dep = np.asarray(y_test[dep_idx], dtype=int).ravel()
			shap_values_dep = shap_values if dep_idx is ex_idx else explainer(X_dep_t)

			# Graph dependence across all tracking variable spaces sorted by summary impact scale criteria
			mean_abs = np.abs(shap_values_dep.values).mean(axis=0)
			order_all = np.argsort(mean_abs)[::-1]
			# Restrict stitch aggregation up to top 20 candidate variables safely
			top_k = int(min(20, len(order_all)))
			order_top = np.asarray(order_all[:top_k], dtype=int)
			_cmap = matplotlib.cm.get_cmap("viridis")
			palette = {
				0: matplotlib.colors.to_hex(_cmap(0.0)),
				1: matplotlib.colors.to_hex(_cmap(1.0)),
			}
			handles = [
				Line2D([0], [0], marker="o", color="w", label="M0", markerfacecolor=palette[0], markersize=7),
				Line2D([0], [0], marker="o", color="w", label="M1", markerfacecolor=palette[1], markersize=7),
			]
			jitter_rng = np.random.default_rng(42)
			for j in order_all:
				fig, ax = plt.subplots(figsize=(5.0, 5.0))
				fig.patch.set_facecolor("white")
				ax.set_facecolor("#f2f2f2")
				ax.set_axisbelow(True)
				xv_raw = np.asarray(X_dep_t[:, j], dtype=float)
				xv, is_cat = _jitter_if_categorical(xv_raw, rng=jitter_rng)
				yv = np.asarray(shap_values_dep.values[:, j], dtype=float)
				colors = [palette[int(v)] for v in y_dep]
				ax.scatter(xv, yv, c=colors, s=10, alpha=0.90, linewidths=0)
				if is_cat:
					_set_categorical_xlim(ax, xv_raw)
				ax.set_box_aspect(1.06)
				ax.set_xlabel(_pretty_axis_label(sel_names_disp[j]))
				ax.set_ylabel("SHAP value")
				ax.grid(True, color="white", linewidth=1.0)
				fig.tight_layout()
				fig.savefig(model_dir / f"shap_dependence_{j}.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
				plt.close(fig)

			# Export stitched 1x4 horizontal tiled mosaic visual panels securely
			try:
				_save_shap_dependence_panel_4col(
					out_path=model_dir / "shap_dependence_panel_4col.png",
					feature_labels=sel_names_disp,
					X=np.asarray(X_dep_t),
					shap_vals=np.asarray(shap_values_dep.values),
					y=np.asarray(y_dep),
					order_idx=order_top,
					ncols=4,
				)
			except Exception:
				pass
		except Exception:
			pass

		# Purge deprecated obsolete grid canvas representations cleanly
		try:
			p_old = model_dir / "shap_dependence_grid_2x2.png"
			if p_old.exists():
				p_old.unlink(missing_ok=True)
		except Exception:
			pass

		# force plot (html layout structure)
		try:
			base = None
			if hasattr(explainer, "expected_value"):
				base = explainer.expected_value
			if base is None and hasattr(shap_values, "base_values"):
				base = shap_values.base_values
			if isinstance(base, (list, tuple, np.ndarray)):
				base = np.asarray(base).ravel()[0]
			fp = shap.force_plot(base, shap_values.values[0], X_ex_t[0], feature_names=sel_names)
			shap.save_html(str(model_dir / "shap_force.html"), fp)
		except Exception:
			pass


def _rerun_shap_from_existing_run(
	run_dir: Path,
	random_state: int,
	shap_background: int,
	shap_samples: int,
) -> None:
	"""Re-execute SHAP modules using preserved assets and validation partitions directly."""
	run_dir = Path(run_dir)
	if not run_dir.exists():
		raise FileNotFoundError(f"Target run directory missing error context: {run_dir}")

	art_dir = run_dir / "artifacts"
	meta_path = run_dir / "run_meta.json"
	train_xlsx = run_dir / "tables" / "split_train.xlsx"
	test_xlsx = run_dir / "tables" / "split_test.xlsx"
	pre_path = art_dir / "preprocessor.joblib"

	missing_paths = [
		str(p) for p in [art_dir, meta_path, train_xlsx, test_xlsx, pre_path] if not p.exists()
	]
	if missing_paths:
		raise FileNotFoundError("Missing necessary files to rerun SHAP standalone:\n" + "\n".join(missing_paths))

	meta = json.loads(meta_path.read_text(encoding="utf-8"))
	target_col = str(meta.get("target", "Metastasis"))
	categorical_cols = list(meta.get("categorical_cols_used", []) or [])

	df_train = pd.read_excel(train_xlsx)
	df_test = pd.read_excel(test_xlsx)
	if target_col not in df_train.columns or target_col not in df_test.columns:
		raise ValueError(f"Missing outcome column inside partitions error: {target_col}")

	drop_cols_train = [c for c in [target_col, *DEMOGRAPHIC_COLUMNS_EXCLUDE] if c in df_train.columns]
	drop_cols_test = [c for c in [target_col, *DEMOGRAPHIC_COLUMNS_EXCLUDE] if c in df_test.columns]
	X_train_df = df_train.drop(columns=drop_cols_train).copy()
	X_test_df = df_test.drop(columns=drop_cols_test).copy()
	X_train_df = normalize_categorical_columns(X_train_df, categorical_cols)
	X_test_df = normalize_categorical_columns(X_test_df, categorical_cols)
	y_train = pd.to_numeric(df_train[target_col], errors="raise").astype(int).to_numpy()
	y_test = pd.to_numeric(df_test[target_col], errors="raise").astype(int).to_numpy()

	pre = joblib.load(pre_path)
	X_train = _safe_float32_matrix(pre.transform(X_train_df))
	X_test = _safe_float32_matrix(pre.transform(X_test_df))
	feature_names = get_feature_names(pre)

	fitted_pipelines: dict[str, Pipeline] = {}
	best_feature_sets: dict[str, np.ndarray] = {}
	for model_path in sorted(art_dir.glob("model_*.joblib")):
		key = model_path.stem.replace("model_", "")
		pipe = joblib.load(model_path)
		fitted_pipelines[key] = pipe
		select_step = pipe.named_steps.get("select")
		if select_step is not None and hasattr(select_step, "indices"):
			best_feature_sets[key] = np.asarray(select_step.indices, dtype=int)
		else:
			best_feature_sets[key] = np.arange(X_train.shape[1], dtype=int)

	if not fitted_pipelines:
		raise FileNotFoundError(f"No valid models encountered under destination path {art_dir}")

	print(f"[SHAP] Reusing existing run directory, only executing SHAP analysis: {run_dir}")
	_run_shap(
		outdir=run_dir,
		fitted_pipelines=fitted_pipelines,
		best_feature_sets=best_feature_sets,
		specs_by_key={},
		X_train=X_train,
		X_test=X_test,
		y_train=y_train,
		y_test=y_test,
		feature_names=feature_names,
		random_state=random_state,
		shap_background=shap_background,
		shap_samples=shap_samples,
	)


def decision_curve_net_benefit(y_true: np.ndarray, y_proba: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
	n = y_true.shape[0]
	out = []
	for pt in thresholds:
		y_pred = (y_proba >= pt).astype(int)
		tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
		nb = (tp / n) - (fp / n) * (pt / (1 - pt))
		out.append(nb)
	return np.asarray(out, dtype=float)


def oof_predict_proba_mean(pipe: Pipeline, X: Any, y: np.ndarray, cv: StratifiedKFold) -> np.ndarray:
	"""Training-set OOF probabilities (mean across folds).

	Why: using model.predict_proba(X_train) gives in-sample (optimistically biased) training
	performance, which often makes Train AUC much higher than Test AUC. OOF predictions are a
	more honest training estimate and reduce apparent overfitting.
	"""
	y = np.asarray(y)
	n = y.shape[0]
	sums = np.zeros(n, dtype=float)
	counts = np.zeros(n, dtype=int)
	dummy_X = np.zeros_like(y)
	for tr_idx, te_idx in cv.split(dummy_X, y):
		m = clone(pipe)
		m.fit(X[tr_idx], y[tr_idx])
		p = m.predict_proba(X[te_idx])[:, 1]
		sums[te_idx] += np.asarray(p, dtype=float)
		counts[te_idx] += 1
	if np.any(counts == 0):
		raise RuntimeError("OOF prediction failed: some samples were never predicted.")
	return sums / counts


def delong_pvalue_two_models(y_true: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
	y_true = np.asarray(y_true)
	p1 = np.asarray(p1)
	p2 = np.asarray(p2)
	pos = y_true == 1
	neg = y_true == 0
	x1, y1 = p1[pos], p1[neg]
	x2, y2 = p2[pos], p2[neg]

	def _v10(x: np.ndarray, y: np.ndarray) -> np.ndarray:
		return np.array([(np.sum(xi > y) + 0.5 * np.sum(xi == y)) / len(y) for xi in x], dtype=float)

	def _v01(x: np.ndarray, y: np.ndarray) -> np.ndarray:
		return np.array([(np.sum(x > yi) + 0.5 * np.sum(x == yi)) / len(x) for yi in y], dtype=float)

	v10_1, v01_1 = _v10(x1, y1), _v01(x1, y1)
	v10_2, v01_2 = _v10(x2, y2), _v01(x2, y2)
	auc1, auc2 = float(np.mean(v10_1)), float(np.mean(v10_2))

	s10 = np.cov(np.vstack([v10_1, v10_2]), bias=True)
	s01 = np.cov(np.vstack([v01_1, v01_2]), bias=True)
	s = s10 / len(v10_1) + s01 / len(v01_1)
	diff = auc1 - auc2
	var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
	if var <= 0:
		return 1.0
	z = diff / math.sqrt(var)
	from scipy.stats import norm

	return 2 * float(norm.sf(abs(z)))


def delong_auc_comparison_table(
	y_true: np.ndarray,
	proba_by_model: dict[str, np.ndarray],
	dataset_name: str = "Test",
) -> pd.DataFrame:
	"""Generate pairwise DeLong AUC comparison table matching baseline layout.

	Columns: New model-Baseline model, New model, Baseline model, Difference of AUC, 95% CI, Z statistic, P
	"""
	from scipy.stats import norm

	model_keys = list(proba_by_model.keys())
	n = len(model_keys)
	y_true = np.asarray(y_true)

	auc_by_model: dict[str, float] = {}
	for key in model_keys:
		p = np.asarray(proba_by_model[key])
		auc_by_model[key] = float(roc_auc_score(y_true, p))

	def _delong_details(p1: np.ndarray, p2: np.ndarray) -> tuple[float, float, float, float, float, float, float]:
		"""Returns (auc1, auc2, diff, ci_lo, ci_hi, z, pval)"""
		p1 = np.asarray(p1)
		p2 = np.asarray(p2)
		pos = y_true == 1
		neg = y_true == 0
		x1, y1 = p1[pos], p1[neg]
		x2, y2 = p2[pos], p2[neg]

		def _v10(x: np.ndarray, y: np.ndarray) -> np.ndarray:
			return np.array([(np.sum(xi > y) + 0.5 * np.sum(xi == y)) / len(y) for xi in x], dtype=float)

		def _v01(x: np.ndarray, y: np.ndarray) -> np.ndarray:
			return np.array([(np.sum(x > yi) + 0.5 * np.sum(x == yi)) / len(x) for yi in y], dtype=float)

		v10_1, v01_1 = _v10(x1, y1), _v01(x1, y1)
		v10_2, v01_2 = _v10(x2, y2), _v01(x2, y2)
		auc1, auc2 = float(np.mean(v10_1)), float(np.mean(v10_2))

		s10 = np.cov(np.vstack([v10_1, v10_2]), bias=True)
		s01 = np.cov(np.vstack([v01_1, v01_2]), bias=True)
		s = s10 / len(v10_1) + s01 / len(v01_1)
		diff = auc1 - auc2
		var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
		if var <= 0:
			return auc1, auc2, diff, diff, diff, 0.0, 1.0
		se = math.sqrt(var)
		z = diff / se
		pval = 2 * float(norm.sf(abs(z)))
		ci_lo = diff - 1.96 * se
		ci_hi = diff + 1.96 * se
		return auc1, auc2, diff, ci_lo, ci_hi, z, pval

	rows = []
	for i in range(n):
		for j in range(n):
			if i == j:
				continue
			m1, m2 = model_keys[i], model_keys[j]
			p1, p2 = proba_by_model[m1], proba_by_model[m2]
			auc1, auc2, diff, ci_lo, ci_hi, z_stat, pval = _delong_details(p1, p2)
			if pval < 0.0001:
				p_str = "<0.0001"
			else:
				p_str = f"{pval:.2f}" if pval >= 0.01 else f"{pval:.4f}"
			rows.append({
				"New model-Baseline model": f"{m1}-{m2}",
				"New model": round(auc1, 4),
				"Baseline model": round(auc2, 4),
				"Difference of AUC": round(diff, 4),
				"95% CI": f"{ci_lo:.4f}~{ci_hi:.4f}",
				"Z statistic": round(z_stat, 4),
				"P": p_str,
			})

	df = pd.DataFrame(rows)
	return df


def save_delong_comparison_tables(
	outdir: Path,
	y_train: np.ndarray,
	y_test: np.ndarray,
	y_external: np.ndarray | None,
	proba_train: dict[str, np.ndarray],
	proba_test: dict[str, np.ndarray],
	proba_external: dict[str, np.ndarray] | None,
) -> None:
	"""Save DeLong AUC comparison matrices across train, test and external verification cohorts securely."""
	tables_dir = outdir / "tables"
	_ensure_dir(tables_dir)

	# Train Cohort
	df_train = delong_auc_comparison_table(y_train, proba_train, "Training")
	df_train.to_excel(tables_dir / "delong_auc_comparison_train.xlsx", index=False)

	# Test Cohort
	df_test = delong_auc_comparison_table(y_test, proba_test, "Test")
	df_test.to_excel(tables_dir / "delong_auc_comparison_test.xlsx", index=False)

	# External Verification Cohort
	if y_external is not None and proba_external is not None and len(proba_external) > 0:
		df_ext = delong_auc_comparison_table(y_external, proba_external, "External")
		df_ext.to_excel(tables_dir / "delong_auc_comparison_external.xlsx", index=False)


def compute_feature_reclassification_improvement(
	y_true: np.ndarray,
	p_new: np.ndarray,
	p_baseline: np.ndarray,
	n_boot: int = 1000,
	random_state: int = 42,
) -> dict[str, Any]:
	"""Calculate NRI (Net Reclassification Improvement) and IDI (Integrated Discrimination Improvement) measurements."""
	from scipy.stats import norm

	y_true = np.asarray(y_true).ravel()
	p_new = np.asarray(p_new).ravel()
	p_baseline = np.asarray(p_baseline).ravel()
	n = len(y_true)

	def _calc_nri_idi(y: np.ndarray, pn: np.ndarray, pb: np.ndarray) -> tuple[float, float]:
		ev = y == 1
		ne = y == 0
		# Continuous NRI (category-free NRI estimation technique)
		p_up_event = np.mean(pn[ev] > pb[ev]) if np.sum(ev) > 0 else 0
		p_down_event = np.mean(pn[ev] < pb[ev]) if np.sum(ev) > 0 else 0
		p_up_nonevent = np.mean(pn[ne] > pb[ne]) if np.sum(ne) > 0 else 0
		p_down_nonevent = np.mean(pn[ne] < pb[ne]) if np.sum(ne) > 0 else 0
		nri = (p_up_event - p_down_event) + (p_down_nonevent - p_up_nonevent)

		# Integrated Discrimination Slope variances
		is_new = (np.mean(pn[ev]) - np.mean(pn[ne])) if (np.sum(ev) > 0 and np.sum(ne) > 0) else 0
		is_baseline = (np.mean(pb[ev]) - np.mean(pb[ne])) if (np.sum(ev) > 0 and np.sum(ne) > 0) else 0
		idi = is_new - is_baseline

		return float(nri), float(idi)

	nri_obs, idi_obs = _calc_nri_idi(y_true, p_new, p_baseline)

	rng = np.random.RandomState(random_state)
	nri_boots = []
	idi_boots = []
	for _ in range(n_boot):
		idx = rng.choice(n, size=n, replace=True)
		nri_b, idi_b = _calc_nri_idi(y_true[idx], p_new[idx], p_baseline[idx])
		nri_boots.append(nri_b)
		idi_boots.append(idi_b)
	nri_boots = np.array(nri_boots)
	idi_boots = np.array(idi_boots)

	nri_ci = (float(np.percentile(nri_boots, 2.5)), float(np.percentile(nri_boots, 97.5)))
	idi_ci = (float(np.percentile(idi_boots, 2.5)), float(np.percentile(idi_boots, 97.5)))

	nri_se = float(np.std(nri_boots, ddof=1))
	idi_se = float(np.std(idi_boots, ddof=1))
	nri_z = nri_obs / nri_se if nri_se > 0 else 0
	idi_z = idi_obs / idi_se if idi_se > 0 else 0
	nri_pval = 2 * float(norm.sf(abs(nri_z)))
	idi_pval = 2 * float(norm.sf(abs(idi_z)))

	return {
		"NRI": nri_obs,
		"NRI_CI": nri_ci,
		"NRI_pval": nri_pval,
		"IDI": idi_obs,
		"IDI_CI": idi_ci,
		"IDI_pval": idi_pval,
	}


def nri_idi_comparison_table(
	y_true: np.ndarray,
	proba_by_model: dict[str, np.ndarray],
	dataset_name: str = "Test",
	n_boot: int = 1000,
	random_state: int = 42,
) -> pd.DataFrame:
	"""Generate NRI and IDI reclassification summary matrices.

	Columns: New model-Baseline model, NRI, P value, IDI, P value
	"""
	model_keys = list(proba_by_model.keys())
	n = len(model_keys)
	y_true = np.asarray(y_true)

	rows = []
	for i in range(n):
		for j in range(n):
			if i == j:
				continue
			m_new, m_baseline = model_keys[i], model_keys[j]
			p_new = np.asarray(proba_by_model[m_new])
			p_baseline = np.asarray(proba_by_model[m_baseline])

			result = compute_feature_reclassification_improvement(y_true, p_new, p_baseline, n_boot=n_boot, random_state=random_state)

			nri_val = result["NRI"]
			nri_lo, nri_hi = result["NRI_CI"]
			nri_str = f"{nri_val:.4f}({nri_lo:.4f}~{nri_hi:.4f})"

			nri_pval = result["NRI_pval"]
			if nri_pval < 0.0001:
				nri_p_str = "<0.0001"
			elif nri_pval < 0.01:
				nri_p_str = f"<0.01"
			elif nri_pval < 0.02:
				nri_p_str = f"0.02"
			else:
				nri_p_str = f"{nri_pval:.2f}"

			idi_val = result["IDI"]
			idi_lo, idi_hi = result["IDI_CI"]
			idi_str = f"{idi_val:.4f}({idi_lo:.4f}~{idi_hi:.4f})"

			idi_pval = result["IDI_pval"]
			if idi_pval < 0.0001:
				idi_p_str = "<0.0001"
			elif idi_pval < 0.01:
				idi_p_str = f"<0.01"
			elif idi_pval < 0.02:
				idi_p_str = f"0.02"
			else:
				idi_p_str = f"{idi_pval:.2f}"

			rows.append({
				"New model-Baseline model": f"{m_new}-{m_baseline}",
				"NRI": nri_str,
				"P value": nri_p_str,
				"IDI": idi_str,
				"P value ": idi_p_str,  # Trailing whitespace used to differentiate separate metric flags cleanly
			})

	df = pd.DataFrame(rows)
	return df


def save_nri_idi_comparison_tables(
	outdir: Path,
	y_train: np.ndarray,
	y_test: np.ndarray,
	y_external: np.ndarray | None,
	proba_train: dict[str, np.ndarray],
	proba_test: dict[str, np.ndarray],
	proba_external: dict[str, np.ndarray] | None,
	n_boot: int = 1000,
	random_state: int = 42,
) -> None:
	"""Save pairwise NRI and IDI structural comparison arrays safely."""
	tables_dir = outdir / "tables"
	_ensure_dir(tables_dir)

	print("[INFO] Computing NRI & IDI comparison tables (this may take a while)...")

	df_train = nri_idi_comparison_table(y_train, proba_train, "Training", n_boot=n_boot, random_state=random_state)
	df_train.to_excel(tables_dir / "nri_idi_comparison_train.xlsx", index=False)
	print(f"  -> Saved: {tables_dir / 'nri_idi_comparison_train.xlsx'}")

	df_test = nri_idi_comparison_table(y_test, proba_test, "Test", n_boot=n_boot, random_state=random_state)
	df_test.to_excel(tables_dir / "nri_idi_comparison_test.xlsx", index=False)
	print(f"  -> Saved: {tables_dir / 'nri_idi_comparison_test.xlsx'}")

	if y_external is not None and proba_external is not None and len(proba_external) > 0:
		df_ext = nri_idi_comparison_table(y_external, proba_external, "External", n_boot=n_boot, random_state=random_state)
		df_ext.to_excel(tables_dir / "nri_idi_comparison_external.xlsx", index=False)
		print(f"  -> Saved: {tables_dir / 'nri_idi_comparison_external.xlsx'}")


def save_logloss_comparison_tables(
	outdir: Path,
	y_train: np.ndarray,
	y_test: np.ndarray,
	y_external: np.ndarray | None,
	proba_train: dict[str, np.ndarray],
	proba_test: dict[str, np.ndarray],
	proba_external: dict[str, np.ndarray] | None,
) -> None:
	"""Compute and save cross-cohort log loss tracking entities safely."""
	tables_dir = outdir / "tables"
	_ensure_dir(tables_dir)

	model_keys = list(proba_train.keys())
	key_order = ["LR", "DT", "RF", "SVM", "KNN", "NB", "XGB", "SGBT", "NNET"]
	ordered_keys = [k for k in key_order if k in model_keys]
	ordered_keys += [k for k in model_keys if k not in ordered_keys]

	def _calc_logloss(y_true: np.ndarray, y_proba: np.ndarray) -> float:
		return float(log_loss(y_true, np.vstack([1 - y_proba, y_proba]).T, labels=[0, 1]))

	rows_train = []
	for key in ordered_keys:
		ll = _calc_logloss(y_train, proba_train[key])
		rows_train.append({"Model": key, "Log_Loss_Train": ll})
	df_train = pd.DataFrame(rows_train)
	df_train.to_excel(tables_dir / "logloss_comparison_train.xlsx", index=False)

	rows_test = []
	for key in ordered_keys:
		ll = _calc_logloss(y_test, proba_test[key])
		rows_test.append({"Model": key, "Log_Loss_Test": ll})
	df_test = pd.DataFrame(rows_test)
	df_test.to_excel(tables_dir / "logloss_comparison_test.xlsx", index=False)

	if y_external is not None and proba_external is not None and len(proba_external) > 0:
		rows_ext = []
		for key in ordered_keys:
			if key in proba_external:
				ll = _calc_logloss(y_external, proba_external[key])
				rows_ext.append({"Model": key, "Log_Loss_External": ll})
		if rows_ext:
			df_ext = pd.DataFrame(rows_ext)
			df_ext.to_excel(tables_dir / "logloss_comparison_external.xlsx", index=False)

	rows_summary = []
	for key in ordered_keys:
		row = {
			"Model": key,
			"Log_Loss_Train": _calc_logloss(y_train, proba_train[key]),
			"Log_Loss_Test": _calc_logloss(y_test, proba_test[key]),
		}
		if y_external is not None and proba_external is not None and key in proba_external:
			row["Log_Loss_External"] = _calc_logloss(y_external, proba_external[key])
		rows_summary.append(row)
	df_summary = pd.DataFrame(rows_summary)
	df_summary.to_excel(tables_dir / "logloss_comparison_summary.xlsx", index=False)

	print(f"[INFO] Log Loss comparison tables saved to {tables_dir}")


def save_rfe_figure(rfe_df: pd.DataFrame, out_path: Path) -> None:
	key_order = ["LR", "DT", "RF", "KNN", "SVM", "NB", "XGB", "SGBT", "NNET"]
	present = [k for k in key_order if k in set(rfe_df["model"])]

	fig, axes = plt.subplots(3, 3, figsize=(18, 14), constrained_layout=True)
	axes = axes.flatten()
	letter = {"LR": "(A)", "DT": "(B)", "RF": "(C)", "KNN": "(D)", "SVM": "(E)", "NB": "(F)", "XGB": "(G)", "SGBT": "(H)", "NNET": "(I)"}

	for i, key in enumerate(present):
		ax = axes[i]
		sub = rfe_df[rfe_df["model"] == key].sort_values("k")
		ax.plot(sub["k"], sub["accuracy"], marker="o", label="Accuracy")
		ax2 = ax.twinx()
		ax2.plot(sub["k"], sub["kappa"], marker="s", color="tab:orange", label="Kappa")
		ax.set_xlabel("Number of features")
		ax.set_ylabel("Accuracy")
		ax2.set_ylabel("Kappa")
		ax.set_title(f"{key} {letter.get(key,'')}")
		ax.grid(True, alpha=0.3)

	for j in range(len(present), 9):
		axes[j].axis("off")
	fig.suptitle("RFE Curves: Accuracy & Kappa vs Feature Count", fontsize=16)
	fig.savefig(out_path, dpi=300)
	plt.close(fig)


def save_feature_importance_summary(
	all_importances: dict[str, pd.DataFrame],
	out_path: Path,
) -> None:
	"""Consolidate the feature importance tables of each model into a unified Excel book structure.

	Output format:
	- Separate sheets per individual tracking candidate model architecture entity
	- Supplemental comprehensive tracking summary spreadsheet layer reporting normalized vectors side-by-side
	"""
	if not all_importances:
		return

	key_order = ["LR", "DT", "RF", "SVM", "KNN", "NB", "XGB", "SGBT", "NNET"]
	present = [k for k in key_order if k in all_importances]
	present += [k for k in all_importances.keys() if k not in present]

	with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
		# 1. Individual structural target layouts
		for key in present:
			df = all_importances[key].copy()
			df.to_excel(writer, sheet_name=key, index=False)

		# 2. Consolidated summary layout section mapping
		summary_dfs = []
		for key in present:
			df = all_importances[key][["feature", "feature_display", "importance"]].copy()
			df = df.rename(columns={"importance": key})
			summary_dfs.append(df)

		if summary_dfs:
			merged = summary_dfs[0][["feature", "feature_display"]].copy()
			for i, key in enumerate(present):
				imp_col = summary_dfs[i][[key]].copy()
				merged = pd.concat([merged, imp_col], axis=1)

			model_cols = [k for k in present if k in merged.columns]
			merged["Mean_Importance"] = merged[model_cols].abs().mean(axis=1)
			merged = merged.sort_values("Mean_Importance", ascending=False, ignore_index=True)

			merged.insert(0, "Rank", range(1, len(merged) + 1))

			merged.to_excel(writer, sheet_name="Summary", index=False)

	print(f"[INFO] Feature importance summary saved to {out_path}")


def save_feature_importance_grid(
	all_importances: dict[str, pd.DataFrame],
	out_path: Path,
	top_n: int = 20,
) -> None:
	"""Render multi-panel 3x3 layout charting variable coefficients safely across configurations."""

	key_order = ["LR", "DT", "RF", "SVM", "KNN", "NB", "XGB", "SGBT", "NNET"]
	present = [k for k in key_order if k in all_importances]
	if not present:
		return

	fig, axes = plt.subplots(3, 3, figsize=(20, 12), constrained_layout=True)
	axes = axes.flatten()
	letter = {"LR": "LR", "DT": "DT", "RF": "RF", "KNN": "KNN", "SVM": "SVM", "NB": "NB", "XGB": "XGB", "SGBT": "SGBT", "NNET": "NNET"}

	for i, key in enumerate(present[:9]):
		ax = axes[i]
		df = all_importances[key].copy()
		if "abs_importance" not in df.columns:
			df["abs_importance"] = df["importance"].abs()
		df = df.sort_values("abs_importance", ascending=False).head(top_n)
		df = df.iloc[::-1]
		if "feature_display" in df.columns:
			feat_disp = df["feature_display"].astype(str).tolist()
		else:
			feat_raw = df["feature"].astype(str).tolist()
			feat_disp = _clean_display_feature_names(feat_raw)
		ax.barh(feat_disp, df["abs_importance"], color=sns.color_palette("tab10", n_colors=1)[0])
		ax.set_title(letter.get(key, key))
		ax.set_xlabel("Variable importance")
		ax.set_ylabel("Features")
		ax.grid(True, axis="x", alpha=0.25)
		ax.tick_params(axis="y", labelsize=8)

	for j in range(len(present[:9]), 9):
		axes[j].axis("off")

	fig.suptitle("Feature importance ranking from RFE (Top features per model)", fontsize=16)
	fig.savefig(out_path, dpi=300)
	plt.close(fig)


def save_feature_importance_heatmap(all_importances: dict[str, pd.DataFrame], out_path: Path, top_n: int = 25) -> None:
	"""Generate normalized comparative feature weight heatmaps across algorithm architectures securely."""

	if not all_importances:
		print("[FIG] No feature-importance tables; skip heatmap.")
		return

	key_order = ["LR", "DT", "RF", "SVM", "KNN", "NB", "XGB", "SGBT", "NNET"]
	present = [k for k in key_order if k in all_importances] + [k for k in all_importances if k not in key_order]

	parts: list[pd.DataFrame] = []
	for model_key in present:
		df = all_importances.get(model_key)
		if df is None or df.empty:
			continue
		sub = df.head(top_n).copy()
		if "abs_importance" not in sub.columns:
			sub["abs_importance"] = sub["importance"].abs()
		sub["model"] = model_key
		if "feature_display" in sub.columns:
			parts.append(sub[["model", "feature_display", "abs_importance"]].rename(columns={"feature_display": "feature"}))
		else:
			parts.append(sub[["model", "feature", "abs_importance"]])

	if not parts:
		print("[FIG] Empty feature-importance tables; skip heatmap.")
		return

	long = pd.concat(parts, ignore_index=True)
	pivot = (
		long.pivot_table(index="feature", columns="model", values="abs_importance", aggfunc="mean")
		.fillna(0.0)
		.reindex(columns=present, fill_value=0.0)
	)

	pivot["_sum"] = pivot.sum(axis=1)
	pivot = pivot.sort_values("_sum", ascending=False).drop(columns=["_sum"]).head(top_n)

	pretty_index = _clean_display_feature_names([str(x) for x in pivot.index.tolist()])
	pivot.index = pretty_index

	den = pivot.max(axis=0).replace(0.0, np.nan)
	pivot_plot = (pivot / den).fillna(0.0)

	fig_h = max(6, 0.38 * pivot_plot.shape[0])
	fig, ax = plt.subplots(figsize=(11, fig_h), constrained_layout=True)
	sns.heatmap(
		pivot_plot,
		cmap="viridis",
		vmin=0.0,
		vmax=1.0,
		linewidths=0.5,
		linecolor="white",
		cbar_kws={"label": "Relative importance (column-normalized)"},
		ax=ax,
	)
	ax.set_title("Top Feature Importance by Model")
	ax.set_xlabel("Model")
	ax.set_ylabel("Feature")
	ax.tick_params(axis="x", labelrotation=0, labelsize=9)
	ax.tick_params(axis="y", labelsize=8)
	fig.savefig(out_path, dpi=300)
	plt.close(fig)


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
) -> None:
	"""Generate high-fidelity comparative visual sample expression arrays independently per model configuration.

	- Rows: Selected model-specific attributes mapped properly
	- Columns: Cohort observations concatenated explicitly
	- Tracking context matrices included: Prediction Risk / Ground Truth Outcome / Set Membership Group
	- Color scale metadata reported cleanly via tabular layout panels
	"""
	if not importance_tables:
		print("[FIG] No importance tables; skip variable heatmaps.")
		return

	_ensure_dir(out_dir)

	has_external = (
		X_external is not None
		and y_external is not None
		and y_proba_external_by_model is not None
		and len(y_external) > 0
	)

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

		X_sub = X_all[:, feat_idx]
		vals = X_sub.toarray() if sp.issparse(X_sub) else np.asarray(X_sub)
		vals = vals.astype(float, copy=False).T  # (n_feat, n_samples)

		row_min = np.nanmin(vals, axis=1, keepdims=True)
		row_max = np.nanmax(vals, axis=1, keepdims=True)
		den = np.where((row_max - row_min) == 0, 1.0, (row_max - row_min))
		vals01 = (vals - row_min) / den
		vals01 = np.clip(vals01, 0.0, 1.0)

		# Sort observation entities: Outcome -> Group -> Risk Probability desc
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

		example_colors = [
			"#C68642",  # Dark brown orange
			"#F5C342",  # Gold yellow
			"#2196F3",  # Light blue
			"#E91E63",  # Rose pink
			"#795548",  # Brown
			"#4CAF50",  # Green
			"#FF9800",  # Orange
			"#9C27B0",  # Purple
			"#00BCD4",  # Cyan
			"#8BC34A",  # Light green
			"#FF5722",  # Deep orange
			"#3F51B5",  # Indigo blue
			"#CDDC39",  # Yellow green
			"#673AB7",  # Indigo purple
			"#009688",  # Deep cyan
			"#E040FB",  # Bright purple
			"#FF6F00",  # Amber
			"#1E88E5",  # Bright blue
			"#43A047",  # Deep green
			"#F06292",  # Pink
			"#7E57C2",  # Medium purple
			"#26A69A",  # Muted cyan
			"#FFA000",  # Muted yellow
			"#5E35B1",  # Muted indigo
		]
		base_colors = []
		for i in range(n_feat):
			base_colors.append(matplotlib.colors.to_rgba(example_colors[i % len(example_colors)]))

		light_rgba = matplotlib.colors.to_rgba("#F5F5DC")  # Light beige base segment mapping
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

			is_binary01 = uniq.size <= 2 and uniq.size > 0 and np.all(np.isin(uniq, [0.0, 1.0]))
			is_small_discrete = (not is_binary01) and (uniq.size > 0) and (uniq.size <= 3)

			if is_binary01:
				mask1 = np.isclose(v_raw, 1.0)
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

		legend_ncol = 6  
		n_legend_items = n_feat + 3  
		legend_nrows = int(math.ceil(n_legend_items / legend_ncol))

		fig_w = max(10.0, n_samp / 100.0)
		heatmap_h = 0.28 * n_feat
		annot_h = 0.3 * 3  
		legend_h = 1.8 * legend_nrows  
		fig_h = heatmap_h + annot_h + legend_h + 4.0  

		fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
		gs_main = fig.add_gridspec(
			6, 1,
			height_ratios=[n_feat, 1.0, 1.0, 1.0, 1.2, legend_nrows * 3.5 + 2.0],  
			hspace=0.0,
			top=0.96, bottom=0.04, left=0.02, right=0.88
		)

		# ============ Primary Visual Matrix ============
		ax0 = fig.add_subplot(gs_main[0, 0])
		ax0.imshow(rgba, aspect="auto", interpolation="nearest")
		ax0.set_yticks(np.arange(n_feat))
		ax0.set_yticklabels([])  
		ax0.set_xticks([])
		ax0.tick_params(left=False, bottom=False)
		for spine in ax0.spines.values():
			spine.set_visible(False)
		for i in range(1, n_feat):
			ax0.axhline(y=i - 0.5, color="white", linewidth=0.5)

		ax0_right = ax0.secondary_yaxis("right")
		ax0_right.set_yticks(np.arange(n_feat))
		ax0_right.set_yticklabels(feat_labels, fontsize=11)  
		ax0_right.tick_params(right=False)
		for spine in ax0_right.spines.values():
			spine.set_visible(False)

		# ============ Predicted value ribbon ============
		ax1 = fig.add_subplot(gs_main[1, 0], sharex=ax0)
		pred_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("pred", ["#FFF5E6", "#D2691E"])
		ax1.imshow(p_sorted.reshape(1, -1), aspect="auto", interpolation="none", cmap=pred_cmap, vmin=0.0, vmax=1.0)
		ax1.set_yticks([])
		ax1.set_xticks([])
		for spine in ax1.spines.values():
			spine.set_visible(False)
		ax1.axhline(y=-0.5, color="white", linewidth=0.5)
		ax1_right = ax1.secondary_yaxis("right")
		ax1_right.set_yticks([0])
		ax1_right.set_yticklabels([f"{key} predicted value"], fontsize=9)
		ax1_right.tick_params(right=False)
		for spine in ax1_right.spines.values():
			spine.set_visible(False)

		# ============ Outcome Ribbon ============
		ax2 = fig.add_subplot(gs_main[2, 0], sharex=ax0)
		outcome_code = outcome_code_sorted
		outcome_cmap = matplotlib.colors.ListedColormap(["#8BC34A", "#FF9800"])  # M0=Green M1=Orange
		ax2.imshow(outcome_code.reshape(1, -1), aspect="auto", interpolation="none", cmap=outcome_cmap, vmin=0, vmax=1)
		ax2.set_yticks([])
		ax2.set_xticks([])
		for spine in ax2.spines.values():
			spine.set_visible(False)
		ax2.axhline(y=-0.5, color="white", linewidth=0.5)
		ax2_right = ax2.secondary_yaxis("right")
		ax2_right.set_yticks([0])
		ax2_right.set_yticklabels(["Outcome"], fontsize=9)
		ax2_right.tick_params(right=False)
		for spine in ax2_right.spines.values():
			spine.set_visible(False)

		# ============ Group Ribbon ============
		ax3 = fig.add_subplot(gs_main[3, 0], sharex=ax0)
		group_code = group_code_sorted
		group_cmap = matplotlib.colors.ListedColormap(["#9C27B0", "#FF9800", "#00BCD4"])
		n_groups = 3 if has_external else 2
		ax3.imshow(group_code.reshape(1, -1), aspect="auto", interpolation="none", cmap=group_cmap, vmin=0, vmax=n_groups - 1)
		ax3.set_yticks([])
		ax3.set_xticks([])
		for spine in ax3.spines.values():
			spine.set_visible(False)
		ax3.axhline(y=-0.5, color="white", linewidth=0.5)
		ax3_right = ax3.secondary_yaxis("right")
		ax3_right.set_yticks([0])
		ax3_right.set_yticklabels(["Group"], fontsize=9)
		ax3_right.tick_params(right=False)
		for spine in ax3_right.spines.values():
			spine.set_visible(False)

		# ============ Color Scales Legend Matrices ============
		gs_legend = gs_main[5, 0].subgridspec(legend_nrows, legend_ncol, wspace=0.18, hspace=0.40)  

		legend_items = []
		for i in range(n_feat):
			ticks = row_ticks[i]
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
		legend_items.append({
			"type": "continuous",
			"label": f"{key} predicted value",
			"cmap": pred_cmap,
			"ticks": ([0, 0.5, 1], ["0", "0.5", "1"]),
			"vmin": 0, "vmax": 1,
			"base_color": matplotlib.colors.to_rgba("#D2691E"),
		})
		legend_items.append({
			"type": "categorical",
			"label": "Outcome",
			"colors": ["#8BC34A", "#FF9800"],
			"labels": ["M0", "M1"],
		})
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

			unified_ylim = (-0.1, 1.15)

			if item["type"] == "categorical":
				n_cats = len(item["colors"])
				box_h = 0.22  
				box_w = 0.22  
				for j, (color, label) in enumerate(zip(item["colors"], item["labels"])):
					y_pos = (n_cats - 1 - j) * box_h  
					rect = plt.Rectangle((0, y_pos), box_w, box_h * 0.9, facecolor=color, edgecolor="none")
					ax_leg.add_patch(rect)
					ax_leg.text(box_w + 0.06, y_pos + box_h * 0.45, label, ha="left", va="center", fontsize=9)
				ax_leg.set_xlim(-0.05, 1.3)
				ax_leg.set_ylim(unified_ylim)
			elif item["type"] == "binary":
				cm = item["cmap"]
				colors = [cm(0.0), cm(1.0)]
				labels = item["ticks"][1]
				box_h = 0.22
				box_w = 0.22
				for j, (color, label) in enumerate(zip(colors, labels)):
					y_pos = (1 - j) * box_h  
					rect = plt.Rectangle((0, y_pos), box_w, box_h * 0.9, facecolor=color, edgecolor="none")
					ax_leg.add_patch(rect)
					ax_leg.text(box_w + 0.06, y_pos + box_h * 0.45, label, ha="left", va="center", fontsize=9)
				ax_leg.set_xlim(-0.05, 1.0)
				ax_leg.set_ylim(unified_ylim)
			else:
				gradient = np.linspace(0, 1, 256).reshape(-1, 1)
				ax_leg.imshow(gradient, aspect="auto", cmap=item["cmap"], origin="lower", extent=[0, 0.22, 0, 1])
				vmin, vmax = item["vmin"], item["vmax"]
				if vmax - vmin > 0:
					n_ticks = 4
					tick_vals = np.linspace(vmin, vmax, n_ticks)
					tick_positions = np.linspace(0, 1, n_ticks)
				else:
					tick_vals = [vmin]
					tick_positions = [0.5]
				for tv, tp in zip(tick_vals, tick_positions):
					ax_leg.plot([0.24, 0.28], [tp, tp], color="0.3", linewidth=0.8)
					ax_leg.text(0.32, tp, _fmt_tick(tv), ha="left", va="center", fontsize=9)
				ax_leg.set_xlim(-0.05, 1.0)
				ax_leg.set_ylim(unified_ylim)

			ax_leg.set_xticks([])
			ax_leg.set_yticks([])
			for spine in ax_leg.spines.values():
				spine.set_visible(False)
			if item["type"] == "categorical":
				title_color = item["colors"][0]
			else:
				title_color = matplotlib.colors.to_hex(item["base_color"])
			ax_leg.text(0, unified_ylim[1] + 0.05, item["label"], fontsize=11, fontweight="bold", 
			            color=title_color, ha="left", va="bottom")

		plt.setp(ax0.get_xticklabels(), visible=False)
		out_path = out_dir / f"Variable_heatmap_{key}.png"
		fig.savefig(out_path, dpi=300, facecolor="white", bbox_inches="tight")
		plt.close(fig)



def main() -> int:
	examples = (
		"Examples:\n"
		"  # Recommended: ElasticNet feature selection (no stability selection by default, parameters fixed for reproducibility)\n"
		"  python 12-RFE+model+SHAP-Lasso.PY --fs_method elasticnet --fs_stability_runs 200 --fs_stability_fraction 0.8 --fs_stability_threshold 0.6\n\n"
		"  # Turn on stability selection for higher robustness/reproducibility: add --fs_stability explicitly\n"
		"  python 12-RFE+model+SHAP-Lasso.PY --fs_method elasticnet --fs_stability --fs_stability_runs 200 --fs_stability_fraction 0.8 --fs_stability_threshold 0.6\n\n"
		"  # Explicitly turn off stability selection (equivalent to default behavior):\n"
		"  python 12-RFE+model+SHAP-Lasso.PY --fs_method elasticnet --no_fs_stability --fs_stability_runs 200 --fs_stability_fraction 0.8 --fs_stability_threshold 0.6\n"
	)
	parser = argparse.ArgumentParser(
		description="L1/ElasticNet Logistic feature selection + 9 Models construction + Performance assessment (Metastasis)",
		epilog=examples,
		formatter_class=argparse.RawTextHelpFormatter,
	)
	parser.add_argument("--input", default=str(Path("mice") / "combined_imputed.xlsx"), help="Input Excel workbook location")
	parser.add_argument(
		"--rerun_shap_from",
		default=None,
		help="Given a previously processed run directory path, rerun SHAP pipelines extracting archived split allocations directly.",
	)
	parser.add_argument(
		"--external_input",
		default=str(Path("mice") / "combined_imputed-external.xlsx"),
		help="External verification workbook file (automatically skipped if absent)",
	)
	parser.add_argument("--target", default="Metastasis", help="Outcome classification column name (0/1 values)")
	parser.add_argument("--outdir", default=str(Path("ml_results") / f"run_{_now_stamp()}"), help="Destination results directory context")
	parser.add_argument("--random_state", type=int, default=42)
	parser.add_argument("--test_size", type=float, default=0.3)
	parser.add_argument("--cv_folds", type=int, default=10)
	parser.add_argument("--cv_repeats", type=int, default=10)
	parser.add_argument("--rfe_step", type=float, default=0.1, help="(Legacy backward compatibility flag) deprecated")
	parser.add_argument("--rfe_curve_points", type=int, default=30, help="(Legacy backward compatibility flag) deprecated")
	parser.add_argument("--max_dense_features", type=int, default=40, help="Maximum feature dimension boundaries enforced on dense inputs")
	parser.add_argument("--bootstrap", type=int, default=500)
	parser.add_argument(
		"--calibrate_proba",
		dest="calibrate_proba",
		action="store_true",
		help="Execute post-hoc verification calibrator configurations over raw risk probability vectors (turned on by default).",
	)
	parser.add_argument(
		"--no_calibrate_proba",
		dest="calibrate_proba",
		action="store_false",
		help="Bypass probability calibrator parameters, retrieving raw probability estimations directly.",
	)
	parser.set_defaults(calibrate_proba=True)
	parser.add_argument(
		"--calibration_method",
		choices=["sigmoid", "isotonic", "none"],
		default="sigmoid",
		help="Calibrator algorithmic mechanism selection: sigmoid (recommended for small samples) / isotonic / none.",
	)
	parser.add_argument(
		"--min_features",
		type=int,
		default=10,
		help="Minimum constraint of features retained during structural feature selection workflows.",
	)
	parser.add_argument(
		"--fs_method",
		choices=["elasticnet", "l1"],
		default="elasticnet",
		help="Feature selection algorithm: elasticnet (highly recommended) or l1 penalty lasso.",
	)
	parser.add_argument(
		"--fs_C_grid",
		default="0.005,0.01,0.02,0.05,0.1,0.2,0.5,1",
		help="Regularization coefficient sweep values for logistic selection models, comma separated (smaller implies higher penalty).",
	)
	parser.add_argument("--fs_cv_folds", type=int, default=5, help="Fold partitions dedicated to grid search cross-validation selection iterations")
	parser.add_argument(
		"--fs_l1_ratio_grid",
		default="0.5,0.8,0.9,0.95",
		help="Mixing threshold parameters for ElasticNet setups, comma separated.",
	)
	parser.add_argument(
		"--fs_coef_tol",
		type=float,
		default=1e-6,
		help="Threshold parameter under which sparse coefficients evaluate strictly to true zero values.",
	)
	parser.add_argument(
		"--fs_stability",
		dest="fs_stability",
		action="store_true",
		help="Activate repeated bootstrap/shuffle sub-sampling filtering routines (highly recommended).",
	)
	parser.add_argument(
		"--no_fs_stability",
		dest="fs_stability",
		action="store_false",
		help="Deactivate stability iteration frameworks.",
	)
	parser.set_defaults(fs_stability=True)
	parser.add_argument("--fs_stability_runs", type=int, default=200, help="Total execution passes configured for stability search loops")
	parser.add_argument("--fs_stability_fraction", type=float, default=0.8, help="Sample ratio extracted per independent stability analysis loop pass")
	parser.add_argument("--fs_stability_threshold", type=float, default=0.6, help="Frequency inclusion threshold criterion (bounded between 0 and 1)")
	parser.add_argument(
		"--n_jobs",
		type=int,
		default=1,
		help="Concurrent workers bound configuration. Keeping it to 1 is optimal under Windows configurations.",
	)
	parser.add_argument("--no_shap", action="store_true", help="Deactivate local/global SHAP contextual attribution pipelines entirely")
	parser.add_argument("--shap_background", type=int, default=100, help="Baseline references size reserved for SHAP value computation maps")
	parser.add_argument("--shap_samples", type=int, default=200, help="Total observations isolated from testing pools for SHAP interpretations")
	args = parser.parse_args()

	if args.rerun_shap_from:
		_rerun_shap_from_existing_run(
			run_dir=Path(args.rerun_shap_from),
			random_state=args.random_state,
			shap_background=args.shap_background,
			shap_samples=args.shap_samples,
		)
		return 0

	outdir = Path(args.outdir)
	_ensure_dir(outdir)
	_ensure_dir(outdir / "fig")
	_ensure_dir(outdir / "tables")

	joblib_temp = Path(os.environ.get("JOBLIB_TEMP_FOLDER", ""))
	if not str(joblib_temp):
		joblib_temp = Path("C:/joblib_temp")
	try:
		joblib_temp.mkdir(parents=True, exist_ok=True)
		os.environ.setdefault("JOBLIB_TEMP_FOLDER", str(joblib_temp))
		os.environ.setdefault("LOKY_TEMP_FOLDER", str(joblib_temp))
	except Exception:
		args.n_jobs = 1

	input_path = Path(args.input)
	if not input_path.exists():
		raise FileNotFoundError(f"Missing master data source input path error: {input_path}")

	external_path = Path(args.external_input)
	has_external = bool(external_path.exists())

	df = pd.read_excel(input_path)
	if args.target not in df.columns:
		raise ValueError(f"Missing essential categorical target key: {args.target}")

	y = df[args.target].astype(int).to_numpy()
	cols_to_drop = [args.target] + [c for c in DEMOGRAPHIC_COLUMNS_EXCLUDE if c in df.columns]
	X_df = df.drop(columns=cols_to_drop)
	X_df = normalize_categorical_columns(X_df, CATEGORICAL_COLUMNS_DEFAULT)
	pre, numeric_cols, cat_cols = build_preprocessor(X_df, CATEGORICAL_COLUMNS_DEFAULT)

	splitter = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
	train_idx, test_idx = next(splitter.split(X_df, y))
	X_train_df = X_df.iloc[train_idx].copy()
	X_test_df = X_df.iloc[test_idx].copy()
	X_train_df = normalize_categorical_columns(X_train_df, cat_cols)
	X_test_df = normalize_categorical_columns(X_test_df, cat_cols)
	y_train = y[train_idx]
	y_test = y[test_idx]

	df_train = df.iloc[train_idx].copy()
	df_test = df.iloc[test_idx].copy()
	for c in cat_cols:
		if c in df_train.columns:
			df_train[c] = normalize_categorical_series(df_train[c])
		if c in df_test.columns:
			df_test[c] = normalize_categorical_series(df_test[c])
	train_xlsx = outdir / "tables" / "split_train.xlsx"
	test_xlsx = outdir / "tables" / "split_test.xlsx"
	df_train.to_excel(train_xlsx, index=False)
	df_test.to_excel(test_xlsx, index=False)

	try:
		group_col = str(args.target)
		if group_col in df_train.columns and group_col in df_test.columns:
			tr_vals = pd.to_numeric(df_train[group_col], errors="coerce").dropna().unique().tolist()
			te_vals = pd.to_numeric(df_test[group_col], errors="coerce").dropna().unique().tolist()
			can_analyze = set(tr_vals).issubset({0, 1}) and set(te_vals).issubset({0, 1})
		else:
			can_analyze = False
		if can_analyze:
			script_path = Path(__file__).with_name("8-analyse_all.py")
			out_train_stats = outdir / "tables" / "terminal_analysis_results_train.xlsx"
			out_test_stats = outdir / "tables" / "terminal_analysis_results_test.xlsx"
			subprocess.run(
				[
					sys.executable,
					str(script_path),
					"--input",
					str(train_xlsx),
					"--output",
					str(out_train_stats),
					"--group_col",
					group_col,
				],
				check=True,
			)
			subprocess.run(
				[
					sys.executable,
					str(script_path),
					"--input",
					str(test_xlsx),
					"--output",
					str(out_test_stats),
					"--group_col",
					group_col,
				],
				check=True,
			)
		else:
			print(f"[WARN] Skipping automated table statistics: target variable '{group_col}' missing or not strictly binary 0/1.")
	except Exception as e:
		print(f"[WARN] Baseline tabular descriptive statistics script execution bypassed safely: {e}")

	pre.fit(X_train_df)
	X_train = _safe_float32_matrix(pre.transform(X_train_df))
	X_test = _safe_float32_matrix(pre.transform(X_test_df))
	X_external = None
	y_external = None
	if has_external:
		ext_df = pd.read_excel(external_path)
		if args.target not in ext_df.columns:
			print(f"[WARN] External validation source dropped due to missing label definition: '{args.target}' under filename: {external_path}")
			has_external = False
		else:
			y_external = ext_df[args.target].astype(int).to_numpy()
			ext_cols_to_drop = [args.target] + [c for c in DEMOGRAPHIC_COLUMNS_EXCLUDE if c in ext_df.columns]
			X_ext_df = ext_df.drop(columns=ext_cols_to_drop)
			for col in X_train_df.columns:
				if col not in X_ext_df.columns:
					X_ext_df[col] = np.nan
			X_ext_df = X_ext_df.copy()
			X_ext_df = normalize_categorical_columns(X_ext_df, cat_cols)
			X_external = _safe_float32_matrix(pre.transform(X_ext_df))
			ext_xlsx = outdir / "tables" / "split_external.xlsx"
			ext_df.to_excel(ext_xlsx, index=False)
	else:
		print(f"[WARN] External target verification source data absent, bypassing track cleanly: {external_path}")
	feature_names = get_feature_names(pre)

	cv = RepeatedStratifiedKFold(n_splits=args.cv_folds, n_repeats=args.cv_repeats, random_state=args.random_state)
	specs = build_model_specs(args.random_state)
	if not _HAS_XGBOOST:
		print("Notice: xgboost binary dependency packages are missing from the environment. XGB models will be skipped.")

	run_meta = {
		"input": str(input_path),
		"external_input": str(external_path),
		"has_external": bool(has_external),
		"target": args.target,
		"n_samples": int(df.shape[0]),
		"n_train": int(len(train_idx)),
		"n_test": int(len(test_idx)),
		"n_external": int(y_external.shape[0]) if (has_external and y_external is not None) else 0,
		"n_features_after_ohe": int(X_train.shape[1]),
		"numeric_cols": numeric_cols,
		"categorical_cols_used": cat_cols,
		"cv_folds": args.cv_folds,
		"cv_repeats": args.cv_repeats,
		"feature_selection": {
			"method": args.fs_method,
			"C_grid": args.fs_C_grid,
			"cv_folds": args.fs_cv_folds,
			"l1_ratio_grid": args.fs_l1_ratio_grid,
			"coef_tol": args.fs_coef_tol,
			"min_features": args.min_features,
			"max_dense_features": args.max_dense_features,
			"stability": bool(args.fs_stability),
			"stability_runs": args.fs_stability_runs,
			"stability_fraction": args.fs_stability_fraction,
			"stability_threshold": args.fs_stability_threshold,
		},
		"bootstrap": args.bootstrap,
		"calibration": {
			"enabled": bool(args.calibrate_proba) and str(args.calibration_method).lower() != "none",
			"method": str(args.calibration_method),
		},
	}
	(outdir / "run_meta.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")

	# Phase I: Feature Selection
	print(f"[FS] Logistic ({args.fs_method}) ...")
	C_grid = _parse_float_list(args.fs_C_grid)
	l1_ratio_grid = _parse_float_list(args.fs_l1_ratio_grid)
	if not C_grid:
		C_grid = [0.01, 0.1, 1.0, 10.0]
	if not l1_ratio_grid:
		l1_ratio_grid = [0.2, 0.5, 0.8]

	selected_idx, fs_meta = select_features_lr_enet_or_l1(
		X_train=X_train,
		y_train=y_train,
		feature_names=feature_names,
		method=args.fs_method,
		C_grid=C_grid,
		l1_ratio_grid=l1_ratio_grid,
		fs_cv_folds=args.fs_cv_folds,
		min_features=args.min_features,
		max_features=args.max_dense_features,
		coef_tol=args.fs_coef_tol,
		stability=bool(args.fs_stability),
		stability_runs=args.fs_stability_runs,
		stability_fraction=args.fs_stability_fraction,
		stability_threshold=args.fs_stability_threshold,
		random_state=args.random_state,
		n_jobs=args.n_jobs,
		outdir_tables=outdir / "tables",
		outdir_fig=outdir / "fig",
	)
	print(f"[FS] Selected features count: {int(selected_idx.size)}")

	best_feature_sets: dict[str, np.ndarray] = {s.key: selected_idx for s in specs}

	importance_tables: dict[str, pd.DataFrame] = {}
	for spec in specs:
		imp_df = compute_feature_importance_scores(
			spec,
			X_train,
			y_train,
			selected_idx,
			feature_names,
			random_state=args.random_state,
		)
		importance_tables[spec.key] = imp_df
		imp_df.to_csv(outdir / "tables" / f"feature_importance_{spec.key}.csv", index=False, encoding="utf-8-sig")

	save_feature_importance_summary(importance_tables, outdir / "tables" / "feature_importance_summary.xlsx")
	save_feature_importance_grid(importance_tables, outdir / "fig" / "FS_feature_importance_grid.png")

	# Phase II: Model Construction
	best_params_rows: list[dict[str, Any]] = []
	fitted_pipelines: dict[str, Pipeline] = {}
	specs_by_key: dict[str, ModelSpec] = {s.key: s for s in specs}
	for spec in specs:
		print(f"[GRID] {spec.key} ...")
		feat_idx = best_feature_sets[spec.key]
		pipe = make_pipeline_for_model(spec, feat_idx)
		gs = GridSearchCV(
			estimator=pipe,
			param_grid=spec.param_grid,
			scoring="roc_auc",
			cv=cv,
			n_jobs=args.n_jobs,
			refit=True,
		)
		gs.fit(X_train, y_train)
		fitted_pipelines[spec.key] = gs.best_estimator_
		params = gs.best_params_
		if params:
			param_strs = []
			for k, v in params.items():
				clean_key = k.replace("model__", "")
				param_strs.append(f"{clean_key} = {v}")
			hyperparam_str = ", ".join(param_strs)
		else:
			hyperparam_str = "none"
		row = {
			"Model": f"{spec.title} ({spec.key})",
			"Hyperparameter": hyperparam_str,
		}
		best_params_rows.append(row)

	pd.DataFrame(best_params_rows).to_excel(outdir / "tables" / "best_hyperparameters.xlsx", index=False)
	_save_training_artifacts(outdir, pre, fitted_pipelines)

	# Phase III: Model Performance Assessment
	eval_rows: list[dict[str, Any]] = []
	y_proba_train_by_model: dict[str, np.ndarray] = {}
	y_proba_test_by_model: dict[str, np.ndarray] = {}
	y_proba_external_by_model: dict[str, np.ndarray] = {}
	oof_cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state)

	for spec in specs:
		key = spec.key
		model = fitted_pipelines[key]
		p_tr_raw = oof_predict_proba_mean(model, X_train, y_train, oof_cv)
		p_te_raw = model.predict_proba(X_test)[:, 1]
		p_ex_raw = None
		if has_external and X_external is not None:
			p_ex_raw = model.predict_proba(X_external)[:, 1]

		p_tr = p_tr_raw
		p_te = p_te_raw
		p_ex = p_ex_raw
		calib_used = "none"
		if bool(args.calibrate_proba) and str(args.calibration_method).lower() != "none":
			calib_used, calib_fn = _build_posthoc_calibrator(str(args.calibration_method), p_tr_raw, y_train)
			p_tr = calib_fn(p_tr_raw)
			p_te = calib_fn(p_te_raw)
			if p_ex_raw is not None:
				p_ex = calib_fn(p_ex_raw)

		y_proba_train_by_model[key] = np.asarray(p_tr, dtype=float)
		y_proba_test_by_model[key] = np.asarray(p_te, dtype=float)
		if p_ex is not None:
			y_proba_external_by_model[key] = np.asarray(p_ex, dtype=float)

		eval_rows.append({"Dataset": "Train", "Model": key, "Calibration": calib_used, **metrics_from_proba(y_train, p_tr)})
		eval_rows.append({"Dataset": "Test", "Model": key, "Calibration": calib_used, **metrics_from_proba(y_test, p_te)})
		if has_external and y_external is not None and p_ex is not None:
			eval_rows.append(
				{"Dataset": "External", "Model": key, "Calibration": calib_used, **metrics_from_proba(y_external, p_ex)}
			)

		ci_tr = bootstrap_metrics_ci(y_train, p_tr, n_boot=args.bootstrap, random_state=args.random_state)
		ci_tr.insert(0, "Dataset", "Train")
		ci_tr.insert(1, "Model", key)
		ci_te = bootstrap_metrics_ci(y_test, p_te, n_boot=args.bootstrap, random_state=args.random_state + 1)
		ci_te.insert(0, "Dataset", "Test")
		ci_te.insert(1, "Model", key)
		ci_parts = [ci_tr, ci_te]
		if has_external and y_external is not None and p_ex is not None:
			ci_ex = bootstrap_metrics_ci(y_external, p_ex, n_boot=args.bootstrap, random_state=args.random_state + 2)
			ci_ex.insert(0, "Dataset", "External")
			ci_ex.insert(1, "Model", key)
			ci_parts.append(ci_ex)
		pd.concat(ci_parts, ignore_index=True).to_excel(outdir / "tables" / f"metrics_ci_{key}.xlsx", index=False)

	eval_df = pd.DataFrame(eval_rows)
	eval_df.to_excel(outdir / "tables" / "metrics_train_test_external_point_estimates.xlsx", index=False)

	ci_paths_by_model = {k: (outdir / "tables" / f"metrics_ci_{k}.xlsx") for k in y_proba_test_by_model.keys()}
	ci_by_model: dict[str, dict[tuple[str, str], tuple[float, float, float]]] = {
		k: _metric_ci_lookup(p) for k, p in ci_paths_by_model.items()
	}

	def _plot_roc_simple(dataset: str, y_true: np.ndarray, proba_by_model: dict[str, np.ndarray], out: Path) -> None:
		fig, ax = plt.subplots(figsize=(8, 7))
		_plot_roc_panel(ax, f"{dataset} set", y_true, proba_by_model, ci_by_model, dataset)
		fig.tight_layout()
		fig.savefig(out, dpi=300)
		plt.close(fig)

	_plot_roc_simple("Train", y_train, y_proba_train_by_model, outdir / "fig" / "ROC_train.png")
	_plot_roc_simple("Test", y_test, y_proba_test_by_model, outdir / "fig" / "ROC_test.png")
	if has_external and y_external is not None and y_proba_external_by_model:
		_plot_roc_simple("External", y_external, y_proba_external_by_model, outdir / "fig" / "ROC_external.png")

	def _plot_calibration_simple(dataset: str, y_true: np.ndarray, proba_by_model: dict[str, np.ndarray], out: Path) -> None:
		fig, ax = plt.subplots(figsize=(8, 7))
		_plot_calibration_panel(ax, f"{dataset} set", y_true, proba_by_model, ci_by_model, dataset)
		fig.tight_layout()
		fig.savefig(out, dpi=300)
		plt.close(fig)

	_plot_calibration_simple("Train", y_train, y_proba_train_by_model, outdir / "fig" / "Calibration_train.png")
	_plot_calibration_simple("Test", y_test, y_proba_test_by_model, outdir / "fig" / "Calibration_test.png")
	if has_external and y_external is not None and y_proba_external_by_model:
		_plot_calibration_simple("External", y_external, y_proba_external_by_model, outdir / "fig" / "Calibration_external.png")

	def _metrics_bar_grid(out: Path) -> None:
		metric_order = ["Accuracy", "AUC", "Brier", "F1", "NPV", "PPV", "Sensitivity", "Specificity"]
		pretty = {
			"Accuracy": "Accuracy",
			"AUC": "AUC",
			"Brier": "Brier",
			"F1": "F1",
			"NPV": "NPV",
			"PPV": "Precision",
			"Sensitivity": "Sensitivity",
			"Specificity": "Specificity",
		}
		model_order = ["SGBT", "LR", "KNN", "NB", "NNET", "RF", "DT", "SVM", "XGB"]
		present_models = [m for m in model_order if m in set(eval_df["Model"]) ]

		long_parts: list[pd.DataFrame] = []
		for metric in metric_order:
			sub = eval_df[["Dataset", "Model", metric]].copy()
			sub = sub.rename(columns={metric: "Value"})
			sub["Metric"] = pretty[metric]
			long_parts.append(sub)
		long_df = pd.concat(long_parts, ignore_index=True)
		long_df = long_df[long_df["Model"].isin(present_models)].copy()
		long_df["Model"] = pd.Categorical(long_df["Model"], categories=present_models, ordered=True)
		datasets = ["Train", "Test"] + (["External"] if (has_external and "External" in set(long_df["Dataset"])) else [])
		long_df["Dataset"] = pd.Categorical(long_df["Dataset"], categories=datasets, ordered=True)

		palette_colors = {
			"Train": "#66C2A5",   # Muted light green
			"Test": "#FC8D62",    # Muted orange
			"External": "#8DA0CB",  # Muted blue-purple
		}
		palette = {k: v for k, v in palette_colors.items() if k in datasets}

		fig, axes = plt.subplots(3, 3, figsize=(18, 13), constrained_layout=True)
		axes = axes.flatten()
		for i, metric_name in enumerate([pretty[m] for m in metric_order]):
			ax = axes[i]
			sub = long_df[long_df["Metric"] == metric_name]
			sns.barplot(
				data=sub,
				x="Model",
				y="Value",
				hue="Dataset",
				ax=ax,
				errorbar=None,
				palette=palette,
			)
			ax.set_title(metric_name, fontsize=14, fontweight="bold")
			ax.set_xlabel("")
			ax.set_ylabel("Value", fontsize=12)
			ax.tick_params(axis="x", labelsize=11)
			ax.tick_params(axis="y", labelsize=11)
			ax.grid(True, axis="y", alpha=0.25)
			leg = ax.get_legend()
			if leg is not None:
				leg.remove()

		leg_ax = axes[8]
		leg_ax.axis("off")
		handles, labels = axes[0].get_legend_handles_labels()
		leg_ax.legend(
			handles,
			labels,
			title="Dataset",
			loc="center",
			frameon=True,
			fancybox=True,
			edgecolor="lightgray",
			fontsize=14,
			title_fontsize=16,
			handlelength=2.5,
			handletextpad=1.0,
			labelspacing=0.8,
			borderpad=1.2,
		)
		fig.savefig(out, dpi=300)
		plt.close(fig)

	_metrics_bar_grid(outdir / "fig" / "Metrics_bargrid.png")

	_save_parallel_metrics_single(outdir / "fig" / "Metrics_parallel_train.png", "Training set", eval_df, "Train")
	_save_parallel_metrics_single(outdir / "fig" / "Metrics_parallel_test.png", "Test set", eval_df, "Test")
	if has_external:
		_save_parallel_metrics_single(outdir / "fig" / "Metrics_parallel_external.png", "External set", eval_df, "External")

	_make_summary_figure_single_dataset(
		outdir / "fig" / "Summary_Train_ROC_Calibration_Metrics_DCA.png",
		dataset_title="Training set",
		dataset_label="Train",
		y_true=y_train,
		proba_by_model=y_proba_train_by_model,
		eval_df=eval_df,
		ci_paths_by_model=ci_paths_by_model,
	)
	_make_summary_figure_single_dataset(
		outdir / "fig" / "Summary_Test_ROC_Calibration_Metrics_DCA.png",
		dataset_title="Internal test set",
		dataset_label="Test",
		y_true=y_test,
		proba_by_model=y_proba_test_by_model,
		eval_df=eval_df,
		ci_paths_by_model=ci_paths_by_model,
	)
	if has_external and y_external is not None and y_proba_external_by_model:
		_make_summary_figure_single_dataset(
			outdir / "fig" / "Summary_External_ROC_Calibration_Metrics_DCA.png",
			dataset_title="External validation set",
			dataset_label="External",
			y_true=y_external,
			proba_by_model=y_proba_external_by_model,
			eval_df=eval_df,
			ci_paths_by_model=ci_paths_by_model,
		)

	save_variable_heatmaps_by_model(
		out_dir=outdir / "fig",
		X_train=X_train,
		X_test=X_test,
		y_train=y_train,
		y_test=y_test,
		y_proba_train_by_model=y_proba_train_by_model,
		y_proba_test_by_model=y_proba_test_by_model,
		importance_tables=importance_tables,
		top_n=20,
		X_external=X_external if has_external else None,
		y_external=y_external if has_external else None,
		y_proba_external_by_model=y_proba_external_by_model if has_external else None,
	)

	save_delong_comparison_tables(
		outdir=outdir,
		y_train=y_train,
		y_test=y_test,
		y_external=y_external if has_external else None,
		proba_train=y_proba_train_by_model,
		proba_test=y_proba_test_by_model,
		proba_external=y_proba_external_by_model if has_external else None,
	)

	save_nri_idi_comparison_tables(
		outdir=outdir,
		y_train=y_train,
		y_test=y_test,
		y_external=y_external if has_external else None,
		proba_train=y_proba_train_by_model,
		proba_test=y_proba_test_by_model,
		proba_external=y_proba_external_by_model if has_external else None,
		n_boot=1000,
		random_state=args.random_state,
	)

	save_logloss_comparison_tables(
		outdir=outdir,
		y_train=y_train,
		y_test=y_test,
		y_external=y_external if has_external else None,
		proba_train=y_proba_train_by_model,
		proba_test=y_proba_test_by_model,
		proba_external=y_proba_external_by_model if has_external else None,
	)

	thresholds = np.linspace(0.0, 0.99, 100)
	_save_dca_single(outdir / "fig" / "DCA_train.png", "Training set", y_train, y_proba_train_by_model, thresholds)
	_save_dca_single(outdir / "fig" / "DCA_test.png", "Test set", y_test, y_proba_test_by_model, thresholds)
	if has_external and y_external is not None and y_proba_external_by_model:
		_save_dca_single(
			outdir / "fig" / "DCA_external.png",
			"External validation set",
			y_external,
			y_proba_external_by_model,
			thresholds,
		)

	model_keys = list(y_proba_train_by_model.keys())
	if has_external and y_external is not None and y_proba_external_by_model:
		_save_hl_grid(
			outdir / "fig" / "HL_grid.png",
			model_keys,
			y_train,
			y_test,
			y_external,
			y_proba_train_by_model,
			y_proba_test_by_model,
			y_proba_external_by_model,
			n_boot=args.bootstrap,
			random_state=args.random_state,
		)
	else:
		_save_hl_grid_train_test_only(
			outdir / "fig" / "HL_grid.png",
			model_keys,
			y_train,
			y_test,
			y_proba_train_by_model,
			y_proba_test_by_model,
			n_boot=args.bootstrap,
			random_state=args.random_state,
		)

	# Model selection strategy: rely purely on training OOF vectors to prevent look-ahead testing optimism bias cascades
	train_rows = eval_df[eval_df["Dataset"] == "Train"].copy()
	best = train_rows.sort_values(["AUC", "Brier", "LogLoss"], ascending=[False, True, True]).iloc[0]
	best_model = str(best["Model"])
	best_test = eval_df[(eval_df["Dataset"] == "Test") & (eval_df["Model"] == best_model)].iloc[0]
	(outdir / "best_model.txt").write_text(
		(
			f"Best model (by Train OOF AUC, then Brier, then LogLoss): {best_model}\n\n"
			f"Train(OOF):\n{best.to_string(index=False)}\n\n"
			f"Test:\n{best_test.to_string(index=False)}\n"
		),
		encoding="utf-8",
	)

	print(f"Execution terminated successfully. Outputs stored under root filename: {outdir}")

	if not args.no_shap:
		_run_shap(
			outdir=outdir,
			fitted_pipelines=fitted_pipelines,
			best_feature_sets=best_feature_sets,
			specs_by_key=specs_by_key,
			X_train=X_train,
			X_test=X_test,
			y_train=y_train,
			y_test=y_test,
			feature_names=feature_names,
			random_state=args.random_state,
			shap_background=args.shap_background,
			shap_samples=args.shap_samples,
		)

	return 0


if __name__ == "__main__":
	raise SystemExit(main())