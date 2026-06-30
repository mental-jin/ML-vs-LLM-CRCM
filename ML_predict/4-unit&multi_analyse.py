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
# 1. Configuration Region
# ==========================================
file_path = r'tmp\merged_1634.xlsx' # Path to Excel data file
target_col = 'Metastasis'          # Name of dependent variable column (Excel header matching)
exclude_cols = ['Name', 'ReportTime', 'VisitNumber', 'Number']    # Independent columns to exclude (set empty to skip exclusions)

# Multivariable models use imputed data sheets to limit sample dropouts from missing cells
multivar_file_path = r'tmp\merged_1634.xlsx'

# Forced Variable Types overrides (takes priority over automatic inference heuristics)
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

# Multivariable Specific Overrides (For publication-ready "Independent Predictors" models):
# - Highest priority, isolates adjustments exclusively inside multivariable formula lists.
# - Leaves blank to reuse parent defaults.
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

# Map factor reference baseline parameters explicitly: {'ColumnHeader': 'BaselineValue'}
REFERENCE_MAP: Dict[str, str] = {"Sex":'Female', "T stage":'T1-T2', "N stage":'0', "Differentiation grade":'G1', "Vascular invasion":'0', "Perineural invasion":'0',
    "Carcinoma nodule":'0', "MLH1":'1', "MSH2":'1', "PMS2":'1', "MSH6":'1', "Family history":'0',
    "Colonic obstruction":'0', "Hypertension":'0', "Diabetes":'0', "Coronary artery disease":'0', "Hyperlipidemia":'0',
    "BRAF mutant":'0', "KRAS mutant":'0', "NRAS mutant":'0', "mGPS":'0',"Ki67":'0',
    "MSI-H":'0'}

# Categorical determination cut-off value (columns with unique terms fewer than this are handled as categorical)
cat_threshold = 10 
logit_method = None            # Optimization solver method (None falls back to statsmodels Newton-Raphson)
logit_maxiter = 1000           # Maximized iterations allowed to solve coefficients
exp_clip_value = 50            # Threshold to truncate exponent elements to prevent float overflow

# Multivariable Regression Stability Constraints
MV_STANDARDIZE_CONTINUOUS = True   # Standardize continuous features via z-scoring
MV_Z_CLIP = 8.0                    # Numeric range to truncate z-scores to suppress float overflow
MV_REG_ALPHA = 0.5                 # Regularization penalty weight parameter for fit_regularized
MV_REG_L1_WT = 0.0                 # Regularization choice parameter: 0=Pure L2 Ridge, 1=Pure L1 Lasso

# Structural features to optimize performance (Limits matrix singular state errors from high colinear properties)
MV_PRUNE_CONTINUOUS_BY_CORR = True
MV_CORR_THRESHOLD = 0.90           # Spearman correlation index cutoff: drops the latter tracking column if |rho| exceeds threshold
MV_COLLAPSE_RARE_LEVELS = True
MV_MIN_LEVEL_COUNT = 10            # Minimum sample frequency required per factor bucket; otherwise groups into 'Other'

# Ordinal numeric trend modeling targets list
ORDINAL_TREND_VARS: List[str] = []

# Multivariable evaluation columns inventory list
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
# 2. Data Loading & Preprocessing
# ==========================================
def clean_column_name(name):
    """
    Cleans column names to remove invalid formula characters for safe patsy operations.
    e.g., "BMI (kg/m2)" -> "BMI_kg_m2"
    """
    new_name = re.sub(r'[^\w]', '_', name)
    return new_name


def make_unique_safe_names(original_names: List[str]) -> Tuple[List[str], Dict[str, str], Dict[str, str]]:
    """Generates clean, non-duplicate, formula-safe label strings for patsy equations."""
    used: Dict[str, int] = {}
    safe_names: List[str] = []
    safe_to_orig: Dict[str, str] = {}
    orig_to_safe: Dict[str, str] = {}

    for orig in original_names:
        base = clean_column_name(str(orig))
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
    """Escapes string quotes securely for patsy model configuration steps."""
    return value.replace("\\", "\\\\").replace("'", "\\'")


def coerce_binary_target(series: pd.Series) -> Tuple[pd.Series, Optional[Dict[str, int]]]:
    """Coerces dependent variable entries to a structured 0/1 indicator."""
    s = series.copy()
    if s.dropna().map(type).isin([bool]).all():
        return s.astype(int), None

    s_num = pd.to_numeric(s, errors='coerce')
    uniq_num = sorted(set(s_num.dropna().unique().tolist()))
    if set(uniq_num).issubset({0.0, 1.0}) and len(uniq_num) in (1, 2):
        return s_num.astype('Int64').astype(float).astype('float64'), None

    s_str = s.astype(str).str.strip()
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

    uniq = [x for x in pd.unique(s_str.dropna()) if x is not None]
    if len(uniq) == 2:
        auto_map = {str(uniq[0]): 0, str(uniq[1]): 1}
        return s_str.map(auto_map).astype('float64'), auto_map

    raise ValueError(
        f"Dependent outcome column '{target_col}' cannot be mapped to binary 0/1 parameters. Address formatting manually."
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
    """Drops collinear continuous attributes based on pairwise Spearman correlation scores."""
    cols = [c for c in cols_in_order if c in df_in.columns]
    if len(cols) < 2:
        return cols
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
        print(f"Notice: Pruned {len(dropped)} collinear continuous columns in multivariable staging: {dropped}")
    return keep


def parse_ordinal_trend(series: pd.Series) -> pd.Series:
    """Extracts leading integers from tracking levels to establish structured order values."""
    s = series.copy()
    s = s.mask(s == '/', np.nan)

    as_num = pd.to_numeric(s, errors='coerce')
    if int(as_num.notna().sum()) == int(s.notna().sum()):
        return as_num

    s_str = s.astype(str).str.strip()
    s_str = s_str.replace({'nan': np.nan, 'NaN': np.nan, 'None': np.nan, '': np.nan})
    extracted = pd.to_numeric(s_str.str.extract(r'(\d+)')[0], errors='coerce')
    return as_num.fillna(extracted)

def safe_exp(value):
    """Limits exponent calculation step limits to avoid numeric system crash bounds."""
    clipped = np.clip(value, -exp_clip_value, exp_clip_value)
    return float(np.exp(clipped))


def infer_is_categorical(orig_name: str, series: pd.Series) -> bool:
    """Evaluates variable array configuration properties based on user rule sheets or value profiles."""
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
    """Standardizes object levels to avoid separate factor groups from formatting differences."""
    s2 = s.copy()
    s2 = s2.mask(s2 == '/', np.nan)

    s_str = s2.astype(str).str.strip()
    s_str = s_str.replace({'nan': np.nan, 'NaN': np.nan, 'None': np.nan, 'NONE': np.nan, '': np.nan})
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
    """Converts Ki67 values into a binary feature (>50 -> 1, <=50 -> 0)."""
    if 'Ki67' not in orig_to_safe:
        return
    safe_ki67 = orig_to_safe['Ki67']
    ki67_numeric = pd.to_numeric(df_in[safe_ki67].replace('/', np.nan), errors='coerce')
    ki67_bin = pd.Series(np.nan, index=df_in.index, dtype='float64')
    ki67_bin[ki67_numeric > 50] = 1.0
    ki67_bin[ki67_numeric <= 50] = 0.0
    df_in[safe_ki67] = ki67_bin

try:
    df = pd.read_excel(file_path)
    print(f"Successfully loaded dataset: {len(df)} rows, {len(df.columns)} columns.")
    
    if target_col not in df.columns:
        raise ValueError(f"Error: Target column '{target_col}' not found in Excel columns.")

    df = df.dropna(subset=[target_col])
    
    safe_names, safe_to_orig, orig_to_safe = make_unique_safe_names(df.columns.tolist())
    df.columns = safe_names
    safe_target = orig_to_safe[target_col]

    df[safe_target], target_mapping = coerce_binary_target(df[safe_target])
    if target_mapping is not None:
        print(f"Notice: Categorical target detected. Auto-mapped to 0/1 configuration parameters: {target_mapping}")

    if 'Ki67' in orig_to_safe:
        safe_ki67 = orig_to_safe['Ki67']
        ki67_numeric = pd.to_numeric(df[safe_ki67].replace('/', np.nan), errors='coerce')
        ki67_bin = pd.Series(np.nan, index=df.index, dtype='float64')
        ki67_bin[ki67_numeric > 50] = 1.0
        ki67_bin[ki67_numeric <= 50] = 0.0
        df[safe_ki67] = ki67_bin
        print(
            "Notice: Ki67 converted into binary feature ranges (>50=1, <=50=0)."
            f" Counts: 0={int((df[safe_ki67]==0).sum())}, 1={int((df[safe_ki67]==1).sum())}, NA={int(df[safe_ki67].isna().sum())}"
        )

    excluded_safe = {orig_to_safe[col] for col in exclude_cols if col in orig_to_safe}

    missing_exclusions = [col for col in exclude_cols if col not in orig_to_safe]
    if missing_exclusions:
        print(f"Notice: The following columns were not found in data and cannot be excluded: {missing_exclusions}")
    excluded_present = [safe_to_orig[col] for col in excluded_safe if col in safe_to_orig]
    if excluded_present:
        print(f"Excluding these columns from analysis: {excluded_present}")
    
except Exception as e:
    print(f"Data loading error: {e}")
    exit()

# ==========================================
# 3. Automated Univariable Analysis
# ==========================================
results_list = []
features = [col for col in df.columns if col != safe_target and col not in excluded_safe]

print("\nRunning Univariable Analyses...")

for safe_col in features:
    original_name = safe_to_orig[safe_col]

    non_na_unique = df[safe_col].dropna().nunique()
    if non_na_unique < 2:
        print(f"Warning: Feature '{original_name}' has insufficient unique values. Skipping.")
        continue
    
    # --- A. Feature Type Determination ---
    unique_count = df[safe_col].nunique(dropna=True)
    is_categorical = False
    is_ordinal_trend = original_name in ORDINAL_TREND_VARS
    if is_ordinal_trend:
        is_categorical = False
    elif original_name in FORCE_CATEGORICAL:
        is_categorical = True
    elif original_name in FORCE_CONTINUOUS:
        is_categorical = False
    else:
        if df[safe_col].dtype == 'object' or df[safe_col].dtype.name == 'category':
            is_categorical = True
        elif unique_count < cat_threshold:
            is_categorical = True
    
    # --- B. Clean and Format Feature Data Arrays ---
    tmp = df[[safe_target, safe_col]].copy()
    if is_ordinal_trend:
        tmp[safe_col] = parse_ordinal_trend(tmp[safe_col])
    else:
        tmp[safe_col] = tmp[safe_col].mask(tmp[safe_col] == '/', np.nan)
        if is_categorical:
            tmp[safe_col] = normalize_categorical_series(tmp[safe_col])
        else:
            tmp[safe_col] = pd.to_numeric(tmp[safe_col], errors='coerce')

    tmp = tmp.dropna(subset=[safe_target, safe_col])
    n_used = int(len(tmp))
    n_event = int((tmp[safe_target] == 1).sum())
    n_nonevent = int((tmp[safe_target] == 0).sum())

    if n_event == 0 or n_nonevent == 0:
        print(
            f"Warning: Target outcome has no variance for feature '{original_name}' (1={n_event}, 0={n_nonevent}). Skipping."
        )
        continue

    if n_used < 10:
        print(f"Warning: Sample size too small for feature '{original_name}' (n={n_used}). Skipping.")
        continue

    # --- C. Equation Formula String Setup ---
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
    
    # --- D. Model Optimization ---
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

        # --- E. Evaluate Likelihood Ratio Test P-values ---
        overall_p = np.nan
        try:
            null_model = smf.logit(formula=f"{safe_target} ~ 1", data=tmp).fit(disp=0, maxiter=logit_maxiter)
            lr_stat = 2.0 * (float(model.llf) - float(null_model.llf))
            df_diff = int(model.df_model - null_model.df_model)
            if df_diff > 0 and np.isfinite(lr_stat):
                overall_p = float(chi2.sf(lr_stat, df_diff))
        except Exception:
            overall_p = np.nan
        
        # --- F. Extract Metrics ---
        params = model.params
        conf = model.conf_int()
        pvalues = model.pvalues

        cat_levels_in_params = []

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
                var_type_label = 'Ordinal (Trend)'
            else:
                var_type_label = 'Categorical' if is_categorical else 'Continuous'

            if is_categorical:
                match = re.search(r'\[T\.(.*?)\]', term)
                if match:
                    level = match.group(1)
                    comp_group = level
                    cat_levels_in_params.append(level)

            results_list.append({
                'Variable Name': display_name,
                'Comparison Group': comp_group,
                'Reference Group': ref_group,
                'Variable Type': var_type_label,
                'Sample Size n': n_used,
                'Events (=1)': n_event,
                'Non-events (=0)': n_nonevent,
                'Overall P-value (LR)': overall_p,
                'OR': or_val,
                '95% CI Lower': ci_lower,
                '95% CI Upper': ci_upper,
                'P-value': p_val,
                '_term': term
            })

    except ConvergenceWarning:
        # Retry with L-BFGS solver if standard NR defaults hit constraints
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
                        var_type_label = 'Ordinal (Trend)'
                    else:
                        var_type_label = 'Categorical' if is_categorical else 'Continuous'

                    if is_categorical:
                        match = re.search(r'\[T\.(.*?)\]', term)
                        if match:
                            comp_group = match.group(1)

                    results_list.append({
                        'Variable Name': display_name,
                        'Comparison Group': comp_group,
                        'Reference Group': ref_group,
                        'Variable Type': var_type_label,
                        'Sample Size n': n_used,
                        'Events (=1)': n_event,
                        'Non-events (=0)': n_nonevent,
                        'Overall P-value (LR)': overall_p,
                        'OR': or_val,
                        '95% CI Lower': ci_lower,
                        '95% CI Upper': ci_upper,
                        'P-value': p_val,
                        '_term': term
                    })

                continue
            except Exception:
                pass

        print(f"Warning: Model for variable '{original_name}' failed to converge and was skipped.")
    except PerfectSeparationError:
        print(f"Warning: Perfect Separation occurred for variable '{original_name}'. Skipped.")
    except Exception as e:
        print(f"Warning: Analysis failed for variable '{original_name}'. Reason: {str(e)[:100]}")

# ==========================================
# 4. Univariable Output Processing
# ==========================================
if results_list:
    res_df = pd.DataFrame(results_list)
    uni_res_df = res_df.copy()
    
    if '_term' in res_df.columns:
        res_df = res_df.drop(columns=['_term'])

    cols_to_round = ['OR', '95% CI Lower', '95% CI Upper', 'P-value', 'Overall P-value (LR)']
    res_df[cols_to_round] = res_df[cols_to_round].round(3)
    
    res_df['95% CI'] = res_df.apply(lambda x: f"{x['95% CI Lower']} - {x['95% CI Upper']}", axis=1)

    def _item_display_row(row: pd.Series) -> str:
        var = str(row.get('Variable Name', ''))
        var_type = str(row.get('Variable Type', ''))
        comp = str(row.get('Comparison Group', '-'))
        if var_type == 'Categorical' and comp not in ('-', '', 'nan', 'None'):
            return _display_level(var, comp, mode='plot')
        return _display_name(var, mode='plot')

    def _ref_display_row(row: pd.Series) -> str:
        var = str(row.get('Variable Name', ''))
        var_type = str(row.get('Variable Type', ''))
        ref = str(row.get('Reference Group', '-'))
        if var_type == 'Categorical' and ref not in ('-', '', 'nan', 'None'):
            return _display_level(var, ref, mode='plot')
        return '-'

    res_df['Item (Display)'] = res_df.apply(_item_display_row, axis=1)
    res_df['Reference (Display)'] = res_df.apply(_ref_display_row, axis=1)
    
    final_df = res_df[
        ['Item (Display)', 'Reference (Display)', 'Variable Name', 'Comparison Group', 'Reference Group', 'Variable Type', 'Sample Size n', 'Events (=1)', 'Non-events (=0)', 'Overall P-value (LR)', 'OR', '95% CI', 'P-value']
    ]
    
    print("\n" + "="*50)
    print("Univariable Analysis Complete! Previewing top rows:")
    print("="*50)
    print(final_df.head(10).to_markdown(index=False))
    
    output_filename = 'Univariable_Analysis_Results.xlsx'
    try:
        final_df.to_excel(output_filename, index=False)
        print(f"\nFull univariable output exported to: {output_filename}")
    except PermissionError:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        alt_filename = f"Univariable_Analysis_Results_{ts}.xlsx"
        final_df.to_excel(alt_filename, index=False)
        print(
            f"\nNotice: File target locked or open elsewhere. Alternate destination used: {alt_filename}"
        )
    
else:
    print("Zero valid results were processed. Review input configurations.")
    uni_res_df = None


# ==========================================
# 5. Multivariable Logistic Regression
# ==========================================
print("\n=== Running Multivariable Logistic Regression ===")

try:
    mv_df_raw = pd.read_excel(multivar_file_path)
    print(f"Multivariable data successfully loaded: {multivar_file_path} ({len(mv_df_raw)} rows, {len(mv_df_raw.columns)} columns).")
except Exception as e:
    print(f"Multivariable data loading failed: {e}")
    mv_df_raw = None

if mv_df_raw is None:
    print("Skipping multivariable logic: input file reference path could not be loaded.")
else:
    mv_safe_names, mv_safe_to_orig, mv_orig_to_safe = make_unique_safe_names(mv_df_raw.columns.tolist())
    mv_df_raw.columns = mv_safe_names

    if target_col not in mv_orig_to_safe:
        print(f"Skipping multivariable tracking: outcome target token missing: '{target_col}'")
    else:
        mv_safe_target = mv_orig_to_safe[target_col]

        try:
            mv_df_raw[mv_safe_target], mv_target_mapping = coerce_binary_target(mv_df_raw[mv_safe_target])
            if mv_target_mapping is not None:
                print(f"Notice: Target converted successfully for multivariable operations: {mv_target_mapping}")
        except Exception as e:
            print(f"Multivariable model cancelled: Dependent outcome conversion failure: {e}")
            mv_safe_target = None

        if mv_safe_target is not None:
            binarize_ki67_inplace(mv_df_raw, mv_orig_to_safe)

        mv_excluded_safe = {mv_orig_to_safe[col] for col in exclude_cols if col in mv_orig_to_safe}

        if mv_safe_target is not None:
            mv_orig_list = dedupe_preserve_order(MULTIVAR_FEATURES_ORIG)
            mv_missing = [v for v in mv_orig_list if v not in mv_orig_to_safe]
            if mv_missing:
                print(f"Notice: The following feature elements are missing in multivariable dataset files and will be skipped: {mv_missing}")

            mv_present_orig = [v for v in mv_orig_list if v in mv_orig_to_safe]
            mv_present_safe = [mv_orig_to_safe[v] for v in mv_present_orig]

            mv_filtered: List[Tuple[str, str]] = []
            for orig, safe in zip(mv_present_orig, mv_present_safe):
                if safe == mv_safe_target:
                    continue
                if safe in mv_excluded_safe:
                    continue
                mv_filtered.append((orig, safe))

            if not mv_filtered:
                print("No feature metrics remain for multivariable modeling steps.")
            else:
                mv_data = mv_df_raw[[mv_safe_target] + [safe for _, safe in mv_filtered]].copy()

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

                        if MV_COLLAPSE_RARE_LEVELS:
                            vc = mv_data[safe].value_counts(dropna=True)
                            rare_levels = vc[vc < MV_MIN_LEVEL_COUNT].index.tolist()
                            if rare_levels:
                                mv_data[safe] = mv_data[safe].where(~mv_data[safe].isin(rare_levels), other='Other')

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

                # Execute drop logic to implement standard complete-case tracking arrays
                mv_data = mv_data.dropna(subset=[mv_safe_target] + [safe for _, safe in mv_filtered])
                mv_n_used = int(len(mv_data))
                mv_n_event = int((mv_data[mv_safe_target] == 1).sum())
                mv_n_nonevent = int((mv_data[mv_safe_target] == 0).sum())
                print(f"Multivariable case counts: n={mv_n_used}, Events(=1)={mv_n_event}, Non-events(=0)={mv_n_nonevent}")

                if mv_continuous_safe_cols and MV_PRUNE_CONTINUOUS_BY_CORR:
                    protected_set = set(mv_continuous_protected)
                    prune_candidates = [c for c in mv_continuous_safe_cols if c not in protected_set]
                    pruned = prune_continuous_by_spearman_corr(
                        mv_data, prune_candidates, threshold=MV_CORR_THRESHOLD
                    )
                    dropped = [c for c in prune_candidates if c not in pruned]
                    if dropped:
                        mv_continuous_safe_cols = mv_continuous_protected + pruned
                        mv_terms = [t for t in mv_terms if t not in dropped]
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
                        f"Notice: Standardized and clipped {len(mv_continuous_safe_cols)} continuous columns within limit bounds [-{MV_Z_CLIP}, {MV_Z_CLIP}]."
                    )

                if mv_n_used < 20 or mv_n_event == 0 or mv_n_nonevent == 0:
                    print("Warning: Sample properties or variance distribution insufficient for multivariable operations.")
                else:
                    mv_formula = f"{mv_safe_target} ~ " + " + ".join(mv_terms)

                    def _fit_mv(method_override: Optional[str] = None):
                        """Fits multivariable model and handles collinearity using QR pivoting to maintain safe matrix inversion ranks."""
                        y, X = dmatrices(mv_formula, mv_data, return_type='dataframe')
                        y = y.iloc[:, 0].astype('float64')
                        X = X.astype('float64')

                        const_cols: List[str] = []
                        for c in X.columns:
                            if c == 'Intercept':
                                continue
                            if int(X[c].nunique(dropna=False)) <= 1:
                                const_cols.append(c)
                        if const_cols:
                            print(f"Notice: Dropped {len(const_cols)} structural invariant tracking vectors from design matrix.")
                            X = X.drop(columns=const_cols, errors='ignore')

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
                                'Design matrices returned empty parameters. Exiting.'
                            )
                        eps = np.finfo(diag.dtype).eps
                        tol = float(eps * max(X.shape) * diag.max())
                        rank = int((diag > tol).sum())
                        if rank <= 0:
                            raise ValueError(
                                'Rank determination returned 0 matrix size calculations. Check features for absolute invariant traits.'
                            )
                        keep_idx = sorted(piv[:rank])
                        if rank < X.shape[1]:
                            dropped_cols = [X.columns[i] for i in piv[rank:]]
                            print(
                                f"Warning: Matrix rank fell short of parameter size dimensions. "
                                f"Automatically dropped {len(dropped_cols)} collinear columns to finish modeling paths safely."
                            )
                        X = X.iloc[:, keep_idx]

                        n_params = max(1, int(X.shape[1]) - 1)
                        epv_event = float(mv_n_event) / float(n_params)
                        epv_nonevent = float(mv_n_nonevent) / float(n_params)
                        if epv_event < 10.0 or epv_nonevent < 10.0:
                            print(
                                f"Warning: Low Events Per Variable ratio (EPV Event={epv_event:.2f}, Non-Event={epv_nonevent:.2f}, Parameters={n_params})."
                                " Standard error calculations may be unstable due to overfitting constraints."
                            )

                        fit_kwargs: Dict[str, Any] = {'disp': 0, 'maxiter': logit_maxiter}
                        chosen = method_override or logit_method
                        if not chosen:
                            chosen = 'lbfgs'
                        fit_kwargs['method'] = chosen

                        try:
                            with warnings.catch_warnings():
                                warnings.filterwarnings('error', category=ConvergenceWarning)
                                warnings.filterwarnings('error', category=RuntimeWarning)
                                warnings.filterwarnings('ignore', category=HessianInversionWarning)
                                return sm.Logit(y, X).fit(**fit_kwargs)
                        except RuntimeWarning:
                            print(
                                "Warning: Floating point exception triggered during maximum likelihood solving. "
                                "Falling back to regularization solver path rules (fit_regularized)."
                            )
                            return sm.Logit(y, X).fit_regularized(
                                alpha=MV_REG_ALPHA, L1_wt=MV_REG_L1_WT, disp=0, maxiter=logit_maxiter
                            )
                        except Exception as e:
                            msg = str(e)
                            if 'Singular matrix' in msg or 'singular' in msg.lower():
                                print(
                                    "Warning: Execution aborted by singular matrix state. "
                                    "Falling back to regularization solver path rules (fit_regularized)."
                                )
                                return sm.Logit(y, X).fit_regularized(
                                    alpha=MV_REG_ALPHA, L1_wt=MV_REG_L1_WT, disp=0, maxiter=logit_maxiter
                                )
                            raise

                    mv_model = None
                    try:
                        mv_model = _fit_mv()
                    except ConvergenceWarning:
                        if logit_method is None:
                            try:
                                mv_model = _fit_mv(method_override='lbfgs')
                            except Exception as e:
                                print(f"Warning: Multivariable retry failed: {str(e)[:120]}")
                                mv_model = None
                    except PerfectSeparationError:
                        print("Warning: Aborted due to Perfect Separation inside tracking levels. Review data groups manually.")
                        mv_model = None
                    except Exception as e:
                        print(f"Warning: Multivariable model estimation step failed: {str(e)[:120]}")
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

                            base_safe = term
                            comp_group = "-"
                            ref_group = "-"

                            if term.startswith('C('):
                                m_var = re.match(r"C\(([^,\)]+)", term)
                                if m_var:
                                    base_safe = m_var.group(1)
                                m_level = re.search(r"\[T\.(.*?)\]", term)
                                if m_level:
                                    comp_group = m_level.group(1)

                            orig_name = mv_safe_to_orig.get(base_safe, base_safe)
                            if orig_name in ORDINAL_TREND_VARS:
                                var_type_label = 'Ordinal (Trend)'
                            else:
                                var_type_label = 'Categorical' if orig_name in mv_cat_vars_orig else 'Continuous'
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
                                'Variable Name': orig_name,
                                'Comparison Group': comp_group,
                                'Reference Group': ref_group,
                                'Variable Type': var_type_label,
                                'Sample Size n': mv_n_used,
                                'Events (=1)': mv_n_event,
                                'Non-events (=0)': mv_n_nonevent,
                                'OR': or_val,
                                '95% CI Lower': ci_lower,
                                '95% CI Upper': ci_upper,
                                'P-value': p_val,
                            })

                        if mv_rows:
                            mv_df = pd.DataFrame(mv_rows)
                            mv_res_df = mv_df.copy()
                            mv_df[['OR', '95% CI Lower', '95% CI Upper', 'P-value']] = mv_df[
                                ['OR', '95% CI Lower', '95% CI Upper', 'P-value']
                            ].round(3)
                            mv_df['95% CI'] = mv_df.apply(
                                lambda x: f"{x['95% CI Lower']} - {x['95% CI Upper']}", axis=1
                            )

                            mv_df['Item (Display)'] = mv_df.apply(
                                lambda r: _display_level(str(r['Variable Name']), str(r['Comparison Group']), mode='plot')
                                if str(r['Variable Type']) == 'Categorical' and str(r['Comparison Group']) not in ('-', '', 'nan', 'None')
                                else _display_name(str(r['Variable Name']), mode='plot'),
                                axis=1,
                            )
                            mv_df['Reference (Display)'] = mv_df.apply(
                                lambda r: _display_level(str(r['Variable Name']), str(r['Reference Group']), mode='plot')
                                if str(r['Variable Type']) == 'Categorical' and str(r['Reference Group']) not in ('-', '', 'nan', 'None')
                                else '-',
                                axis=1,
                            )
                            mv_final = mv_df[
                                ['Item (Display)', 'Reference (Display)', 'Variable Name', 'Comparison Group', 'Reference Group', 'Variable Type', 'Sample Size n', 'Events (=1)', 'Non-events (=0)', 'OR', '95% CI', 'P-value']
                            ]

                            mv_output = 'Multivariable_Analysis_Results.xlsx'
                            try:
                                mv_final.to_excel(mv_output, index=False)
                                print(f"Multivariable model details successfully saved to: {mv_output}")
                            except PermissionError:
                                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                                alt = f"Multivariable_Analysis_Results_{ts}.xlsx"
                                mv_final.to_excel(alt, index=False)
                                print(
                                    f"Notice: Target destination is locked or busy. Exported alternative file instead: {alt}"
                                )
                        else:
                            print("Multivariable execution finalized but returned zero tracking arrays.")
                            mv_res_df = None


# ==========================================
# 6. Combined Analysis Summary Table
# ==========================================
print("\n=== Generating Consolidated Analysis Summary Table ===")


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
    need_cols = {'Variable Name', 'Comparison Group', 'Reference Group', 'OR', '95% CI Lower', '95% CI Upper', 'P-value'}
    if not need_cols.issubset(set(res.columns)):
        return lookup, ref_map
    for _, row in res.iterrows():
        var = str(row.get('Variable Name', ''))
        comp = str(row.get('Comparison Group', '-'))
        ref = str(row.get('Reference Group', '-'))
        if var and ref and ref not in ('-', 'nan', 'None') and var not in ref_map:
            ref_map[var] = ref
        lookup[(var, comp)] = {
            'OR': row.get('OR', np.nan),
            '95% CI Lower': row.get('95% CI Lower', np.nan),
            '95% CI Upper': row.get('95% CI Upper', np.nan),
            'P-value': row.get('P-value', np.nan),
            'Reference Group': ref,
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
        raise ValueError('Required univariable references are missing. Verify execution paths are intact.')

    summary_base = df.copy()
    summary_base = summary_base.dropna(subset=[safe_target])
    n0_all = int((summary_base[safe_target] == 0).sum())
    n1_all = int((summary_base[safe_target] == 1).sum())

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
                    uni_txt = _format_or_ci_p(u['OR'], u['95% CI Lower'], u['95% CI Upper'], u['P-value']) if u else ''
                    m = mv_lookup.get((orig_name, lv))
                    mv_txt = _format_or_ci_p(m['OR'], m['95% CI Lower'], m['95% CI Upper'], m['P-value']) if m else ''

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
            uni_txt = _format_or_ci_p(u['OR'], u['95% CI Lower'], u['95% CI Upper'], u['P-value']) if u else ''
            m = mv_lookup.get((orig_name, '-'))
            mv_txt = _format_or_ci_p(m['OR'], m['95% CI Lower'], m['95% CI Upper'], m['P-value']) if m else ''

            rows_out.append({
                'Characteristic': _display_name(orig_name, mode='plot'),
                col_m0: m0_txt,
                col_m1: m1_txt,
                'OR (univariable)': uni_txt,
                'OR (multivariable)': mv_txt,
            })

    if rows_out:
        out_df = pd.DataFrame(rows_out)
        out_file = 'Consolidated_Analysis_Summary_Table.xlsx'
        try:
            out_df.to_excel(out_file, index=False)
            print(f"Consolidated publication table successfully exported to: {out_file}")
        except PermissionError:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            alt = f"Consolidated_Analysis_Summary_Table_{ts}.xlsx"
            out_df.to_excel(alt, index=False)
            print(f"Notice: Destination path target is locked. Saved to alternate filename instead: {alt}")
    else:
        print('Consolidated table production skipped: Zero valid tracking features found.')
except Exception as e:
    print(f"Consolidated table compilation skipped due to runtime errors: {str(e)[:200]}")