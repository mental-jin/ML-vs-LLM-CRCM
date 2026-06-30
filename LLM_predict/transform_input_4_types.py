"""
Purpose:
    Process and export input data for LLMs:
    - Read samples from original JSON (keyed by ID string), validate and select specified target columns.
    - Support rounding numerical fields and saving different output versions.

Optional configs:
    - `target_columns`: List of target column names used for column selection and validation.
    - `decimals` (in `main`): Number of decimal places for rounding.

Input/Output paths:
    - Default input: `LLM/data/input_4_types/test_set.json` (can be modified via command line)
    - Output examples:
            - `LLM/data/input_4_types/test_set_selected.json`
            - `LLM/data/input_4_types/test_set_rounded.json`
            - `LLM/data/input_4_types/test_set_selected_rounded.json`
"""

import json
from typing import Any, Dict, List
from numbers import Number

target_columns: List[str] = [
    "Carcinoma nodule",
    "PMS2",
    "Differentiation grade",
    "Vascular invasion",
    "KRAS mutant",
    "TNLE",
    "Treg cells %",
    "N stage",
    "CEA",
    "T stage",
    "IL8",
    "Perineural invasion",
    "PLN",
    "mGPS",
    "BRAF mutant",
    "CD3+ count",
    "CD19+ B cells %",
    "Ki67",
    "Tumor size",
    "CD4+ count",
    "Total protein",
    "MSH6",
    "NLR",
    "CA242",
    "LDH",
    "Family history",
    "Colonic obstruction",
    "CA199",
    "NK cells %",
    "CD19+ count",
    "Metastasis",
]


def load_json(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(obj: Any, path: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def collect_all_keys(data: Dict[str, Any]) -> set:
    keys = set()
    for entry in data.values():
        metrics = entry.get('metrics', {}) or {}
        results = entry.get('results', {}) or {}
        keys.update(metrics.keys())
        keys.update(results.keys())
    return keys


def validate_target_columns(data: Dict[str, Any], targets: List[str]) -> None:
    present = collect_all_keys(data)
    missing = [c for c in targets if c not in present]
    if missing:
        raise ValueError(f"Error: The following columns could not be found in the JSON data: {missing}")


def round_value(v: Any, decimals: int) -> Any:
    if isinstance(v, bool):
        return v
    if isinstance(v, Number):
        try:
            return round(v, decimals)
        except Exception:
            return v
    return v


def round_entry(entry: Dict[str, Any], decimals: int) -> Dict[str, Any]:
    out = {'metrics': {}, 'results': {}}
    for k, v in (entry.get('metrics') or {}).items():
        out['metrics'][k] = round_value(v, decimals)
    for k, v in (entry.get('results') or {}).items():
        out['results'][k] = round_value(v, decimals)
    return out


def select_columns(entry: Dict[str, Any], targets: List[str]) -> Dict[str, Any]:
    metrics = entry.get('metrics', {}) or {}
    results = entry.get('results', {}) or {}
    out = {}
    for col in targets:
        if col in metrics:
            out[col] = metrics[col]
        elif col in results:
            out[col] = results[col]
        else:
            out[col] = None
    return out


def main(input_path: str,
         out_selected: str,
         out_rounded: str,
         out_selected_rounded: str,
         decimals: int = 2) -> None:
    print(f"Reading: {input_path}")
    data = load_json(input_path)

    # Validate that target columns exist in data (at least appearing in one sample)
    validate_target_columns(data, target_columns)

    # 1) Select specific column names
    selected = {k: select_columns(v, target_columns) for k, v in data.items()}
    save_json(selected, out_selected)
    print(f"Saved JSON with selected columns: {out_selected}")

    # 2) Round all numerical values to `decimals` places (keep original structure)
    rounded = {k: round_entry(v, decimals) for k, v in data.items()}
    save_json(rounded, out_rounded)
    print(f"Saved rounded JSON: {out_rounded}")

    # 3) Select specific column names and round all numerical values to `decimals` places
    selected_rounded = {}
    for k, v in data.items():
        picked = select_columns(v, target_columns)
        # round picked values
        for col, val in picked.items():
            picked[col] = round_value(val, decimals)
        selected_rounded[k] = picked

    save_json(selected_rounded, out_selected_rounded)
    print(f"Saved selected and rounded JSON: {out_selected_rounded}")


if __name__ == '__main__':
    # Default paths, can be overridden in command line
    input_path = 'LLM/data/input_4_types/test_set.json'
    out_selected = 'LLM/data/input_4_types/test_set_selected.json'
    out_rounded = 'LLM/data/input_4_types/test_set_rounded.json'
    out_selected_rounded = 'LLM/data/input_4_types/test_set_selected_rounded.json'
    main(input_path, out_selected, out_rounded, out_selected_rounded, decimals=2)