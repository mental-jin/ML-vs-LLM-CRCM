"""
Aggregate evaluation results of 4 input types and output aggregated JSON.

Purpose: Traverse the four summary files under `results/0_4_types_inputs/<model>/`, 
extract the `accuracy` value for each model under different input processing methods, 
and generate an aggregated file `aggregate_4_types_inputs.json`.

Expected output JSON format example:
{
    "deepseek-v4-pro": {
        "1_Original": 0.7,
        "2_Selected": 0.62,
        "3_Rounded": 0.78,
        "4_Selected_Rounded": 0.82
    }
}

Field Descriptions:
- Outer key (e.g., `deepseek-v4-pro`): Model name, corresponding to the directory `results/0_4_types_inputs/<model_name>/`.
- Inner key (`1_Original` / `2_Selected` / `3_Rounded` / `4_Selected_Rounded`): Identifier for the four input types.
- Value: The corresponding `accuracy` (numerical value) in the summary file, or `null` if it cannot be read or does not exist.

File mappings (script constant `FILE_MAP`):
- `1_Original` -> `1_Original_summary_metrics.json`
- `2_Selected` -> `2_Selected_summary_metrics.json`
- `3_Rounded` -> `3_Rounded_summary_metrics.json`
- `4_Selected_Rounded` -> `4_Selected_Rounded_summary_metrics.json`

Output location: `results/0_4_types_inputs/aggregate_4_types_inputs.json`

Differences among the four input types:
- `1_Original`: Uses the original complete input (no column selection or value transformation).
- `2_Selected`: Keeps only a pre-selected set of columns as model input, removing all other columns.
- `3_Rounded`: Rounds all numerical fields to 2 decimal places based on the original structure, retaining all columns.
- `4_Selected_Rounded`: First selects specific columns, then rounds these numerical fields to 2 decimal places (selection + rounding).
"""

import os
import json
from typing import Dict, Any

RESULTS_DIR = 'LLM/data/results/0_4_types_inputs'
FILE_MAP = {
    '1_Original': '1_Original_summary_metrics.json',
    '2_Selected': '2_Selected_summary_metrics.json',
    '3_Rounded': '3_Rounded_summary_metrics.json',
    '4_Selected_Rounded': '4_Selected_Rounded_summary_metrics.json',
}


def safe_load_json(path: str) -> Dict[str, Any] | None:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def collect_accuracies(results_dir: str = RESULTS_DIR) -> Dict[str, Dict[str, float | None]]:
    agg: Dict[str, Dict[str, float | None]] = {}
    if not os.path.isdir(results_dir):
        return agg

    for model_name in sorted(os.listdir(results_dir)):
        model_dir = os.path.join(results_dir, model_name)
        if not os.path.isdir(model_dir):
            continue

        agg.setdefault(model_name, {})
        for key, fname in FILE_MAP.items():
            path = os.path.join(model_dir, fname)
            data = safe_load_json(path)
            accuracy = None
            if isinstance(data, dict):
                accuracy = data.get('accuracy')
                # try nested fields
                if accuracy is None and 'metrics' in data and isinstance(data['metrics'], dict):
                    accuracy = data['metrics'].get('accuracy')
            try:
                if accuracy is not None:
                    accuracy = float(accuracy)
            except Exception:
                accuracy = None

            agg[model_name][key] = accuracy

    return agg


def main():
    agg = collect_accuracies()
    out_dir = os.path.dirname(RESULTS_DIR)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, 'aggregate_4_types_inputs.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)
    print(f'Summary saved: {out_path}')


if __name__ == '__main__':
    main()