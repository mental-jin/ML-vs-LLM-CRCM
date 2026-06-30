"""
说明 (Purpose):
    处理并导出用于 LLM 的输入数据：
    - 从原始 JSON 读取样本（按 id 字符串键），验证并选择指定的目标列，
    - 支持对数值字段进行四舍五入并保存不同输出版本。

可配置项 (Optional configs):
    - `target_columns`: 目标列名列表，用于选择和验证列。
    - `decimals` (在 `main` 中): 四舍五入小数位数。

输入/输出路径 (Input/Output paths):
    - 默认输入：`LLM/data/input_4_types/test_set.json`（可通过命令行修改）
    - 输出示例：
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
        raise ValueError(f"错误：在 JSON 数据中找不到以下列: {missing}")


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
    print(f"读取：{input_path}")
    data = load_json(input_path)

    # 验证目标列在数据中存在（至少在某个样本中出现）
    validate_target_columns(data, target_columns)

    # 1) 选取特定列名
    selected = {k: select_columns(v, target_columns) for k, v in data.items()}
    save_json(selected, out_selected)
    print(f"已保存选取列的 JSON: {out_selected}")

    # 2) 将所有数值保留 decimals 位小数（保留原结构）
    rounded = {k: round_entry(v, decimals) for k, v in data.items()}
    save_json(rounded, out_rounded)
    print(f"已保存四舍五入后的 JSON: {out_rounded}")

    # 3) 选取特定列名，且将所有数值保留 decimals 位小数
    selected_rounded = {}
    for k, v in data.items():
        picked = select_columns(v, target_columns)
        # round picked values
        for col, val in picked.items():
            picked[col] = round_value(val, decimals)
        selected_rounded[k] = picked

    save_json(selected_rounded, out_selected_rounded)
    print(f"已保存选取并四舍五入的 JSON: {out_selected_rounded}")


if __name__ == '__main__':
    # 默认路径，可在命令行中覆盖
    input_path = 'LLM/data/input_4_types/test_set.json'
    out_selected = 'LLM/data/input_4_types/test_set_selected.json'
    out_rounded = 'LLM/data/input_4_types/test_set_rounded.json'
    out_selected_rounded = 'LLM/data/input_4_types/test_set_selected_rounded.json'
    main(input_path, out_selected, out_rounded, out_selected_rounded, decimals=2)
