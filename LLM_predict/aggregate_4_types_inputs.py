"""
汇总 4 种输入类型的评估结果并输出聚合 JSON

用途：遍历 `results/0_4_types_inputs/<model>/` 下的四个 summary 文件，提取每个模型在
不同输入处理方式下的 `accuracy` 值，生成一个聚合文件 `aggregate_4_types_inputs.json`。

期望输出 JSON 格式示例：
{
    "deepseek-v4-pro": {
        "1_Original": 0.7,
        "2_Selected": 0.62,
        "3_Rounded": 0.78,
        "4_Selected_Rounded": 0.82
    }
}

字段说明：
- 外层键（例如 `deepseek-v4-pro`）：模型名，对应目录 `results/0_4_types_inputs/<模型名>/`。
- 内层键（`1_Original` / `2_Selected` / `3_Rounded` / `4_Selected_Rounded`）：四种输入类型的标识。
- 值：对应 summary 文件中的 `accuracy`（数值），若无法读取或不存在则为 `null`。

文件对应关系（脚本内常量 `FILE_MAP`）：
- `1_Original` -> `1_Original_summary_metrics.json`
- `2_Selected` -> `2_Selected_summary_metrics.json`
- `3_Rounded` -> `3_Rounded_summary_metrics.json`
- `4_Selected_Rounded` -> `4_Selected_Rounded_summary_metrics.json`

输出位置：`results/0_4_types_inputs/aggregate_4_types_inputs.json`

四种输入类型的区别：
- `1_Original`：使用原始完整输入（不做列筛选或值变换）。
- `2_Selected`：只保留预先选定的一组列作为模型输入，去除其他列。
- `3_Rounded`：在原始结构上将所有数值型字段四舍五入到 2 位小数，保留全部列。
- `4_Selected_Rounded`：先选取特定列，再对这些数值字段四舍五入到 2 位小数（选列+四舍五入）。
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
    print(f'已保存汇总：{out_path}')


if __name__ == '__main__':
    main()
