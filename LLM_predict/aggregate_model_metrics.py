"""
汇总模型评估指标（accuracy）脚本

用途：扫描 `results/<experiment>/<model>/` 目录下的汇总文件（如 summary.json 或 summary_2p_2n.json），
提取每个模型在各个实验中的 `accuracy` 值，生成一个聚合 JSON。

文件保存到 `results/aggregate_model_metrics.json`。

期望输出格式示例：
{
    "deepseek-v4-pro": {
        "1_zero_shot_naive": 0.7,
        "2_few_shot": 0.62,
        "3_prompt_aug": 0.78,
        "4_mix_few_aug": 0.82
    }
}

说明：
- 本脚本通过 `EXPERIMENT_SUMMARY_FILE` 字典确定每个实验下要读取的 summary 文件名。
- 如果找不到文件或无法解析 `accuracy`，对应位置会写 `null`（Python 中为 None）。
"""

import os
import json
from typing import Dict, Any


RESULTS_ROOT = 'LLM/data/results'
# mapping: experiment -> expected summary filename under results/<experiment>/<model>/
EXPERIMENT_SUMMARY_FILE = {
    '1_zero_shot_naive': 'summary_metrics.json',
    '2_few_shot': 'summary_2p_2n.json',
    '3_prompt_aug': 'summary_metrics.json',
    '4_mix_few_aug': 'summary_2p_2n.json',
}


def safe_load_json(path: str) -> Dict[str, Any] | None:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def collect_metrics(results_root: str = RESULTS_ROOT) -> Dict[str, Dict[str, float | None]]:
    agg: Dict[str, Dict[str, float | None]] = {}

    for exp, summary_name in EXPERIMENT_SUMMARY_FILE.items():
        exp_dir = os.path.join(results_root, exp)
        if not os.path.isdir(exp_dir):
            # skip missing experiment directories
            continue

        for model_name in os.listdir(exp_dir):
            model_dir = os.path.join(exp_dir, model_name)
            if not os.path.isdir(model_dir):
                continue

            summary_path = os.path.join(model_dir, summary_name)
            data = safe_load_json(summary_path)

            accuracy = None
            if isinstance(data, dict):
                # try common keys
                if 'accuracy' in data:
                    accuracy = data.get('accuracy')
                elif 'metrics' in data and isinstance(data['metrics'], dict):
                    accuracy = data['metrics'].get('accuracy')

                # normalize numeric-like strings to float
                try:
                    if accuracy is not None:
                        accuracy = float(accuracy)
                except Exception:
                    # leave as-is (could be None)
                    pass

            # ensure model key exists
            agg.setdefault(model_name, {})
            agg[model_name][exp] = accuracy

    return agg


def main():
    agg = collect_metrics()
    out_path = os.path.join(RESULTS_ROOT, 'aggregate_model_metrics.json')
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)
    print(f'已保存汇总文件: {out_path}')


if __name__ == '__main__':
    main()
