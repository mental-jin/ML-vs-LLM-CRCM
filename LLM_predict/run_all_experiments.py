"""
批量运行入口：
    - 从 `model_matrix.json` 读取模型列表和实验脚本列表
    - 依次对每个模型执行每个实验脚本
    - 每个脚本仍然可以单独运行，不影响单模型测试
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_FILE = ROOT_DIR / "model_matrix.json"


def load_config(config_file: Path) -> dict:
    # 统一从一个配置文件读取模型列表和脚本列表，避免手动改代码。
    with config_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_model_list(config: dict) -> list[str]:
    # 兼容两种写法：字典结构的模型项，或直接写字符串模型名。
    models = []
    for item in config.get("models", []):
        if isinstance(item, dict):
            if item.get("enabled", True) and item.get("name"):
                models.append(item["name"])
        elif isinstance(item, str):
            models.append(item)
    return models


def build_script_list(config: dict) -> list[str]:
    # 实验脚本列表保持固定，批量入口只负责按清单调度。
    scripts = config.get("scripts", [])
    return [script for script in scripts if isinstance(script, str)]


def run_job(script_path: Path, model_name: str) -> tuple[str, str, int]:
    """运行单个脚本与模型组合，返回脚本名、模型名和退出码。"""
    # 每个任务都单独启动一个 Python 子进程，彼此隔离。
    result = subprocess.run(
        [sys.executable, str(script_path), "--model-name", model_name],
        cwd=str(ROOT_DIR.parent.parent),
        check=False,
    )
    return script_path.name, model_name, result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch run all LLM prediction experiments for all configured models.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_FILE), help="Path to model_matrix.json")
    # 默认并行；如需临时切换执行方式，可在这里调整默认值。
    parser.add_argument("--parallel", action="store_true", default=True, help="Run jobs in parallel")
    parser.add_argument("--max-workers", type=int, default=10, help="Max parallel jobs when --parallel is enabled")
    args = parser.parse_args()

    config_file = Path(args.config).resolve()
    config = load_config(config_file)
    models = build_model_list(config)
    scripts = build_script_list(config)

    if not models:
        print(f"No enabled models found in {config_file}")
        return 1

    if not scripts:
        print(f"No scripts found in {config_file}")
        return 1

    print(f"Loaded {len(models)} models and {len(scripts)} scripts from {config_file}")

    jobs = []
    for model_name in models:
        for script_name in scripts:
            script_path = ROOT_DIR / script_name
            if not script_path.exists():
                print(f"[SKIP] Missing script: {script_path}")
                continue
            jobs.append((script_path, model_name))

    if not jobs:
        print("No runnable jobs found.")
        return 1

    if args.parallel:
        # 并行模式：一次提交所有任务，用 max_workers 控制同时运行的子进程数。
        print(f"Running {len(jobs)} jobs in parallel with max_workers={args.max_workers}")
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(run_job, script_path, model_name): (script_path.name, model_name)
                for script_path, model_name in jobs
            }

            for future in as_completed(futures):
                script_name, model_name = futures[future]
                try:
                    finished_script, finished_model, returncode = future.result()
                except Exception as exc:
                    print(f"[FAIL] {script_name} for {model_name} raised {exc!r}")
                    return 1

                if returncode != 0:
                    print(f"[FAIL] {finished_script} for {finished_model} exited with code {returncode}")
                    return returncode

        print("All experiments completed successfully.")
        return 0

    # 默认串行模式：按模型、按脚本顺序逐个执行，便于调试和复现实验。
    for model_name in models:
        print("=" * 80)
        print(f"Model: {model_name}")
        print("=" * 80)

        for script_name in scripts:
            script_path = ROOT_DIR / script_name
            if not script_path.exists():
                print(f"[SKIP] Missing script: {script_path}")
                continue

            print(f"[RUN] {script_name} --model-name {model_name}")
            result = subprocess.run(
                [sys.executable, str(script_path), "--model-name", model_name],
                cwd=str(ROOT_DIR.parent.parent),
                check=False,
            )

            if result.returncode != 0:
                print(f"[FAIL] {script_name} for {model_name} exited with code {result.returncode}")
                return result.returncode

    print("All experiments completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())