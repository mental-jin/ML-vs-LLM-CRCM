"""
Batch run entry point:
    - Read the list of models and experiment scripts from `model_matrix.json`
    - Execute each experiment script for each model sequentially
    - Each script can still be run individually without affecting single-model testing
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
    # Read model list and script list from a single configuration file uniformly to avoid manual code changes.
    with config_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_model_list(config: dict) -> list[str]:
    # Compatible with two formats: model items as dictionary structures, or model names directly as strings.
    models = []
    for item in config.get("models", []):
        if isinstance(item, dict):
            if item.get("enabled", True) and item.get("name"):
                models.append(item["name"])
        elif isinstance(item, str):
            models.append(item)
    return models


def build_script_list(config: dict) -> list[str]:
    # The list of experiment scripts remains fixed; the batch entry is only responsible for scheduling according to the list.
    scripts = config.get("scripts", [])
    return [script for script in scripts if isinstance(script, str)]


def run_job(script_path: Path, model_name: str) -> tuple[str, str, int]:
    """Run a single script and model combination, returning script name, model name, and exit code."""
    # Each task starts a separate Python subprocess, keeping them isolated from one another.
    result = subprocess.run(
        [sys.executable, str(script_path), "--model-name", model_name],
        cwd=str(ROOT_DIR.parent.parent),
        check=False,
    )
    return script_path.name, model_name, result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch run all LLM prediction experiments for all configured models.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_FILE), help="Path to model_matrix.json")
    # Parallel by default; if you need to temporarily switch the execution mode, you can adjust the default value here.
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
        # Parallel mode: Submit all tasks at once, controlling concurrent subprocesses with max_workers.
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

    # Default sequential mode: Run step-by-step by model and script order, easy for debugging and experiment replication.
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