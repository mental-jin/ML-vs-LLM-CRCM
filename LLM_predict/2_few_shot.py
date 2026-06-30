"""
Purpose:
    Perform Few-Shot evaluation using a fixed candidate pool and a fixed test set:
    - Extract several positive/negative examples from `few_shot_pool.json` and insert them into the Prompt.
    - Predict sample-by-sample on the fixed test set and save the model output and statistical summary.

Optional configs:
    - `MODEL_NAME`, `INPUT_DIR`, `OUTPUT_DIR`, `FEW_SHOT_POS_COUNT`, `FEW_SHOT_NEG_COUNT`, etc.
    - Environment variable `DASHSCOPE_API_KEY` used for API authentication.

Input/Output paths:
    - Input: `LLM/data/input/few_shot_pool.json`, `LLM/data/input/test_set.json`
    - Output: `LLM/data/results/2_few_shot/{MODEL_NAME}/output_{p}p_{n}n.json`
    - Summary: `LLM/data/results/2_few_shot/{MODEL_NAME}/summary_{p}p_{n}n.json`
"""

import os
import json
import time
import argparse
from openai import OpenAI

# ================= 1. Initialize Configuration =================
# Please ensure that DASHSCOPE_API_KEY is set in your environment variables
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def resolve_model_name(default_model_name: str) -> str:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--model-name", default=os.getenv("MODEL_NAME", default_model_name))
    args = parser.parse_args()
    return args.model_name

MODEL_NAME = resolve_model_name("kimi-k2.6")
INPUT_DIR = "LLM/data/input"
OUTPUT_DIR = f"LLM/data/results/2_few_shot/{MODEL_NAME}"  # Modified output directory to distinguish experiments

# [Critical Modification]: No longer reading the full dataset, but reading the "candidate pool" and "test set" separately
POOL_FILE = f"{INPUT_DIR}/few_shot_pool.json"  # Candidate pool providing examples
TEST_FILE = f"{INPUT_DIR}/test_set.json"       # Fixed test set

# --- Few-Shot Core Configuration ---
FEW_SHOT_POS_COUNT = 2  # Number of positive samples extracted from the candidate pool (Metastasized, label is 1)
FEW_SHOT_NEG_COUNT = 2  # Number of negative samples extracted from the candidate pool (Non-metastasized, label is 0)
# -------------------------

# To prevent experimental results with different ratios from overwriting each other, it is recommended to add the number of examples to the output file name
OUTPUT_FILE = f"{OUTPUT_DIR}/output_{FEW_SHOT_POS_COUNT}p_{FEW_SHOT_NEG_COUNT}n.json"
SUMMARY_FILE = f"{OUTPUT_DIR}/summary_{FEW_SHOT_POS_COUNT}p_{FEW_SHOT_NEG_COUNT}n.json"

def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

make_dirs(OUTPUT_DIR)
# =================================================

def load_json_if_exists(file_path: str) -> dict:
    """Load existing JSON results; return an empty dictionary if it does not exist or is empty."""
    if not os.path.exists(file_path):
        return {}

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}

def save_json(file_path: str, data: dict) -> None:
    """Write the dictionary to a JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def summarize_results(results: dict) -> tuple[int, int]:
    """Count the number of valid samples and correct samples."""
    valid_samples = 0
    correct_predictions = 0

    for item in results.values():
        if not isinstance(item, dict):
            continue

        true_label = item.get("true_label")
        model_output = item.get("model_output", {})
        predicted_label = model_output.get("predicted_label", -1)

        if true_label is None:
            continue

        if model_output.get("success") and predicted_label in [0, 1]:
            valid_samples += 1
            if predicted_label == true_label:
                correct_predictions += 1

    return valid_samples, correct_predictions

def should_skip_existing_result(existing_result: dict) -> bool:
    """Skip only when historical results were successful; failed results need to be re-executed."""
    if not isinstance(existing_result, dict):
        return False
    model_output = existing_result.get("model_output", {})
    return bool(model_output.get("success"))


def get_few_shot_examples(pool_data, pos_count, neg_count):
    """
    [New Logic]: Extract a specified number of positive and negative samples sequentially from a fixed candidate pool as Prompt examples.
    """
    pos_pool = []
    neg_pool = []
    
    # Force sorting to ensure that the order of extracted examples is absolutely consistent every time
    sorted_pool = sorted(pool_data.items(), key=lambda item: int(item[0]))
    
    for pid, data in sorted_pool:
        label = data.get("results", {}).get("Metastasis")
        if label == 1:
            pos_pool.append((pid, data))
        elif label == 0:
            neg_pool.append((pid, data))
            
    # Slice the required number of examples
    few_shot_pos = pos_pool[:pos_count]
    few_shot_neg = neg_pool[:neg_count]
    
    return few_shot_pos + few_shot_neg


def build_few_shot_prompt(few_shot_examples, target_metrics: dict) -> str:
    """
    Build a Few-Shot Prompt containing multiple examples.
    """
    base_instruction = (
        "Based on the following indicators, predict whether the patient will develop\n"
        "distant tumor metastasis during a follow-up period of at least one year.\n\n"
        f"[Patient Indicators]\n{metrics_str}\n\n"
        "[Output Constraint]\n"
        "Output strictly a single digit only. Do not include any explanation,\n"
        "punctuation, or other characters:\n"
        "0 = no metastasis\n"
        "1 = metastasis"
        "The following are reference cases from historical patients. Learn the data patterns they exhibit:\n"
    )
    
    examples_text = ""
    for i, (pid, data) in enumerate(few_shot_examples):
        metrics = data.get("metrics", {})
        label = data.get("results", {}).get("Metastasis")
        metrics_str = "\n".join([f"- {k}: {v}" for k, v in metrics.items()])
        
        examples_text += (
            f"=== Reference Example {i+1} ===\n"
            f"[Patient Indicators]\n{metrics_str}\n"
            f"[Ground-Truth Metastasis Status]\n{label}\n\n"
        )
        
    target_metrics_str = "\n".join([f"- {k}: {v}" for k, v in target_metrics.items()])
    target_text = (
        f"=== Now predict the following target patient ===\n"
        f"[Patient Indicators]\n{target_metrics_str}\n"
        f"[Prediction]\n"
    )
    
    return base_instruction + examples_text + target_text


def predict_metastasis(prompt_text: str) -> dict:
    """
    Call the Bailian API and return a complete dictionary containing all model output information.
    """
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {'role': 'system', 'content': 'You are a rigorous medical AI assistant, strictly complying with output format requirements.'},
                {'role': 'user', 'content': prompt_text}
            ],
            temperature=0.0,
        )
        
        message = completion.choices[0].message
        reasoning = getattr(message, 'reasoning_content', None)
        if not reasoning and hasattr(message, 'model_dump'):
             reasoning = message.model_dump().get('reasoning_content')
             
        result_text = message.content.strip()
        usage = completion.usage.model_dump() if hasattr(completion.usage, 'model_dump') else dict(completion.usage)
        
        try:
            predicted_label = int(result_text)
        except ValueError:
            predicted_label = -1
            
        return {
            "success": True,
            "predicted_label": predicted_label,
            "raw_content": result_text,
            "reasoning_content": reasoning,
            "usage": usage
        }
    
    except Exception as e:
        return {
            "success": False,
            "predicted_label": -1,
            "error_message": str(e)
        }


# ================= 2. Main Execution Flow =================
if __name__ == "__main__":
    start_time = time.time()
    
    # 1. Load candidate pool and get Few-Shot examples
    print(f"Loading examples from candidate pool ({POOL_FILE})...")
    with open(POOL_FILE, 'r', encoding='utf-8') as f:
        pool_data = json.load(f)
        
    few_shot_examples = get_few_shot_examples(pool_data, FEW_SHOT_POS_COUNT, FEW_SHOT_NEG_COUNT)
    
    # 2. Load the fixed test set
    print(f"Loading fixed test set ({TEST_FILE})...")
    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        test_samples = json.load(f)
    
    total_test_samples = len(test_samples)
    final_results = load_json_if_exists(OUTPUT_FILE)
    valid_samples, correct_predictions = summarize_results(final_results)

    if final_results:
        print(f"📌 Loaded {len(final_results)} historical results, subsequent processing will only handle new patient_ids.")

    print(f"Extracted {len(few_shot_examples)} Few-Shot examples as reference (Positive {FEW_SHOT_POS_COUNT}, Negative {FEW_SHOT_NEG_COUNT}).")
    print(f"Model: {MODEL_NAME}, Starting Few-Shot evaluation, a total of {total_test_samples} test records...\n")
    print("-" * 50)

    for patient_id, data in test_samples.items():
        existing_result = final_results.get(patient_id)
        if should_skip_existing_result(existing_result):
            print(f"Model: {MODEL_NAME}, Patient ID: {patient_id} | Existing successful result, skipping")
            continue
        if patient_id in final_results:
            print(f"Model: {MODEL_NAME}, Patient ID: {patient_id} | Historical result failed, re-executing")

        metrics = data.get("metrics", {})
        true_label = data.get("results", {}).get("Metastasis")
        
        if true_label is None:
            continue

        prompt = build_few_shot_prompt(few_shot_examples, metrics)
        
        result_dict = predict_metastasis(prompt)
        predicted_label = result_dict.get("predicted_label", -1)
        
        final_results[patient_id] = {
            "true_label": true_label,
            "model_output": result_dict
        }

        save_json(OUTPUT_FILE, final_results)
        
        if result_dict.get("success") and predicted_label in [0, 1]:
            valid_samples += 1
            is_correct = (predicted_label == true_label)
            if is_correct:
                correct_predictions += 1
            
            mark = "✅" if is_correct else "❌"
            print(f"Model: {MODEL_NAME}, Patient ID: {patient_id} | True Value: {true_label} | Predicted Value: {predicted_label} {mark}")
        else:
            print(f"Model: {MODEL_NAME}, Patient ID: {patient_id} | ⚠️ Prediction exception")

    print("-" * 50)
    
    # ================= 3. Save Results and Statistics =================
    save_json(OUTPUT_FILE, final_results)
    print(f"\n✅ Prediction results and detailed model outputs have been successfully saved to: {OUTPUT_FILE}")

    end_time = time.time()
    elapsed_time = end_time - start_time

    if valid_samples > 0:
        accuracy = correct_predictions / valid_samples
        print(f"Model: {MODEL_NAME}, [Evaluation Completed]")
        print(f"Model: {MODEL_NAME}, Total Test Samples: {valid_samples} (Successful responses)")
        print(f"Model: {MODEL_NAME}, Correct Predictions: {correct_predictions}")
        print(f"Model: {MODEL_NAME}, Few-Shot Accuracy: {accuracy:.2%}")
        
        summary_stats = {
            "experiment_type": "Few-Shot",
            "model_name": MODEL_NAME,
            "test_set_file": TEST_FILE,
            "few_shot_config": {
                "pool_file": POOL_FILE,
                "positive_examples": FEW_SHOT_POS_COUNT,
                "negative_examples": FEW_SHOT_NEG_COUNT,
                "example_ids": [pid for pid, _ in few_shot_examples]
            },
            "total_test_samples": total_test_samples,
            "valid_samples": valid_samples,
            "correct_predictions": correct_predictions,
            "accuracy": round(accuracy, 4),
            "elapsed_time_seconds": round(elapsed_time, 2)
        }
        
        with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
            json.dump(summary_stats, f, ensure_ascii=False, indent=4)
        print(f"Model: {MODEL_NAME}, ✅ Summary statistics information has been saved to: {SUMMARY_FILE}\n")
        
    else:
        print(f"Model: {MODEL_NAME}, No valid prediction results found, please check API configuration or network.")
        
    print(f"Model: {MODEL_NAME}, Total execution time: {elapsed_time:.2f} seconds")