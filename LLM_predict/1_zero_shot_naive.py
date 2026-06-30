"""
Purpose:
    Perform Zero-Shot predictions on a fixed test set:
    - Convert each patient's `metrics` into a natural language Prompt, call the model to predict whether distant metastasis occurs (0/1).
    - Save original model outputs and statistical summaries.

Optional configs:
    - Constants at the top of the script such as `MODEL_NAME`, `INPUT_DIR`, `OUTPUT_DIR`, etc.
    - Environment variable `DASHSCOPE_API_KEY` used for API authentication.

Input/Output paths:
    - Input: `LLM/data/input/test_set.json`
    - Output: `LLM/data/results/1_zero_shot_naive/{MODEL_NAME}/output_results.json`
    - Summary: `LLM/data/results/1_zero_shot_naive/{MODEL_NAME}/summary_metrics.json`
"""

import os
import json
import time
import argparse
from openai import OpenAI


# ================= 1. Initialize Configuration =================
# Please ensure that DASHSCOPE_API_KEY is set in your environment variables.
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def resolve_model_name(default_model_name: str) -> str:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--model-name", default=os.getenv("MODEL_NAME", default_model_name))
    args = parser.parse_args()
    return args.model_name

MODEL_NAME = resolve_model_name("kimi-k2.6")  # qwen-plus or qwen-max is recommended
INPUT_DIR = "LLM/data/input"                      # Directory storing input JSON files
OUTPUT_DIR = f"LLM/data/results/1_zero_shot_naive/{MODEL_NAME}"

# [Key Modification]: No longer reading original full data, but reading the split fixed test set instead.
DATA_FILE = f"{INPUT_DIR}/test_set.json" 

OUTPUT_FILE = f"{OUTPUT_DIR}/output_results.json" # Results storage path
SUMMARY_FILE = f"{OUTPUT_DIR}/summary_metrics.json" # Statistical metrics storage path
# =================================================

def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

make_dirs(OUTPUT_DIR)

def load_json_if_exists(file_path: str) -> dict:
    """Read existing JSON results; return an empty dictionary if it doesn't exist or is empty."""
    if not os.path.exists(file_path):
        return {}

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}

def save_json(file_path: str, data: dict) -> None:
    """Write dictionary to a JSON file."""
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
        else:
            print(f"⚠️ Invalid prediction result, unable to count accuracy: {model_output} {model_output.get('error_message', 'Unknown Error')}")

    return valid_samples, correct_predictions

def should_skip_existing_result(existing_result: dict) -> bool:
    """Skip only when the historical result was successful; failed results need to be re-executed."""
    if not isinstance(existing_result, dict):
        return False
    model_output = existing_result.get("model_output", {})
    return bool(model_output.get("success"))

def build_zero_shot_prompt(metrics: dict) -> str:
    """
    Convert the patient metrics dictionary into a natural language Prompt.
    """
    metrics_str = "\n".join([f"- {k}: {v}" for k, v in metrics.items()])
    
    prompt = (
        "Based on the following indicators, predict whether the patient will develop\n"
        "distant tumor metastasis during a follow-up period of at least one year.\n\n"
        f"[Patient Indicators]\n{metrics_str}\n\n"
        "[Output Constraint]\n"
        "Output strictly a single digit only. Do not include any explanation,\n"
        "punctuation, or other characters:\n"
        "0 = no metastasis\n"
        "1 = metastasis"
    )
    return prompt

def predict_metastasis(prompt_text: str) -> dict:
    """
    Call the Bailian API and return a complete dictionary containing all model output information (no longer returns just an int).
    """
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {'role': 'system', 'content': 'You are a rigorous medical AI assistant, strictly complying with output format requirements.'},
                {'role': 'user', 'content': prompt_text}
            ],
            temperature=0.0,  # Keep output stable
        )
        
        message = completion.choices[0].message
        
        # 1. Extract reasoning process
        reasoning = getattr(message, 'reasoning_content', None)
        if not reasoning and hasattr(message, 'model_dump'):
             reasoning = message.model_dump().get('reasoning_content')
             
        # 2. Extract final answer
        result_text = message.content.strip()
        
        # 3. Extract Token consumption statistics (convert to dict format for easy JSON saving)
        usage = completion.usage.model_dump() if hasattr(completion.usage, 'model_dump') else dict(completion.usage)
        
        # Try to convert to integer
        try:
            predicted_label = int(result_text)
        except ValueError:
            predicted_label = -1
            
        # Return all packed data
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
    
    print(f"Loading fixed test set data ({DATA_FILE})...")
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        patients_data = json.load(f)
    
    total_samples = len(patients_data)
    final_results = load_json_if_exists(OUTPUT_FILE)
    valid_samples, correct_predictions = summarize_results(final_results)

    if final_results:
        print(f"📌 Loaded {len(final_results)} historical results, only processing new patient_ids from now on.")

    print(f"Model: {MODEL_NAME}, starting Zero-Shot evaluation on {total_samples} data items...\n")
    print("-" * 50)

    for patient_id, data in patients_data.items():
        existing_result = final_results.get(patient_id)
        if should_skip_existing_result(existing_result):
            print(f"Model: {MODEL_NAME}, Patient ID: {patient_id} | Successful result already exists, skipping")
            continue
        if patient_id in final_results:
            print(f"Model: {MODEL_NAME}, Patient ID: {patient_id} | Historical result failed, re-executing")

        metrics = data.get("metrics", {})
        true_label = data.get("results", {}).get("Metastasis")
        
        if true_label is None:
            continue

        prompt = build_zero_shot_prompt(metrics)

        # Call API to get packed dictionary
        result_dict = predict_metastasis(prompt)
        predicted_label = result_dict.get("predicted_label", -1)
        
        # Combine true label and all information returned by the model for recording
        final_results[patient_id] = {
            "true_label": true_label,
            "model_output": result_dict  # This includes reasoning, usage, and everything else
        }

        save_json(OUTPUT_FILE, final_results)
        
        # Output minimalist logs to the terminal only, without printing reasoning details
        if result_dict.get("success") and predicted_label in [0, 1]:
            valid_samples += 1
            is_correct = (predicted_label == true_label)
            if is_correct:
                correct_predictions += 1
            
            mark = "✅" if is_correct else "❌"
            print(f"Model: {MODEL_NAME}, Patient ID: {patient_id} | True Value: {true_label} | Predicted Value: {predicted_label} {mark}")
        else:
            print(f"Model: {MODEL_NAME}, Patient ID: {patient_id} | ⚠️ Prediction anomaly")

    print("-" * 50)
    
    # ================= 3. Save Results and Statistics =================
    save_json(OUTPUT_FILE, final_results)
    print(f"\n✅ Prediction results and detailed model outputs have been successfully saved to: {OUTPUT_FILE}")

    end_time = time.time()
    elapsed_time = end_time - start_time

    if valid_samples > 0:
        accuracy = correct_predictions / valid_samples
        print(f"Model: {MODEL_NAME}, [Evaluation Completed]")
        print(f"Model: {MODEL_NAME}, Total Test Samples: {valid_samples} (Successful Responses)")
        print(f"Model: {MODEL_NAME}, Correct Predictions: {correct_predictions}")
        print(f"Model: {MODEL_NAME}, Zero-Shot Baseline Accuracy: {accuracy:.2%}")
        
        # Construct summary statistics dictionary and write to a new JSON file
        summary_stats = {
            "experiment_type": "Zero-Shot Baseline",
            "model_name": MODEL_NAME,
            "test_set_file": DATA_FILE,  # Record which test set is used
            "total_samples": total_samples,
            "valid_samples": valid_samples,
            "correct_predictions": correct_predictions,
            "accuracy": round(accuracy, 4), 
            "elapsed_time_seconds": round(elapsed_time, 2)
        }
        
        with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
            json.dump(summary_stats, f, ensure_ascii=False, indent=4)
        print(f"Model: {MODEL_NAME}, ✅ Summary metrics successfully saved to: {SUMMARY_FILE}\n")
        
    else:
        print(f"Model: {MODEL_NAME}, no valid prediction results. Please check your API configurations or network.")
        
    print(f"Model: {MODEL_NAME}, Total execution time: {elapsed_time:.2f} seconds")