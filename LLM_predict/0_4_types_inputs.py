"""
=============================================================================
[File Input/Output and Directory Structure Description]

1. Input Data Directory (INPUT_DIR): 
   Path: LLM/data/input_4_types/
   Description: Contains JSON test data for 4 processing levels:
         - test_set.json (Original full data)
         - test_set_selected.json (Retains 30 columns only)
         - test_set_rounded.json (Numerical values rounded to 2 decimal places)
         - test_set_selected_rounded.json (Selected columns + 2 decimal places)

2. Results Output Directory (OUTPUT_DIR): 
   Path: LLM/data/results/0_4_types_inputs/{MODEL_NAME}/
   Description: The script will automatically create this directory if it does not exist.

3. Generated Output Files (OUTPUT FILES):
   The code will iterate through the 4 types of inputs above and generate 2 exclusive 
   files for each experiment, with prefixes corresponding to the experiment name:
   - <Experiment_Name>_output_results.json  : Stores specific prediction results, true labels, 
                                              and complete model API responses for each patient 
                                              (including elapsed time, token usage, etc.).
   - <Experiment_Name>_summary_metrics.json : Stores macro statistical results for the scheme 
                                              (total test samples, successful predictions, 
                                              and final Accuracy).

   * Experiment name prefixes include: 1_Original, 2_Selected, 3_Rounded, 4_Selected_Rounded
=============================================================================
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
INPUT_DIR = "LLM/data/input_4_types"
OUTPUT_DIR = f"LLM/data/results/0_4_types_inputs/{MODEL_NAME}"

# Configure 4 different input data dictionaries
INPUT_FILES = {
    "1_Original": f"{INPUT_DIR}/test_set.json",
    "2_Selected": f"{INPUT_DIR}/test_set_selected.json",
    "3_Rounded": f"{INPUT_DIR}/test_set_rounded.json",
    # "4_Selected_Rounded": f"{INPUT_DIR}/test_set_selected_rounded.json" # Consistent with 1_zero_shot_navie.py
}
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
        "Based on the following metrics, please predict whether the patient will develop distant tumor metastasis in the future.\n\n"
        f"[Patient Metrics]\n{metrics_str}\n\n"
        "[Output Constraints]\n"
        "Please strictly output only a single digit, without any explanation, punctuation, or other characters:\n"
        "0 stands for No Metastasis\n"
        "1 stands for Metastasis"
    )
    return prompt

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
            temperature=0.0,  # Keep output stable
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
    global_start_time = time.time()
    
    # Used to record final horizontal comparison results
    experiment_summaries = {}

    # Iterate through 4 data configurations for automated testing
    for exp_name, file_path in INPUT_FILES.items():
        print("=" * 60)
        print(f"🚀 Starting test scheme: {exp_name}, Model: {MODEL_NAME}")
        print(f"📂 Reading file: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"❌ File does not exist, skipping this test: {file_path}")
            continue

        # Dynamically set exclusive output paths for each experiment
        output_file = f"{OUTPUT_DIR}/{exp_name}_output_results.json"
        summary_file = f"{OUTPUT_DIR}/{exp_name}_summary_metrics.json"
        final_results = load_json_if_exists(output_file)

        with open(file_path, 'r', encoding='utf-8') as f:
            patients_data = json.load(f)
        
        total_samples = len(patients_data)

        # Skip already calculated patient_ids, keeping historical results and appending new ones
        valid_samples, correct_predictions = summarize_results(final_results)

        if final_results:
            print(f"📌 Loaded {len(final_results)} historical results, only processing new patient_ids from now on.")

        for patient_id, data in patients_data.items():
            existing_result = final_results.get(patient_id)
            if should_skip_existing_result(existing_result):
                print(f"  Model: {MODEL_NAME}, [ID: {patient_id}] Successful result already exists, skipping")
                continue
            if patient_id in final_results:
                print(f"  Model: {MODEL_NAME}, [ID: {patient_id}] Historical result failed, re-executing")

            # Compatibility: original nested dictionary structure vs flattened dictionary structure
            if "results" in data and isinstance(data["results"], dict):
                true_label = data["results"].get("Metastasis")
                metrics = data.get("metrics", {})
            else:
                true_label = data.get("Metastasis")
                metrics = {k: v for k, v in data.items() if k != "Metastasis"}
            
            if true_label is None:
                continue

            prompt = build_zero_shot_prompt(metrics)

            result_dict = predict_metastasis(prompt)
            predicted_label = result_dict.get("predicted_label", -1)
            
            final_results[patient_id] = {
                "true_label": true_label,
                "model_output": result_dict 
            }

            save_json(output_file, final_results)
            
            if result_dict.get("success") and predicted_label in [0, 1]:
                valid_samples += 1
                is_correct = (predicted_label == true_label)
                if is_correct:
                    correct_predictions += 1
                
                mark = "✅" if is_correct else "❌"
                print(f"  Model: {MODEL_NAME}, [ID: {patient_id}] True: {true_label} | Pred: {predicted_label} {mark}")
            else:
                print(f"  Model: {MODEL_NAME}, [ID: {patient_id}] ⚠️ Prediction anomaly")

        # ---------------- Save single experiment results ----------------
        save_json(output_file, final_results)

        if valid_samples > 0:
            accuracy = correct_predictions / valid_samples
            
            summary_stats = {
                "experiment_type": exp_name,
                "model_name": MODEL_NAME,
                "test_set_file": file_path,
                "total_samples": total_samples,
                "valid_samples": valid_samples,
                "correct_predictions": correct_predictions,
                "accuracy": round(accuracy, 4)
            }
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_stats, f, ensure_ascii=False, indent=4)
                
            print(f"\n📊 Model: {MODEL_NAME}, {exp_name} test completed! Accuracy: {accuracy:.2%}")
            # Record into global summary dictionary for table printing
            experiment_summaries[exp_name] = accuracy
        else:
            print(f"\n⚠️ Model: {MODEL_NAME}, {exp_name} has no valid prediction results.")

    # ================= 3. Print Final Horizontal Comparison =================
    global_elapsed = time.time() - global_start_time
    print("\n" + "=" * 60)
    print(f"🏆 Model: {MODEL_NAME}, [Experiment Summary Report]")
    print("-" * 60)
    print(f"{'Input Data Type':<25} | {'Accuracy'}")
    print("-" * 60)
    for exp, acc in experiment_summaries.items():
        print(f"{exp:<25} | {acc:.2%}")
    print("-" * 60)
    print(f"⏱️ Model: {MODEL_NAME}, Total execution time: {global_elapsed:.2f} seconds")
    print("=" * 60)