"""
Purpose:
    Perform Zero-Shot predictions on a fixed test set using a knowledge-augmented Prompt:
    - Embed a detailed clinical knowledge base and conflict adjudication rules within the Prompt to enhance judgment consistency.
    - Save original model outputs and statistical summaries.

Optional configs:
    - Constants at the top of the script such as `MODEL_NAME`, `INPUT_DIR`, `OUTPUT_DIR`, etc.
    - Environment variable `DASHSCOPE_API_KEY` used for API authentication.

Input/Output paths:
    - Input: `LLM/data/input/test_set.json`
    - Output: `LLM/data/results/3_prompt_aug/{MODEL_NAME}/output_results.json`
    - Summary: `LLM/data/results/3_prompt_aug/{MODEL_NAME}/summary_metrics.json`
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
OUTPUT_DIR = f"LLM/data/results/3_prompt_aug/{MODEL_NAME}"

DATA_FILE = f"{INPUT_DIR}/test_set.json" 

OUTPUT_FILE = f"{OUTPUT_DIR}/output_results.json" 
SUMMARY_FILE = f"{OUTPUT_DIR}/summary_metrics.json" 
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

def build_augmented_prompt(metrics: dict) -> str:
    """
    Construct a knowledge-augmented Prompt containing clinical context and logic adjudication guidelines.
    """
    metrics_str = "\n".join([f"- {k}: {v}" for k, v in metrics.items()])

    prompt = (
        "You are an extremely rigorous algorithmic system for predicting metachronous\ndistant metastasis (MDM) in patients with colorectal cancer (CRC) after\ncurative-intent resection, over a follow-up period of at least one year.\nYour task is to predict whether the patient will develop distant metastasis\nduring follow-up, based on the multiple medical indicators measured at\nbaseline, by strictly comparing them against the weights and directions\nspecified in the knowledge base below.\n\n"

        "[Comprehensive Feature Interpretation Knowledge Base]\n"
        "Examine each indicator across the following five domains. Note: (↑) indicates\nthat an elevated value or a positive result increases metastatic risk;\n(↓) indicates that a decreased value or a loss/absence increases metastatic\nrisk (i.e., higher values are protective).\n\n"
        "Domain 1: Local Pathology and Anatomical Features (baseline local tumor burden)\n"
        "- Carcinoma nodule: High-risk factor. Presence (1) or a higher count sharply\n  increases metastatic risk (↑).\n"
        "- PLN (number of positive lymph nodes): >0 indicates regional nodal metastasis;\n  higher values indicate higher metastatic probability (↑).\n"
        "- Vascular invasion: 1 (Yes) is a risk factor, indicating that a hematogenous\n  dissemination route has been opened (↑).\n"
        "- Perineural invasion: 1 (Yes) is a risk factor, indicating strong local\n  infiltration (↑).\n"
        "- Differentiation grade: G3 (poorly differentiated) increases metastatic\n  probability (↑); G1 (well differentiated) carries low metastatic risk.\n"
        "- T stage & N stage: T3-T4 or N2 indicates deep invasion and high risk (↑);\n  T1-T2 or N0 indicates shallow invasion and low risk.\n"
        "- TNLE (total number of lymph nodes examined): Auxiliary indicator. Limited\n  predictive value; an excessively low count may underestimate nodal stage (↓).\n"
        "- Tumor size: Limited predictive value; larger size indicates heavier tumor\n  burden and may promote metastasis (↑).\n"
        "- Colonic obstruction: Limited predictive value; 1 (Yes) suggests advanced\n  compression and possible higher metastatic tendency (↑).\n\n"
        "Domain 2: Immune Microenvironment and Immune Cells (systemic defense barrier;\ncore weight)\n"
        "- Treg cells (%): The most central immunosuppressive high-risk indicator.\n  Higher values indicate more severe immune evasion and sharply increased\n  metastatic risk (↑).\n"
        "- CD4+ count, CD3+ count, NK cells (%): Core immune defenses (protective\n  factors). Abundant values indicate strong antitumor immunity; markedly\n  decreased values indicate immune exhaustion and increased metastatic risk (↓).\n"
        "- CD19+ B cells (%) & CD19+ count: Humoral-immunity auxiliary indicators.\n  Decreased values are generally accompanied by higher metastatic risk (↓).\n\n"
        "Domain 3: Tumor Markers and Cell Metabolism (systemic radar for micrometastasis;\ncore weight)\n"
        "- CEA: The most sensitive marker; abnormal elevation strongly suggests the\n  presence of micrometastasis (↑).\n"
        "- CA242, CA19-9: Gastrointestinal tumor markers; abnormal elevation suggests\n  metastatic risk (↑).\n"
        "- LDH (lactate dehydrogenase): Marker of metabolic abnormality; >171.79 U/L\n  indicates a markedly increased metastatic risk (↑).\n"
        "- Total protein: Nutritional and consumption indicator; <60 g/L\n  (hypoproteinemia) suggests cachexia and increased metastatic risk (↓).\n\n"
        "Domain 4: Systemic Inflammation and Cytokines (pro-metastatic microenvironment)\n"
        "- NLR (neutrophil-to-lymphocyte ratio): Systemic inflammation indicator;\n  elevation suggests possibly increased metastatic risk (↑).\n"
        "- mGPS: Combined inflammation–nutrition score; >0 indicates systemic\n  inflammation and possibly increased metastatic risk (↑).\n"
        "- IL-8: Pro-inflammatory cytokine; high expression promotes metastasis (↑).\n\n"
        "Domain 5: Gene Mutations and Family History (biological potential)\n"
        "- KRAS mutant, BRAF mutant: 1 (Yes) indicates a mutant type, reflecting strong\n  tumor aggressiveness and increased metastatic risk (↑).\n"
        "- PMS2, MSH6: Mismatch repair proteins. A value of 0 (Negative/loss) indicates\n  dMMR/MSI-H, a protective factor predicting lower metastatic risk (↓).\n\n"

        "[Clinical Conflict Arbitration Rules (Critically Important)]\n"
        "In real-world practice, indicators are often contradictory. Perform\nadversarial reasoning strictly according to the following priority:\n"
        "1. Even if Domain 1 (pathological stage, e.g., N2, vascular/perineural invasion)\n   and Domain 5 (KRAS mutation) present high-risk factors,\n"
        "2. if the patient's Domain 2 (immune defenses, e.g., very low Treg and abundant\n   CD3+/CD4+/NK cells) and Domain 3 (tumor markers such as CEA, LDH) are\n   completely normal or highly protective,\n"
        "3. this indicates that the patient's robust systemic immunity has cleared the\n   circulating micrometastatic tumor cells. In this case, \"the protective\n   efficacy of immunity and metabolism\" should override \"the danger of local\n   pathology,\" and the integrated judgment should be no metastasis.\n\n"

        "[Execution Instruction]\n"
        "Strictly compare the specific values of the patient below, and internally\nperform an additive–subtractive adversarial calculation between \"risk factors\"\nand \"protective factors.\" Pay particular attention to applying the above\n[Clinical Conflict Arbitration Rules] when handling contradictory indicators.\n\n"

        "[Actual Patient Indicators]\n"
        f"{metrics_str}\n\n"

        "[Output Format Constraint]\n"
        "Output strictly a single digit only. Absolutely do not include any explanation,\nanalytical process, punctuation, or other characters:\n"
        "0 = no metastasis (No MDM)\n"
        "1 = metastasis (MDM)"
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
    start_time = time.time()
    
    print(f"Loading fixed test set data ({DATA_FILE})...")
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        patients_data = json.load(f)
    
    total_samples = len(patients_data)
    final_results = load_json_if_exists(OUTPUT_FILE)
    valid_samples, correct_predictions = summarize_results(final_results)

    if final_results:
        print(f"📌 Loaded {len(final_results)} historical results, only processing new patient_ids from now on.")

    print(f"Model: {MODEL_NAME}, starting Knowledge-Augmented evaluation on {total_samples} data items...\n")
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

        prompt = build_augmented_prompt(metrics)

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
            print(f"Model: {MODEL_NAME}, Patient ID: {patient_id} | ⚠️ Prediction anomaly")

    print("-" * 50)
    
    # ================= 3. Save Results and Statistics =================
    save_json(OUTPUT_FILE, final_results)
    print(f"\\n✅ Prediction results and detailed model outputs have been successfully saved to: {OUTPUT_FILE}")

    end_time = time.time()
    elapsed_time = end_time - start_time

    if valid_samples > 0:
        accuracy = correct_predictions / valid_samples
        print(f"Model: {MODEL_NAME}, [Evaluation Completed]")
        print(f"Model: {MODEL_NAME}, Total Test Samples: {valid_samples} (Successful Responses)")
        print(f"Model: {MODEL_NAME}, Correct Predictions: {correct_predictions}")
        print(f"Model: {MODEL_NAME}, Knowledge-Augmented Accuracy: {accuracy:.2%}")
        
        summary_stats = {
            "experiment_type": "Knowledge-Augmented Zero-Shot",
            "model_name": MODEL_NAME,
            "test_set_file": DATA_FILE,
            "total_samples": total_samples,
            "valid_samples": valid_samples,
            "correct_predictions": correct_predictions,
            "accuracy": round(accuracy, 4), 
            "elapsed_time_seconds": round(elapsed_time, 2)
        }
        
        with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
            json.dump(summary_stats, f, ensure_ascii=False, indent=4)
        print(f"Model: {MODEL_NAME}, ✅ Summary metrics successfully saved to: {SUMMARY_FILE}\\n")
        
    else:
        print(f"Model: {MODEL_NAME}, no valid prediction results. Please check your API configurations or network.")
        
    print(f"Model: {MODEL_NAME}, Total execution time: {elapsed_time:.2f} seconds")