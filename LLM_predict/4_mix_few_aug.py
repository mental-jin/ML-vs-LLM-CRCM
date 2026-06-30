"""
Purpose:
    Hybrid strategy combining knowledge augmentation and Few-Shot examples:
    - Contains both a detailed knowledge base and examples extracted from a candidate pool in the Prompt.
    - Performs sample-by-sample prediction on a fixed test set, saving outputs and statistics.

Optional configs:
    - `MODEL_NAME`, `INPUT_DIR`, `OUTPUT_DIR`, `FEW_SHOT_POS_COUNT`, `FEW_SHOT_NEG_COUNT`, etc.
    - Environment variable `DASHSCOPE_API_KEY` used for API authentication.

Input/Output paths:
    - Input: `LLM/data/input/few_shot_pool.json`, `LLM/data/input/test_set.json`
    - Output: `LLM/data/results/4_mix_few_aug/{MODEL_NAME}/output_{p}p_{n}n.json`
    - Stats: `LLM/data/results/4_mix_few_aug/{MODEL_NAME}/summary_{p}p_{n}n.json`
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

MODEL_NAME = resolve_model_name("kimi-k2.6")
INPUT_DIR = "LLM/data/input"
OUTPUT_DIR = f"LLM/data/results/4_mix_few_aug/{MODEL_NAME}"  # Modified output directory to differentiate experiments

# [Key Modification]: No longer reading full data, instead separately reading "candidate pool" and "test set".
POOL_FILE = f"{INPUT_DIR}/few_shot_pool.json"  # Candidate pool providing examples
TEST_FILE = f"{INPUT_DIR}/test_set.json"       # Fixed test set

# --- Few-Shot Core Configuration ---
FEW_SHOT_POS_COUNT = 2  # Number of positive samples extracted from pool (Metastatic, label is 1)
FEW_SHOT_NEG_COUNT = 2  # Number of negative samples extracted from pool (Non-metastatic, label is 0)
# ------------------------------------

# To prevent experimental results with different ratios from overwriting each other, 
# it is recommended to add the number of examples to the output filename.
OUTPUT_FILE = f"{OUTPUT_DIR}/output_{FEW_SHOT_POS_COUNT}p_{FEW_SHOT_NEG_COUNT}n.json"
SUMMARY_FILE = f"{OUTPUT_DIR}/summary_{FEW_SHOT_POS_COUNT}p_{FEW_SHOT_NEG_COUNT}n.json"

def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

make_dirs(OUTPUT_DIR)
# =================================================

def load_json_if_exists(file_path: str) -> dict:
    """Reads existing JSON results; returns an empty dictionary if file does not exist or is empty."""
    if not os.path.exists(file_path):
        return {}

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}

def save_json(file_path: str, data: dict) -> None:
    """Writes a dictionary into a JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def summarize_results(results: dict) -> tuple[int, int]:
    """Counts the number of valid samples and correct predictions."""
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
    """Skips only if historical result was successful; failed results need to be re-executed."""
    if not isinstance(existing_result, dict):
        return False
    model_output = existing_result.get("model_output", {})
    return bool(model_output.get("success"))


def get_few_shot_examples(pool_data, pos_count, neg_count):
    """
    [New Logic]: Extracts the specified number of positive and negative samples 
    sequentially from the fixed candidate pool as Prompt examples.
    """
    pos_pool = []
    neg_pool = []
    
    # Enforce sorting to ensure that the order of extracted examples is absolutely consistent every time.
    sorted_pool = sorted(pool_data.items(), key=lambda item: int(item[0]))
    
    for pid, data in sorted_pool:
        label = data.get("results", {}).get("Metastasis")
        if label == 1:
            pos_pool.append((pid, data))
        elif label == 0:
            neg_pool.append((pid, data))
            
    # Intercept the required number of examples
    few_shot_pos = pos_pool[:pos_count]
    few_shot_neg = neg_pool[:neg_count]
    
    return few_shot_pos + few_shot_neg


def build_few_shot_prompt(few_shot_examples, target_metrics: dict) -> str:
    """
    Builds a Few-Shot Prompt containing multiple examples.
    """
    base_instruction = (
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
        "The following are reference cases from historical patients. Learn the data patterns they exhibit:\n"
    )
    
    examples_text = ""
    for i, (pid, data) in enumerate(few_shot_examples):
        metrics = data.get("metrics", {})
        label = data.get("results", {}).get("Metastasis")
        metrics_str = "\n".join([f"- {k}: {v}" for k, v in metrics.items()])
        
        examples_text += (
            f"=== Reference Example {i+1} ===\n"
            f"[Patient Metrics]\n{metrics_str}\n"
            f"[Ground-Truth Metastasis Status]\n{label}\n\n"
        )
        
    target_metrics_str = "\n".join([f"- {k}: {v}" for k, v in target_metrics.items()])
    target_text = (
        f"=== Now predict the following target patient ===\n"
        "[Execution Instruction]\n"
        "Strictly compare the specific values of the target patient below, integrate the historical examples, and internally perform an additive–subtractive adversarial calculation between \"risk factors\" and \"protective factors.\" Pay particular attention to applying the above [Clinical Conflict Arbitration Rules] when handling contradictory indicators.\n\n"
        
        f"[Patient Indicators]\n{target_metrics_str}\n"
        f"[Prediction]\n"
    )
    
    return base_instruction + examples_text + target_text


def predict_metastasis(prompt_text: str) -> dict:
    """
    Calls the Bailian API and returns a complete dictionary containing all model output info.
    """
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {'role': 'system', 'content': 'You are a rigorous medical AI assistant, strictly adhering to output format requirements.'},
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
    
    # 2. Load fixed test set
    print(f"Loading fixed test set ({TEST_FILE})...")
    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        test_samples = json.load(f)
    
    total_test_samples = len(test_samples)
    final_results = load_json_if_exists(OUTPUT_FILE)
    valid_samples, correct_predictions = summarize_results(final_results)

    if final_results:
        print(f"📌 Loaded {len(final_results)} historical records. Only new patient_ids will be processed subsequently.")

    print(f"Extracted {len(few_shot_examples)} Few-Shot examples for reference (Positive: {FEW_SHOT_POS_COUNT}, Negative: {FEW_SHOT_NEG_COUNT}).")
    print(f"Model: {MODEL_NAME}, starting Few-Shot & Prompt Aug evaluation, total test samples: {total_test_samples}...\n")
    print("-" * 50)

    for patient_id, data in test_samples.items():
        existing_result = final_results.get(patient_id)
        if should_skip_existing_result(existing_result):
            print(f"Model: {MODEL_NAME}, Patient ID: {patient_id} | Successful result exists, skipping")
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
            print(f"Model: {MODEL_NAME}, Patient ID: {patient_id} | True Label: {true_label} | Predicted Label: {predicted_label} {mark}")
        else:
            print(f"Model: {MODEL_NAME}, Patient ID: {patient_id} | ⚠️ Prediction anomaly")

    print("-" * 50)
    
    # ================= 3. Save Results and Statistics =================
    save_json(OUTPUT_FILE, final_results)
    print(f"\n✅ Prediction results and detailed model outputs completely saved to: {OUTPUT_FILE}")

    end_time = time.time()
    elapsed_time = end_time - start_time

    if valid_samples > 0:
        accuracy = correct_predictions / valid_samples
        print(f"【Evaluation Completed】")
        print(f"Model: {MODEL_NAME}, Total Test Samples: {valid_samples} (Successful responses)")
        print(f"Model: {MODEL_NAME}, Correct Predictions: {correct_predictions}")
        print(f"Model: {MODEL_NAME}, Few-Shot & Prompt Aug Accuracy: {accuracy:.2%}")
        
        summary_stats = {
            "experiment_type": "Few-Shot&Prompt Aug",
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
        print(f"Model: {MODEL_NAME}, ✅ Summary statistical info saved to: {SUMMARY_FILE}\n")
        
    else:
        print(f"Model: {MODEL_NAME}, no valid prediction results. Please check API config or network.")
        
    print(f"Total elapsed time: {elapsed_time:.2f} seconds")