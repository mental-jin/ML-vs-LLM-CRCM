"""
说明 (Purpose):
    使用固定候选池和固定测试集执行 Few-Shot 评估：
    - 从 `few_shot_pool.json` 中提取若干正/负示例拼入 Prompt，
    - 对固定测试集逐样本预测并保存模型输出和统计汇总。

可配置项 (Optional configs):
    - `MODEL_NAME`, `INPUT_DIR`, `OUTPUT_DIR`, `FEW_SHOT_POS_COUNT`, `FEW_SHOT_NEG_COUNT` 等
    - 环境变量 `DASHSCOPE_API_KEY` 用于 API 鉴权

输入/输出路径 (Input/Output paths):
    - 输入: `LLM/data/input/few_shot_pool.json`, `LLM/data/input/test_set.json`
    - 输出: `LLM/data/results/2_few_shot/{MODEL_NAME}/output_{p}p_{n}n.json`
    - 统计: `LLM/data/results/2_few_shot/{MODEL_NAME}/summary_{p}p_{n}n.json`
"""

import os
import json
import time
import argparse
from openai import OpenAI

# ================= 1. 初始化配置 =================
# 请确保你的环境变量中已设置 DASHSCOPE_API_KEY
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
OUTPUT_DIR = f"LLM/data/results/2_few_shot/{MODEL_NAME}"  # 修改了输出目录以区分实验

# 【关键修改】：不再读取全量数据，而是分别读取“候选池”和“测试集”
POOL_FILE = f"{INPUT_DIR}/few_shot_pool.json"  # 提供示例的候选池
TEST_FILE = f"{INPUT_DIR}/test_set.json"       # 固定的测试集

# --- Few-Shot 核心配置 ---
FEW_SHOT_POS_COUNT = 2  # 从候选池中提取的正样本数量（已转移，标签为1）
FEW_SHOT_NEG_COUNT = 2  # 从候选池中提取的负样本数量（未转移，标签为0）
# -------------------------

# 为了防止不同比例的实验结果互相覆盖，建议将输出文件名加上示例数量
OUTPUT_FILE = f"{OUTPUT_DIR}/output_{FEW_SHOT_POS_COUNT}p_{FEW_SHOT_NEG_COUNT}n.json"
SUMMARY_FILE = f"{OUTPUT_DIR}/summary_{FEW_SHOT_POS_COUNT}p_{FEW_SHOT_NEG_COUNT}n.json"

def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

make_dirs(OUTPUT_DIR)
# =================================================

def load_json_if_exists(file_path: str) -> dict:
    """读取已有 JSON 结果；不存在或为空时返回空字典。"""
    if not os.path.exists(file_path):
        return {}

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}

def save_json(file_path: str, data: dict) -> None:
    """将字典写入 JSON 文件。"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def summarize_results(results: dict) -> tuple[int, int]:
    """统计有效样本数和正确样本数。"""
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
    """只有历史结果已成功时才跳过；失败结果需要重新执行。"""
    if not isinstance(existing_result, dict):
        return False
    model_output = existing_result.get("model_output", {})
    return bool(model_output.get("success"))


def get_few_shot_examples(pool_data, pos_count, neg_count):
    """
    【全新逻辑】：从固定的候选池中，按顺序提取指定数量的正负样本作为 Prompt 示例。
    """
    pos_pool = []
    neg_pool = []
    
    # 强制排序，确保每次取出来的示例顺序绝对一致
    sorted_pool = sorted(pool_data.items(), key=lambda item: int(item[0]))
    
    for pid, data in sorted_pool:
        label = data.get("results", {}).get("Metastasis")
        if label == 1:
            pos_pool.append((pid, data))
        elif label == 0:
            neg_pool.append((pid, data))
            
    # 截取所需数量的示例
    few_shot_pos = pos_pool[:pos_count]
    few_shot_neg = neg_pool[:neg_count]
    
    return few_shot_pos + few_shot_neg


def build_few_shot_prompt(few_shot_examples, target_metrics: dict) -> str:
    """
    构建包含多个示例的 Few-Shot Prompt
    """
    base_instruction = (
        "请根据以下指标，预测患者未来是否会发生肿瘤远处转移。\n\n"
        "【输出限制】\n"
        "请严格只输出一个数字，不要包含任何解释、标点或其他字符：\n"
        "0 代表 未转移\n"
        "1 代表 已转移\n\n"
        "以下是一些历史患者的参考示例，请学习其中的数据模式：\n"
    )
    
    examples_text = ""
    for i, (pid, data) in enumerate(few_shot_examples):
        metrics = data.get("metrics", {})
        label = data.get("results", {}).get("Metastasis")
        metrics_str = "\n".join([f"- {k}: {v}" for k, v in metrics.items()])
        
        examples_text += (
            f"=== 参考示例 {i+1} ===\n"
            f"【患者指标】\n{metrics_str}\n"
            f"【真实转移情况】\n{label}\n\n"
        )
        
    target_metrics_str = "\n".join([f"- {k}: {v}" for k, v in target_metrics.items()])
    target_text = (
        f"=== 请预测以下目标患者 ===\n"
        f"【患者指标】\n{target_metrics_str}\n"
        f"【预测结果】\n"
    )
    
    return base_instruction + examples_text + target_text


def predict_metastasis(prompt_text: str) -> dict:
    """
    调用百炼 API，返回包含所有模型输出信息的完整字典
    """
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {'role': 'system', 'content': '你是一个严谨的医疗AI助手，严格遵守输出格式要求。'},
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


# ================= 2. 主执行流程 =================
if __name__ == "__main__":
    start_time = time.time()
    
    # 1. 加载候选池并获取 Few-Shot 示例
    print(f"从候选池加载示例 ({POOL_FILE})...")
    with open(POOL_FILE, 'r', encoding='utf-8') as f:
        pool_data = json.load(f)
        
    few_shot_examples = get_few_shot_examples(pool_data, FEW_SHOT_POS_COUNT, FEW_SHOT_NEG_COUNT)
    
    # 2. 加载固定的测试集
    print(f"加载固定测试集 ({TEST_FILE})...")
    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        test_samples = json.load(f)
    
    total_test_samples = len(test_samples)
    final_results = load_json_if_exists(OUTPUT_FILE)
    valid_samples, correct_predictions = summarize_results(final_results)

    if final_results:
        print(f"📌 已加载历史结果 {len(final_results)} 条，后续只处理新增 patient_id。")

    print(f"提取了 {len(few_shot_examples)} 个 Few-Shot 示例作为参考（正样本 {FEW_SHOT_POS_COUNT}，负样本 {FEW_SHOT_NEG_COUNT}）。")
    print(f"Model: {MODEL_NAME}, 开始 Few-Shot 评估，共 {total_test_samples} 条测试数据...\n")
    print("-" * 50)

    for patient_id, data in test_samples.items():
        existing_result = final_results.get(patient_id)
        if should_skip_existing_result(existing_result):
            print(f"Model: {MODEL_NAME}, 患者 ID: {patient_id} | 已存在成功结果，跳过")
            continue
        if patient_id in final_results:
            print(f"Model: {MODEL_NAME}, 患者 ID: {patient_id} | 历史结果失败，重新执行")

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
            print(f"Model: {MODEL_NAME}, 患者 ID: {patient_id} | 真实值: {true_label} | 预测值: {predicted_label} {mark}")
        else:
            print(f"Model: {MODEL_NAME}, 患者 ID: {patient_id} | ⚠️ 预测异常")

    print("-" * 50)
    
    # ================= 3. 保存结果与统计 =================
    save_json(OUTPUT_FILE, final_results)
    print(f"\n✅ 预测结果及详细模型输出已完整保存至: {OUTPUT_FILE}")

    end_time = time.time()
    elapsed_time = end_time - start_time

    if valid_samples > 0:
        accuracy = correct_predictions / valid_samples
        print(f"Model: {MODEL_NAME}, 【评估完成】")
        print(f"Model: {MODEL_NAME}, 总测试样本: {valid_samples} (成功响应)")
        print(f"Model: {MODEL_NAME}, 正确预测数: {correct_predictions}")
        print(f"Model: {MODEL_NAME}, Few-Shot 准确率: {accuracy:.2%}")
        
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
        print(f"Model: {MODEL_NAME}, ✅ 统计汇总信息已保存至: {SUMMARY_FILE}\n")
        
    else:
        print(f"Model: {MODEL_NAME}, 没有有效的预测结果，请检查 API 配置或网络。")
        
    print(f"Model: {MODEL_NAME}, 总执行耗时: {elapsed_time:.2f} 秒")