"""
=============================================================================
【文件输入/输出与目录结构说明】

1. 输入数据目录 (INPUT_DIR): 
   路径: LLM/data/input_4_types/
   说明: 存放 4 种处理级别的 JSON 测试数据：
         - test_set.json (原始全量)
         - test_set_selected.json (仅保留30列)
         - test_set_rounded.json (数值保留2位小数)
         - test_set_selected_rounded.json (选列 + 2位小数)

2. 结果输出目录 (OUTPUT_DIR): 
   路径: LLM/data/results/0_4_types_inputs/{MODEL_NAME}/
   说明: 脚本会自动创建此目录（如不存在）。

3. 生成的输出文件 (OUTPUT FILES):
   代码会遍历上述 4 种输入，为每一种实验生成 2 个专属文件，前缀与实验名对应：
   - <实验名>_output_results.json  : 存放针对每个患者的具体预测结果、真实标签以及完整的模型API响应（包含耗时、Token使用量等）。
   - <实验名>_summary_metrics.json : 存放该方案的宏观统计结果（总测试样本数、成功预测数、最终准确率 Accuracy）。

   * 实验名前缀包括: 1_Original, 2_Selected, 3_Rounded, 4_Selected_Rounded
=============================================================================
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

MODEL_NAME = resolve_model_name("kimi-k2.6")  # 建议使用 qwen-plus 或 qwen-max
INPUT_DIR = "LLM/data/input_4_types"
OUTPUT_DIR = f"LLM/data/results/0_4_types_inputs/{MODEL_NAME}"

# 配置 4 种不同的输入数据字典
INPUT_FILES = {
    "1_Original": f"{INPUT_DIR}/test_set.json",
    "2_Selected": f"{INPUT_DIR}/test_set_selected.json",
    "3_Rounded": f"{INPUT_DIR}/test_set_rounded.json",
    # "4_Selected_Rounded": f"{INPUT_DIR}/test_set_selected_rounded.json" # 和 1_zero_shot_navie.py一致
}
# =================================================

def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

make_dirs(OUTPUT_DIR)

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

def build_zero_shot_prompt(metrics: dict) -> str:
    """
    将患者特征字典转化为自然语言 Prompt
    """
    metrics_str = "\n".join([f"- {k}: {v}" for k, v in metrics.items()])
    
    prompt = (
        "请根据以下指标，预测患者未来是否会发生肿瘤远处转移。\n\n"
        f"【患者指标】\n{metrics_str}\n\n"
        "【输出限制】\n"
        "请严格只输出一个数字，不要包含任何解释、标点或其他字符：\n"
        "0 代表 未转移\n"
        "1 代表 已转移"
    )
    return prompt

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
            temperature=0.0,  # 保持输出稳定
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
    global_start_time = time.time()
    
    # 用于记录最终横向对比结果
    experiment_summaries = {}

    # 遍历 4 种数据配置进行自动化测试
    for exp_name, file_path in INPUT_FILES.items():
        print("=" * 60)
        print(f"🚀 开始测试方案: {exp_name}, Model: {MODEL_NAME}")
        print(f"📂 读取文件: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"❌ 文件不存在，跳过此测试: {file_path}")
            continue

        # 动态设置每个实验的专属输出路径
        output_file = f"{OUTPUT_DIR}/{exp_name}_output_results.json"
        summary_file = f"{OUTPUT_DIR}/{exp_name}_summary_metrics.json"
        final_results = load_json_if_exists(output_file)

        with open(file_path, 'r', encoding='utf-8') as f:
            patients_data = json.load(f)
        
        total_samples = len(patients_data)

        # 已经计算过的 patient_id 直接跳过，保留历史结果并继续追加新结果
        valid_samples, correct_predictions = summarize_results(final_results)

        if final_results:
            print(f"📌 已加载历史结果 {len(final_results)} 条，后续只处理新增 patient_id。")

        for patient_id, data in patients_data.items():
            existing_result = final_results.get(patient_id)
            if should_skip_existing_result(existing_result):
                print(f"  Model: {MODEL_NAME}, [ID: {patient_id}] 已存在成功结果，跳过")
                continue
            if patient_id in final_results:
                print(f"  Model: {MODEL_NAME}, [ID: {patient_id}] 历史结果失败，重新执行")

            # 兼容：原始的嵌套字典结构 vs 扁平化字典结构
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
                print(f"  Model: {MODEL_NAME}, [ID: {patient_id}] 真实: {true_label} | 预测: {predicted_label} {mark}")
            else:
                print(f"  Model: {MODEL_NAME}, [ID: {patient_id}] ⚠️ 预测异常")

        # ---------------- 保存单次实验结果 ----------------
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
                
            print(f"\n📊 Model: {MODEL_NAME}, {exp_name} 测试完成! 准确率: {accuracy:.2%}")
            # 记录到全局汇总字典中以备表格打印
            experiment_summaries[exp_name] = accuracy
        else:
            print(f"\n⚠️ Model: {MODEL_NAME}, {exp_name} 没有有效的预测结果。")

    # ================= 3. 打印最终横向对比 =================
    global_elapsed = time.time() - global_start_time
    print("\n" + "=" * 60)
    print(f"🏆 Model: {MODEL_NAME}, 【实验汇总报告】")
    print("-" * 60)
    print(f"{'输入数据类型':<25} | {'准确率 (Accuracy)'}")
    print("-" * 60)
    for exp, acc in experiment_summaries.items():
        print(f"{exp:<25} | {acc:.2%}")
    print("-" * 60)
    print(f"⏱️ Model: {MODEL_NAME}, 总执行耗时: {global_elapsed:.2f} 秒")
    print("=" * 60)