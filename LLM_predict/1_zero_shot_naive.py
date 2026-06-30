"""
说明 (Purpose):
    对固定测试集执行 Zero-Shot 预测：
    - 将每个患者的 `metrics` 转换为自然语言 Prompt，调用模型预测是否发生远处转移（0/1），
    - 保存模型原始输出与统计汇总。

可配置项 (Optional configs):
    - `MODEL_NAME`, `INPUT_DIR`, `OUTPUT_DIR` 等脚本顶部常量
    - 环境变量 `DASHSCOPE_API_KEY` 用于 API 鉴权

输入/输出路径 (Input/Output paths):
    - 输入: `LLM/data/input/test_set.json`
    - 输出: `LLM/data/results/1_zero_shot_naive/{MODEL_NAME}/output_results.json`
    - 统计: `LLM/data/results/1_zero_shot_naive/{MODEL_NAME}/summary_metrics.json`
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
INPUT_DIR = "LLM/data/input"                      # 存放输入 JSON 文件的目录
OUTPUT_DIR = f"LLM/data/results/1_zero_shot_naive/{MODEL_NAME}"

# 【关键修改】：不再读取原始的全量数据，而是读取拆分后的固定测试集
DATA_FILE = f"{INPUT_DIR}/test_set.json" 

OUTPUT_FILE = f"{OUTPUT_DIR}/output_results.json" # 结果保存路径
SUMMARY_FILE = f"{OUTPUT_DIR}/summary_metrics.json" # 统计指标保存路径
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
        else:
            print(f"⚠️ 无效预测结果，无法统计准确率: {model_output} {model_output.get('error_message', '未知错误')}")

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
    调用百炼 API，返回包含所有模型输出信息的完整字典（不再只返回 int）
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
        
        # 1. 提取思考过程
        reasoning = getattr(message, 'reasoning_content', None)
        if not reasoning and hasattr(message, 'model_dump'):
             reasoning = message.model_dump().get('reasoning_content')
             
        # 2. 提取最终答案
        result_text = message.content.strip()
        
        # 3. 提取 Token 消耗统计 (转换为字典格式方便存 JSON)
        usage = completion.usage.model_dump() if hasattr(completion.usage, 'model_dump') else dict(completion.usage)
        
        # 尝试转换为整数
        try:
            predicted_label = int(result_text)
        except ValueError:
            predicted_label = -1
            
        # 返回打包好的所有数据
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
    
    print(f"加载固定测试集数据 ({DATA_FILE}) 中...")
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        patients_data = json.load(f)
    
    total_samples = len(patients_data)
    final_results = load_json_if_exists(OUTPUT_FILE)
    valid_samples, correct_predictions = summarize_results(final_results)

    if final_results:
        print(f"📌 已加载历史结果 {len(final_results)} 条，后续只处理新增 patient_id。")

    print(f"Model: {MODEL_NAME}, 开始 Zero-Shot 评估，共 {total_samples} 条数据...\n")
    print("-" * 50)

    for patient_id, data in patients_data.items():
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

        prompt = build_zero_shot_prompt(metrics)

        # 调用 API 获取打包字典
        result_dict = predict_metastasis(prompt)
        predicted_label = result_dict.get("predicted_label", -1)
        
        # 将真实标签和模型返回的所有信息组合记录
        final_results[patient_id] = {
            "true_label": true_label,
            "model_output": result_dict  # 这里包含了 reasoning, usage 等一切内容
        }

        save_json(OUTPUT_FILE, final_results)
        
        # 终端仅输出极简日志，不打印思考过程
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
        print(f"Model: {MODEL_NAME}, Zero-Shot 基线准确率: {accuracy:.2%}")
        
        # 构造统计字典并写入新的 JSON 文件
        summary_stats = {
            "experiment_type": "Zero-Shot Baseline",
            "model_name": MODEL_NAME,
            "test_set_file": DATA_FILE,  # 记录使用的是哪个测试集
            "total_samples": total_samples,
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