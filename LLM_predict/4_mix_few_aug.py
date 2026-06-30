"""
说明 (Purpose):
    结合知识增强与 Few-Shot 示例的混合策略：
    - 在 Prompt 中同时包含详尽知识库与从候选池抽取的示例，
    - 对固定测试集逐样本预测并保存输出与统计。

可配置项 (Optional configs):
    - `MODEL_NAME`, `INPUT_DIR`, `OUTPUT_DIR`, `FEW_SHOT_POS_COUNT`, `FEW_SHOT_NEG_COUNT` 等
    - 环境变量 `DASHSCOPE_API_KEY` 用于 API 鉴权

输入/输出路径 (Input/Output paths):
    - 输入: `LLM/data/input/few_shot_pool.json`, `LLM/data/input/test_set.json`
    - 输出: `LLM/data/results/4_mix_few_aug/{MODEL_NAME}/output_{p}p_{n}n.json`
    - 统计: `LLM/data/results/4_mix_few_aug/{MODEL_NAME}/summary_{p}p_{n}n.json`
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
OUTPUT_DIR = f"LLM/data/results/4_mix_few_aug/{MODEL_NAME}"  # 修改了输出目录以区分实验

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
        "你是一个极其严谨的预测根治性手术后结直肠癌（CRC）患者在10年随访时间内发生异时性远处转移（MDM）预测算法系统。你的任务是根据给定的患者基线时测定的多个医学指标，"
        "严格比对以下知识库中的权重与方向，预测该患者在10年内是否会发生远处转移。\n\n"
        
        "【全量特征研判知识库】\n"
        "请逐一核对以下五大维度中的指标。注意：(↑)表示该数值升高或为阳性时增加转移风险；(↓)表示该数值降低或缺失时增加转移风险（即高值为保护因素）。\n\n"
        
        "第一维度：局部病理与解剖特征（局部肿瘤负荷基线）\n"
        "- Carcinoma nodule (癌结节): 高危因子。存在(1)或数量越多，转移风险急剧上升 (↑)。\n"
        "- PLN (阳性淋巴结数): >0 代表发生区域淋巴结转移，数值越大转移概率越高 (↑)。\n"
        "- Vascular invasion (血管侵犯): 1(Yes) 为危险因素，代表血行转移通道初步打开 (↑)。\n"
        "- Perineural invasion (神经侵犯): 1(Yes) 为危险因素，局部浸润强 (↑)。\n"
        "- Differentiation grade (分化程度): G3(低分化) 增加转移概率 (↑)；G1(高分化) 转移风险低。\n"
        "- T stage & N stage: T3-T4 或 N2 代表侵犯深，风险高 (↑)；T1-T2 或 N0 代表侵犯浅，风险低。\n"
        "- TNLE (检出淋巴结总数): 辅助指标。预测效能一般，过低可能低估淋巴结分期 (↓)。\n"
        "- Tumor size (肿瘤大小): 预测效能一般，越大则肿瘤负荷越重，可能促进转移 (↑)。\n"
        "- Colonic obstruction (结肠梗阻): 预测效能一般，1(Yes) 提示晚期压迫，可能易转移 (↑)。\n\n"
        
        "第二维度：免疫微环境与免疫细胞（系统性防御壁垒，核心权重）\n"
        "- Treg cells %: 最核心的免疫抑制高危指标。数值越高，免疫逃逸越严重，转移风险急剧上升 (↑)。\n"
        "- CD4+ count, CD3+ count, NK cells %: 核心免疫防线（保护因子）。这些数值充沛代表抗肿瘤免疫力强；若显著降低，代表免疫耗竭，转移风险上升 (↓)。\n"
        "- CD19+ B cells % & CD19+ count: 体液免疫辅助指标。数值降低通常伴随更高的转移风险 (↓)。\n\n"
        
        "第三维度：肿瘤标志物与细胞代谢（微转移的系统性雷达，核心权重）\n"
        "- CEA: 最敏感的标志物，异常升高强烈提示微转移存在 (↑)。\n"
        "- CA242, CA199: 消化道肿瘤标志物。异常升高提示转移风险 (↑)。\n"
        "- LDH (乳酸脱氢酶): 代谢异常标志。> 171.8 U/L 时，转移风险显著增加 (↑)。\n"
        "- Total protein (总蛋白): 营养与消耗指标。< 60 g/L (低蛋白血症) 提示恶液质，增加转移风险 (↓)。\n\n"
        
        "第四维度：全身炎症与细胞因子（促转移微环境）\n"
        "- NLR (中性粒/淋巴比): 全身炎症指标。升高提示可能增加转移风险 (↑)。\n"
        "- mGPS: 炎症与营养综合评分。> 0 时提示系统性炎症，可能增加转移风险 (↑)。\n"
        "- IL8: 促炎细胞因子。高表达促进转移 (↑)。\n\n"
        
        "第五维度：基因突变与家族史（生物学潜能）\n"
        "- KRAS mutant, BRAF mutant: 1(Yes) 为突变型，代表肿瘤侵袭性强，转移风险增加 (↑)。\n"
        "- PMS2, MSH6: 错配修复蛋白。若为 0(Negative/缺失)，代表 dMMR/MSI-H，为保护因素，预示较低的转移风险 (↓)。\n\n"
        
        "【临床冲突裁决规则（极其重要）】\n"
        "在实际临床中，指标往往是矛盾的，请严格按照以下优先级进行对抗性推理：\n"
        "1. 即使第一维度（病理分期，如N2、脉管神经侵犯）和第五维度（KRAS突变）显示为极高危，\n"
        "2. 但如果该患者的第二维度（免疫防线，如极低的Treg、充沛的CD3/CD4/NK细胞）和第三维度（肿瘤标志物如CEA, LDH）表现为完全正常或高度保护状态，\n"
        "3. 这说明患者强大的系统性免疫清除了游离的微转移癌细胞。此时，“免疫与代谢的保护效力”应压倒“局部病理的危险性”。你必须综合判定为不发生转移。\n\n"
        
        "【输出格式限制】\n"
        "严格只输出一个数字，绝对不要包含任何解释、分析过程、标点或其他字符：\n"
        "0 代表 未转移 (No MDM)\n"
        "1 代表 已转移 (MDM)"
        
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
        "【执行指令】\n"
        "请严格比对下方患者的具体数值，在内心进行『风险因子』与『保护因子』的加减法对抗计算。请特别注意应用上述【临床冲突裁决规则】来处理矛盾指标。\n\n"
        
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
    print(f"Model: {MODEL_NAME}, 开始 Few-Shot&Prompt Aug 评估，共 {total_test_samples} 条测试数据...\n")
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
        print(f"【评估完成】")
        print(f"Model: {MODEL_NAME}, 总测试样本: {valid_samples} (成功响应)")
        print(f"Model: {MODEL_NAME}, 正确预测数: {correct_predictions}")
        print(f"Model: {MODEL_NAME}, Few-Shot&Prompt Aug 准确率: {accuracy:.2%}")
        
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
        print(f"Model: {MODEL_NAME}, ✅ 统计汇总信息已保存至: {SUMMARY_FILE}\n")
        
    else:
        print(f"Model: {MODEL_NAME}, 没有有效的预测结果，请检查 API 配置或网络。")
        
    print(f"总执行耗时: {elapsed_time:.2f} 秒")