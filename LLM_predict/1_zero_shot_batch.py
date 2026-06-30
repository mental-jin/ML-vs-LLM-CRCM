# batch操作会额外收费
import os
import json
import time
from openai import OpenAI

# ================= 1. 配置区 =================
MODEL_NAME = "qwen3.6-flash"
INPUT_DIR = "LLM/data/input"                      # 存放输入 JSON 文件的目录
OUTPUT_DIR = f"LLM/data/1_zero_shot_results/{MODEL_NAME}/batch"
INPUT_DATA_FILE = f"{INPUT_DIR}/input_demo.json"          # 你的原始数据文件
BATCH_TASKS_FILE = f"{OUTPUT_DIR}/batch_tasks.jsonl"   # 生成给云端的任务文件
RAW_RESULT_FILE = f"{OUTPUT_DIR}/batch_results.jsonl"  # 从云端下载的原始结果
FINAL_OUTPUT_FILE = f"{OUTPUT_DIR}/final_predictions.json" # 解析对齐后的最终清洗结果

# 初始化客户端
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
# ============================================

def build_prompt(metrics: dict) -> str:
    """构建自然语言 Prompt"""
    metrics_str = "\n".join([f"- {k}: {v}" for k, v in metrics.items()])
    return (
        "你是一位专业的肿瘤学专家。请根据以下结直肠癌患者的临床特征、免疫指标和病理数据，"
        "预测该患者是否发生了肿瘤转移（Metastasis）。\n\n"
        f"【患者指标】\n{metrics_str}\n\n"
        "【输出限制】\n请严格只输出一个数字：\n0 代表 未转移\n1 代表 已转移"
    )

def step1_prepare_batch_file():
    """步骤 1：将本地 JSON 数据转换为云端需要的 JSONL 格式"""
    print(f"\n--- 步骤 1: 生成 Batch 任务文件 ---")
    with open(INPUT_DATA_FILE, 'r', encoding='utf-8') as f:
        patients_data = json.load(f)
    
    task_count = 0
    with open(BATCH_TASKS_FILE, 'w', encoding='utf-8') as out_f:
        for patient_id, data in patients_data.items():
            metrics = data.get("metrics", {})
            prompt = build_prompt(metrics)
            
            # 构建官方要求的结构
            batch_request = {
                "custom_id": str(patient_id), # 关键：用患者 ID 作为唯一标识
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL_NAME,
                    "temperature": 0.1,
                    "messages": [
                        {'role': 'system', 'content': '你是一个严谨的医疗AI助手，严格遵守输出格式要求。'},
                        {'role': 'user', 'content': prompt}
                    ]
                }
            }
            out_f.write(json.dumps(batch_request, ensure_ascii=False) + "\n")
            task_count += 1

    print(f"✅ 成功生成 {BATCH_TASKS_FILE}，共包含 {task_count} 条数据。")

def step2_submit_and_wait():
    """步骤 2：上传文件，创建任务并轮询等待结果"""
    print(f"\n--- 步骤 2: 提交云端任务 ---")
    
    # 1. 上传文件
    with open(BATCH_TASKS_FILE, "rb") as f:
        batch_file = client.files.create(file=f, purpose="batch")
    file_id = batch_file.id
    print(f"✅ 文件已上传，File ID: {file_id}")

    # 2. 创建 Batch 任务
    batch_job = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h" 
    )
    job_id = batch_job.id
    print(f"✅ 任务已创建，Job ID: {job_id}")

    # 3. 轮询等待
    print("\n--- 步骤 3: 等待云端处理 ---")
    print("💡 提示：Batch 任务在后台排队，通常需要几分钟到十几分钟不等。")
    
    while True:
        job_status = client.batches.retrieve(job_id)
        status = job_status.status
        
        # 获取进度
        req_counts = job_status.request_counts
        completed = req_counts.completed if req_counts else 0
        total = req_counts.total if req_counts else "未知"
        
        print(f"🔄 状态: [{status.upper()}] | 进度: {completed} / {total} 完成")
        
        if status in ['completed', 'failed', 'cancelled', 'expired']:
            break
            
        time.sleep(15) # 每 15 秒查一次，避免触发 API 频率限制

    # 4. 下载结果
    if job_status.status == 'completed':
        print(f"\n--- 步骤 4: 下载结果 ---")
        output_file_id = job_status.output_file_id
        result_content = client.files.content(output_file_id).text
        
        with open(RAW_RESULT_FILE, "w", encoding="utf-8") as f:
            f.write(result_content)
        print(f"✅ 原始结果已保存至 {RAW_RESULT_FILE}")
        return True
    else:
        print(f"❌ 任务异常终止，最终状态: {job_status.status}")
        return False

def step5_parse_and_save_final():
    """步骤 5：解析原始结果，与真实标签对比，并保存为干净的 JSON"""
    print(f"\n--- 步骤 5: 解析并生成最终结果 ---")
    
    # 读取原始标签以备对比
    with open(INPUT_DATA_FILE, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
        
    final_results = {}
    correct_count = 0
    valid_count = 0

    # 逐行解析云端返回的 JSONL
    with open(RAW_RESULT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            res = json.loads(line)
            patient_id = res["custom_id"]
            
            # 提取大模型的回答
            try:
                # 检查接口是否返回了成功的 HTTP 200 状态码
                if res["response"]["status_code"] == 200:
                    raw_prediction = res["response"]["body"]["choices"][0]["message"]["content"].strip()
                    prediction = int(raw_prediction) # 转换为整数
                else:
                    prediction = None
            except Exception as e:
                prediction = None

            # 获取真实标签
            true_label = original_data.get(patient_id, {}).get("results", {}).get("Metastasis")
            
            # 记录到最终字典
            final_results[patient_id] = {
                "true_label": true_label,
                "predicted_label": prediction
            }

            # 统计正确率
            if prediction is not None and true_label is not None:
                valid_count += 1
                if prediction == true_label:
                    correct_count += 1

    # 保存最终文件
    with open(FINAL_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)
        
    print(f"✅ 清洗后的结果已保存至: {FINAL_OUTPUT_FILE}")
    
    if valid_count > 0:
        print("\n" + "="*30)
        print("🎯 基线预测结果统计")
        print(f"有效样本数: {valid_count}")
        print(f"正确预测数: {correct_count}")
        print(f"整体准确率: {correct_count / valid_count:.2%}")
        print("="*30)

# ================= 运行主程序 =================
if __name__ == "__main__":
    step1_prepare_batch_file()
    success = step2_submit_and_wait()
    if success:
        step5_parse_and_save_final()