import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
sys.path.append('/home/aistudio/external-libraries')
import json
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 设备配置
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# 模型配置
model_name = 'codeparrot/codeparrot-small'
max_length = 1024
truncate_ratio = 0.5

# 初始化模型
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def get_logprob(text):
    """计算文本的平均负对数概率（最后1%）"""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    input_ids = inputs['input_ids'][0]
    
    # 计算每个token的负对数概率（跳过第一个token）
    neg_log_probs = [-log_probs[0, i-1, input_ids[i]].item() for i in range(1, input_ids.size(0))]
    
    # 截取最后truncate_ratio部分
    start_idx = int(len(neg_log_probs) * truncate_ratio)
    return np.mean(neg_log_probs[start_idx:]) if neg_log_probs[start_idx:] else 0

def process_human_data():
    """处理人类样本数据（返回带详细信息的列表）"""
    with open('/home/aistudio/RealData/human-code.jsonl', 'r') as f:
        human_code_data = [json.loads(line) for line in f][:200]
    
    human_code_dict = {entry['original_length']: entry['python_code'] for entry in human_code_data}
    
    with open('/home/aistudio/RealData/generated_completions/human-regen.jsonl', 'r') as f:
        human_regen_data = [json.loads(line) for line in f][:200]
    
    detailed_scores = []
    for idx, entry in enumerate(tqdm.tqdm(human_regen_data, desc="Processing Human Data")):
        original_code = human_code_dict.get(entry['original_length'])
        if not original_code:
            continue
        
        original_score = get_logprob(original_code)
        comp_scores = []
        for comp_key in ['completion_0', 'completion_1', 'completion_2']:
            completion = entry['completions'].get(comp_key, '')
            if completion:
                comp_scores.append(get_logprob(completion))
        
        if comp_scores:
            score = np.mean(comp_scores) - original_score
            detailed_scores.append({
                "type": "human",
                "index": f"human_{idx}",
                "original_length": entry['original_length'],
                "score": score,
                "original_preview": original_code[:50] + "...",
                "completion_preview": completion[:50] + "..." if completion else ""
            })
    
    return detailed_scores

def process_machine_data():
    """处理机器样本数据（返回带详细信息的列表）"""
    machine_original_path = '/home/aistudio/detegpt4Code/data/variant3-incoder.jsonl'
    with open(machine_original_path, 'r') as f:
        machine_original_data = [json.loads(line) for line in f][:200]
    
    machine_regen_path = '/home/aistudio/detegpt4Code/result/regen_incoder6B_v3.jsonl'
    with open(machine_regen_path, 'r') as f:
        machine_regen_data = [json.loads(line) for line in f][:200]
    
    detailed_scores = []
    for idx, entry in enumerate(tqdm.tqdm(machine_regen_data, desc="Processing Machine Data")):
        original_entry = next((x for x in machine_original_data if x['index'] == entry['index']), None)
        if not original_entry:
            continue
        
        original_code = original_entry['Variant3']['choices'][0]['message']['content']
        original_score = get_logprob(original_code)
        
        comp_scores = []
        for comp_key in ['Variant3_full_gen_index_0', 'Variant3_full_gen_index_1', 'Variant3_full_gen_index_2']:
            completion = entry.get(comp_key, '')
            completion = completion.replace("Please complete the code as given and don't give other test annotation:", "")
            if completion:
                comp_scores.append(get_logprob(completion))
        
        if comp_scores:
            score = np.mean(comp_scores) - original_score
            detailed_scores.append({
                "type": "machine",
                "index": entry['index'],
                "score": score,
                "original_preview": original_code[:50] + "...",
                "completion_preview": completion[:50] + "..." if completion else ""
            })
    
    return detailed_scores

def print_detailed_scores(samples, num=20, sample_type="Human"):
    """打印详细分数信息"""
    print(f"\n{'='*40}")
    print(f"{sample_type}样本详细分数（前{num}个）")
    print(f"{'索引':<15} | {'分数':<10} | {'原始代码预览':<30} | {'补全代码预览':<30}")
    print("-"*85)
    
    for sample in samples[:num]:
        print(f"{sample['index']:<15} | {sample['score']:>8.4f} | "
              f"{sample['original_preview']:<30} | "
              f"{sample['completion_preview']:<30}")

def analyze_results(human_scores, machine_scores):
    """结果分析"""
    # 提取纯分数
    human_scores_values = [x['score'] for x in human_scores]
    machine_scores_values = [x['score'] for x in machine_scores]
    
    # 打印统计信息
    print("\n统计分析:")
    print(f"人类样本平均分数: {np.mean(human_scores_values):.4f} ± {np.std(human_scores_values):.4f}")
    print(f"机器样本平均分数: {np.mean(machine_scores_values):.4f} ± {np.std(machine_scores_values):.4f}")
    print(f"人类样本分数范围: [{np.min(human_scores_values):.4f}, {np.max(human_scores_values):.4f}]")
    print(f"机器样本分数范围: [{np.min(machine_scores_values):.4f}, {np.max(machine_scores_values):.4f}]")

    # 绘制分布图
    plt.figure(figsize=(10, 6))
    plt.hist(human_scores_values, alpha=0.5, bins=30, label='Human', color='blue')
    plt.hist(machine_scores_values, alpha=0.5, bins=30, label='Machine', color='red')
    plt.axvline(-0.4, color='black', linestyle='--', label='Threshold')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title('Score Distribution')
    plt.legend()
    plt.savefig('/home/aistudio/detegpt4Code/prompt_result/v3/last_token/distribution.png')
    plt.close()

def plot_roc(human_scores, machine_scores):
    """绘制ROC曲线"""
    human_scores_values = [x['score'] for x in human_scores]
    machine_scores_values = [x['score'] for x in machine_scores]
    
    scores = np.array(machine_scores_values + human_scores_values)
    labels = np.array([1]*len(machine_scores_values) + [0]*len(human_scores_values))
    
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Human vs Machine Detection (Log Probability)')
    plt.legend(loc="lower right")
    plt.savefig('/home/aistudio/detegpt4Code/prompt_result/v3/last_token/v3-50%.png')
    plt.close()

if __name__ == "__main__":
    # 处理数据
    human_data = process_human_data()
    machine_data = process_machine_data()
    
    # 打印详细分数
    print_detailed_scores(human_data, num=20, sample_type="Human")
    print_detailed_scores(machine_data, num=20, sample_type="Machine")
    
    # 分析结果
    analyze_results(human_data, machine_data)
    
    # 绘制ROC曲线
    plot_roc(human_data, machine_data)
    
    # 计算性能指标
    human_scores = [x['score'] for x in human_data]
    machine_scores = [x['score'] for x in machine_data]
    all_scores = np.array(machine_scores + human_scores)
    labels = np.array([1]*len(machine_scores) + [0]*len(human_scores))
    
    fpr, tpr, thresholds = roc_curve(labels, all_scores)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    predictions = np.where(all_scores < optimal_threshold, 0, 1)
    accuracy = np.mean(predictions == labels)
    precision = np.sum((predictions == 1) & (labels == 1)) / np.sum(predictions == 1)
    recall = np.sum((predictions == 1) & (labels == 1)) / np.sum(labels == 1)
    f1 = 2 * precision * recall / (precision + recall)
    
    print("\n最终性能指标:")
    print(f"最佳阈值: {optimal_threshold:.4f}")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1值: {f1:.4f}")