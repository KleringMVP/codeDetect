import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
sys.path.append('/home/aistudio/external-libraries')
import Levenshtein
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import torch, json, time

def process_human_data():
    """处理人类样本数据（前200条）"""
    # 加载原始人类代码（前200）
    with open('/home/aistudio/RealData/human-code.jsonl', 'r') as f:
        human_code_data = [json.loads(line) for line in f][:200]  # 添加切片
    
    # 创建匹配字典（使用前200条）
    human_code_dict = { 
        entry['original_length']: entry['python_code'] 
        for entry in human_code_data
    }
    
    # 加载人类补全样本（前200）
    with open('/home/aistudio/RealData/generated_completions/human-regen.jsonl', 'r') as f:
        human_regen_data = [json.loads(line) for line in f][:200]  # 添加切片
    
    gold_distances = []
    for entry in tqdm.tqdm(human_regen_data, desc="Processing Human Data"):
        # 通过original_length获取原始代码
        original_code = human_code_dict.get(entry['original_length'], None)
        
        if not original_code:
            print(f"Warning: 找不到匹配的原始代码，original_length={entry['original_length']}")
            continue
        
        # 计算所有补全样本的编辑距离
        distances = []
        for comp_key in ['completion_0', 'completion_1', 'completion_2']:
            completion = entry['completions'].get(comp_key, '')
            if completion:
                distances.append(Levenshtein.distance(completion, original_code))
        
        if distances:
            gold_distances.append(np.mean(distances))
    
    return gold_distances
def process_machine_data():
    """处理机器样本数据（前200条）"""
    # 加载原始机器代码（前200）
    machine_original_path = '/home/aistudio/detegpt4Code/data/variant3-incoder.jsonl'
    with open(machine_original_path, 'r') as f:
        machine_original_data = [json.loads(line) for line in f][:200]  # 添加切片
    
    # 加载机器补全样本（前200）
    machine_regen_path = '/home/aistudio/detegpt4Code/result/regen_incoder6B_v3.jsonl'
    with open(machine_regen_path, 'r') as f:
        machine_regen_data = [json.loads(line) for line in f][:200]  # 添加切片
    
    fim_distances = []
    for entry in tqdm.tqdm(machine_regen_data, desc="Processing Machine Data"):
        # 获取对应的原始代码
        original_entry = next(x for x in machine_original_data if x['index'] == entry['index'])
        original_code = original_entry['Variant3']['choices'][0]['message']['content']
        
        # 计算所有补全样本的编辑距离
        distances = []
        for comp_key in ['Variant3_full_gen_index_0', 'Variant3_full_gen_index_1', 'Variant3_full_gen_index_2']:
            completion = entry.get(comp_key, '')
            # 删除前缀语句
            completion = completion.replace("Please complete the code as given and don't give other test annotation:", "")
            if completion:
                distances.append(Levenshtein.distance(completion, original_code))
        
        if distances:
            fim_distances.append(np.mean(distances))
    
    return fim_distances

# 主处理流程
if __name__ == "__main__":
    # 处理数据
    gold_prob_all_Leven = process_human_data()
    fim_prob_all_Leven = process_machine_data()

    # 生成标签和得分
    scores_Leven = np.array(fim_prob_all_Leven + gold_prob_all_Leven)
    labels_Leven = np.array([1]*len(fim_prob_all_Leven) + [0]*len(gold_prob_all_Leven))

    # 计算ROC曲线
    fpr, tpr, _ = roc_curve(labels_Leven, scores_Leven)
    roc_auc = auc(fpr, tpr)

    # 绘制ROC曲线
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Human vs Machine Code Detection (Levenshtein Distance)')
    plt.legend(loc="lower right")
    plt.savefig('/home/aistudio/detegpt4Code/lev-results/v3.png')
    plt.show()