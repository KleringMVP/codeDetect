import os
import sys
import json
import argparse
import random
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="/home/aistudio/detectgpt4code/data/Variant3_gpt-3.5-turbo.jsonl")  # 假设文件名为 variant3_data.jsonl
parser.add_argument('--output_dir', default="/home/aistudio/detectgpt4code/result")  # 输出文件夹路径
parser.add_argument('--model', default="facebook/incoder-6B")  # 使用 incoder-6B 模型
parser.add_argument('--max_new_tokens', default=300, type=int)
parser.add_argument('--truncate_ratio', default=0.7, type=float)  # 截断后百分之70
parser.add_argument('--cache_dir', default="/home/aistudio/detectgpt4code/incoder-6B", type=str)  # 自定义模型缓存路径
args = parser.parse_args()

# 强制使用 GPU 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 确保目标缓存目录存在
os.makedirs(args.cache_dir, exist_ok=True)

# 下载并加载模型和分词器
model_name = args.model  # 从命令行传入模型名称
print(f"Downloading and loading model: {model_name} into cache directory: {args.cache_dir}")

# 下载并加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=args.cache_dir)

# 如果 pad_token 不存在，则手动添加
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    print("Pad token added:", tokenizer.pad_token)

# 下载并加载模型
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=args.cache_dir).to(device)

# 加载数据集
with open(args.dataset, 'r') as f:
    data = [json.loads(x) for x in f.read().strip().split("\n")]

# 确保输出目录存在
os.makedirs(args.output_dir, exist_ok=True)

# 输出文件路径修改为新目录
output_file = f"{args.output_dir}/regen_incoder6B_{args.truncate_ratio}.jsonl"
random.seed(43)

# 初始化输出文件并写入文件头（如果需要）
with open(output_file, 'a') as f:
    f.write("")  # 如果需要添加文件头，可以在此修改

# 定义补全函数
def get_incoder_response(prompt, max_tokens):
    prompt_with_instruction = "Please complete the code as given and don't give other test annotation:" + prompt
    inputs = tokenizer(prompt_with_instruction, return_tensors="pt", padding=True, truncation=True)
    
    attention_mask = inputs.get("attention_mask", torch.ones_like(inputs["input_ids"]))
    
    generated_ids = model.generate(
        inputs["input_ids"].to(device),
        attention_mask=attention_mask.to(device),
        max_length=inputs["input_ids"].shape[1] + max_tokens,
        num_return_sequences=3,  # 生成 3 个版本
        do_sample=True,  # 启用采样
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id
    )
    
    return [tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]

# 提取 content 部分的函数
def extract_code_from_variant3(variant3):
    try:
        return variant3.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    except Exception as e:
        print(f"Error extracting content: {e}")
        return ""

# 读取已处理的样本进度
def get_processed_indexes(output_file):
    processed_indexes = set()
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                sample = json.loads(line)
                processed_indexes.add(sample.get('index', -1))  # 假设每行有 index 字段
    return processed_indexes

# 获取已处理的样本索引
processed_indexes = get_processed_indexes(output_file)

# 遍历数据并进行推理
for idx, dd in tqdm(enumerate(data), total=len(data)):
    if idx in processed_indexes:
        continue  # 如果当前样本已经处理过，则跳过

    variant3 = dd['Variant3']  # 获取 Variant3 字段的内容
    code_content = extract_code_from_variant3(variant3)
    
    truncate_len = int((1 - args.truncate_ratio) * len(code_content))  # 截断掉后70%，保留前30%
    prompt = code_content[:truncate_len]  # 保留前30%的内容

    # 生成 3 个补全版本
    gen_completions = get_incoder_response(prompt, args.max_new_tokens)

    # 将结果保存到输出列表中
    output_data = {"Variant3_truncated": prompt, "index": idx}
    for i, gen in enumerate(gen_completions):
        output_data[f"Variant3_full_gen_index_{i}"] = gen

    # 每次生成补全后，立即写入文件
    with open(output_file, 'a') as f:
        f.write(json.dumps(output_data) + "\n")

    print(f"Sample {idx} processed, result written to {output_file}")

print(f"All results saved to {output_file}")
