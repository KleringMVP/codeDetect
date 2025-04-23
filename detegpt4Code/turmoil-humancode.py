import sys
sys.path.append('/home/aistudio/external-libraries')
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import os
import json
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--input_file', default="/home/aistudio/RealData/human-code.jsonl")  # 新生成的JSONL文件
parser.add_argument('--output_dir', default="/home/aistudio/RealData/generated_completions")         # 新输出目录
parser.add_argument('--model', default="facebook/incoder-6B")                                        # 保持原模型
parser.add_argument('--max_new_tokens', default=300, type=int)
parser.add_argument('--truncate_ratio', default=0.7, type=float)                                    # 截断比例
parser.add_argument('--cache_dir', default="/home/aistudio/incoder-6B/models--facebook--incoder-6B/snapshots/18c1a34c6f3c34f07b7a39b1ca4d07777fb9465c", type=str)
args = parser.parse_args()

# 设备设置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 确保目录存在
os.makedirs(args.cache_dir, exist_ok=True)
os.makedirs(args.output_dir, exist_ok=True)

# 加载模型
tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = AutoModelForCausalLM.from_pretrained(args.model, cache_dir=args.cache_dir).to(device)

# 加载新数据集
def load_new_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                item = json.loads(line)
                # 验证必须字段
                if 'python_code' not in item:
                    print(f"警告：跳过缺失 'python_code' 字段的行: {line.strip()}")
                    continue
                data.append(item)
            except json.JSONDecodeError:
                print(f"错误：无法解析的JSON行: {line.strip()}")
    return data

data = load_new_data(args.input_file)
print(f"成功加载 {len(data)} 条数据")

# 定义补全函数（保持原逻辑）
def get_incoder_response(prompt, max_tokens):
    prompt_with_instruction = "Please complete the code as given and don't give other test annotation:" + prompt
    inputs = tokenizer(prompt_with_instruction, return_tensors="pt", padding=True, truncation=True)
    generated_ids = model.generate(
        inputs["input_ids"].to(device),
        attention_mask=inputs["attention_mask"].to(device),
        max_length=inputs["input_ids"].shape[1] + max_tokens,
        num_return_sequences=3,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id
    )
    return [tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]

# 进度追踪函数
def get_processed_indexes(output_file):
    processed = set()
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    processed.add(item['index'])
                except:
                    continue
    return processed

# 输出文件路径
output_file = os.path.join(args.output_dir, f"completions_ratio{args.truncate_ratio}.jsonl")
processed_indexes = get_processed_indexes(output_file)

# 主处理逻辑
for idx, item in tqdm(enumerate(data), total=len(data)):
    if idx in processed_indexes:
        continue

    # 直接获取代码内容
    full_code = item['python_code']
    truncate_len = int((1 - args.truncate_ratio) * len(full_code))
    prompt = full_code[:truncate_len]

    # 生成补全
    try:
        completions = get_incoder_response(prompt, args.max_new_tokens)
    except Exception as e:
        print(f"生成失败 index={idx}: {str(e)}")
        continue

    # 构建输出项
    output_item = {
        "index": idx,
        "original_length": item.get("original_length", len(full_code)),
        "truncated_prompt": prompt,
        "completions": {f"completion_{i}": comp for i, comp in enumerate(completions)}
    }

    # 写入文件
    with open(output_file, 'a') as f:
        f.write(json.dumps(output_item) + "\n")

print(f"处理完成！结果保存至: {output_file}")