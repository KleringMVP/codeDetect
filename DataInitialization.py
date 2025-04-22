import pandas as pd
import json

# 读取 CSV 文件
csv_file = '/home/aistudio/RealData/output.csv'
df = pd.read_csv(csv_file)

# 将 DataFrame 中的每一行转换为 JSON 并写入 JSONL 文件
with open('/home/aistudio/RealData/data0.jsonl', 'w', encoding='utf-8') as jsonl_file:
    for index, row in df.iterrows():
        # 将每行数据转换为字典格式
        row_dict = row.to_dict()
        # 将字典写入 JSONL 文件，每行一个 JSON 对象
        jsonl_file.write(json.dumps(row_dict, ensure_ascii=False) + '\n')

print("CSV 转换为 JSONL 完成！")


# import pandas as pd
# import re

# # 读取 CSV 文件
# csv_file = '/home/aistudio/RealData/variant_1_full.csv'
# df = pd.read_csv(csv_file)

# # 删除指定的列
# df.drop(columns=['Source Name', 'GPT Answer', 'variant'], inplace=True)

# # 使用正则表达式提取 Problem 列中的有效内容，同时保留三引号
# def extract_content(text):
#     # 匹配并提取三重引号内的内容，保留三引号
#     match = re.search(r'("""(.*?)""")', text, re.DOTALL)
#     if match:
#         return match.group(0)  # 返回包含三引号的内容
#     return ''  # 如果没有匹配到，返回空字符串

# # 应用函数处理 Problem 列
# df['Problem'] = df['Problem'].apply(extract_content)

# # 保存修改后的文件
# df.to_csv('/home/aistudio/RealData/output.csv', index=False)

# print("操作完成，文件已保存为 output.csv")

