import json
from tqdm import tqdm

# 配置路径
input_file = "/data/yinghaoliu/datasets/SFT/code/code_v1.1.jsonl" 
output_file = "/data/yinghaoliu/datasets/SFT/code/code_v1.1_clean.jsonl"

# 阈值设置：代码文件通常不会超过 10万字符
# 如果一行超过 100,000 字符，通常是压缩代码或垃圾数据，Tokenizer 处理它会极慢
MAX_CHAR_LENGTH = 50000 

print(f"开始清洗数据，移除长度超过 {MAX_CHAR_LENGTH} 的样本...")

removed_count = 0
kept_count = 0

with open(input_file, 'r', encoding='utf-8') as fin, \
     open(output_file, 'w', encoding='utf-8') as fout:
    
    for line in tqdm(fin):
        try:
            data = json.loads(line)
            # 假设你的文本字段叫 "text" 或 "content"，请根据实际 jsonl 调整
            # 这里做一个通用检查，检查所有 value
            is_too_long = False
            for key, value in data.items():
                if isinstance(value, str) and len(value) > MAX_CHAR_LENGTH:
                    is_too_long = True
                    break
            
            if is_too_long:
                removed_count += 1
                continue
            
            fout.write(line)
            kept_count += 1
        except Exception as e:
            print(f"解析错误: {e}")
            continue

print(f"清洗完成！")
print(f"保留数据: {kept_count}")
print(f"移除由于过长导致卡死的数据: {removed_count}")
print(f"新文件位置: {output_file}")