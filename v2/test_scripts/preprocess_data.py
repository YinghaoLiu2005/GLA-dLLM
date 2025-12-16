import os
import sys
import transformers
from transformers import AutoTokenizer, HfArgumentParser
from datasets import DatasetDict, Dataset

# 引入 lmflow 的模块
# 假设脚本在 v2/test_scripts/ 下，我们需要把 v2/src 加入 path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from lmflow.args import ModelArguments, DatasetArguments, AutoArguments
from lmflow.datasets.dataset import Dataset as LMFlowDataset
from lmflow.models.auto_model import AutoModel

def main():
    # ================= 配置区域 =================
    model_name_or_path = "/data/yinghaoliu/GLA-dLLM/trained_models/Fast_dLLM_v2_1.5B"
    # 确保这里指向的是包含 jsonl 的目录
    dataset_path = "/data/yinghaoliu/datasets/SFT/code" 
    output_path = "/data/yinghaoliu/datasets/SFT/code/processed_code_v1.1_512"
    block_size = 512
    num_proc = 32
    # ===========================================

    print(f"Loading tokenizer from {model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    # 构造 lmflow 参数
    data_args = DatasetArguments(
        dataset_path=dataset_path,
        block_size=block_size,
        disable_group_texts=False,
        preprocessing_num_workers=num_proc,
    )
    
    # 指定文件名模式，确保只加载你想要的文件
    data_args.file_pattern = "code_v1.1_clean.jsonl"

    print(f"Initializing LMFlow Dataset from {dataset_path}...")
    # 1. 使用 LMFlow 逻辑加载数据
    dataset = LMFlowDataset(data_args=data_args)
    raw_dataset = dataset.backend_dataset

    # 2. Sharding: 只保留 1/8 数据
    print(">>> Sharding dataset: keeping 1/8 of the data...")
    if isinstance(raw_dataset, DatasetDict):
        for split in raw_dataset.keys():
            raw_dataset[split] = raw_dataset[split].shard(num_shards=8, index=0)
        # 简单起见，打印第一个 split 的大小
        first_split = list(raw_dataset.keys())[0]
        print(f">>> Dataset size after sharding ({first_split}): {len(raw_dataset[first_split])}")
    else:
        raw_dataset = raw_dataset.shard(num_shards=8, index=0)
        print(f">>> Dataset size after sharding: {len(raw_dataset)}")

    # 3. 确定文本列名
    column_names = raw_dataset.column_names
    if isinstance(column_names, dict):
        first_split = list(column_names.keys())[0]
        column_names = column_names[first_split]
        
    print(f"Detected columns: {column_names}")
    
    text_column_name = "text"
    if "text" not in column_names:
        if "input" in column_names:
            text_column_name = "input"
        elif "content" in column_names:
            text_column_name = "content"
        elif "code" in column_names:
            text_column_name = "code"
        else:
            text_column_name = column_names[0]
    print(f"Using '{text_column_name}' as text column.")

    # === 关键修复：过滤掉非字符串数据 ===
    print(f"Filtering invalid (None/Non-string) entries in column '{text_column_name}'...")
    
    def filter_invalid(example):
        val = example.get(text_column_name)
        # 必须不是 None，且必须是字符串，且长度大于0
        return val is not None and isinstance(val, str) and len(val) > 0

    if isinstance(raw_dataset, DatasetDict):
        for split in raw_dataset.keys():
            raw_dataset[split] = raw_dataset[split].filter(filter_invalid, num_proc=num_proc)
    else:
        raw_dataset = raw_dataset.filter(filter_invalid, num_proc=num_proc)
    print(">>> Filtering done.")
    # =================================

    # 4. Tokenization (增强鲁棒性)
    def tokenize_function(examples):
        # 获取文本列表
        texts = examples[text_column_name]
        # 双重保险：如果过滤漏了，强制转字符串，None 转空串
        # 这一步能彻底防止 ValueError: text input must be of type `str`
        safe_texts = [str(t) if t is not None else "" for t in texts]
        return tokenizer(safe_texts)

    print("Running tokenization...")
    tokenized_datasets = raw_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=column_names, # 移除原始文本列，只留 input_ids
        desc="Tokenizing",
    )

    # 5. Grouping
    from itertools import chain
    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        # 复制 input_ids 到 labels (自回归训练标准做法)
        result["labels"] = result["input_ids"].copy()
        return result

    print(f"Grouping texts into chunks of {block_size}...")
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=num_proc,
        desc="Grouping",
    )

    print(f"Saving processed dataset to {output_path}...")
    lm_datasets.save_to_disk(output_path)
    print("Done!")

if __name__ == "__main__":
    main()