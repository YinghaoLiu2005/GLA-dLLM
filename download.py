import os
from huggingface_hub import snapshot_download

# 配置参数
repo_id = "nvidia/Llama-Nemotron-Post-Training-Dataset"
local_dir = "/data/yinghaoliu/datasets"

# 指定只下载 SFT/chat 目录下的所有文件
# 如果你不加这个参数，它会下载整个 Dataset 仓库
allow_patterns = ["SFT/code/code_v1.1.jsonl"]

print(f">>> 开始下载数据集: {repo_id}")
print(f">>> 目标目录: {local_dir}")
print(f">>> 过滤模式: {allow_patterns}")

try:
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",         # 关键：指定这是数据集，不是模型
        local_dir=local_dir,
        local_dir_use_symlinks=False,# 下载真实文件，而非软链接
        resume_download=True,        # 支持断点续传
        allow_patterns=allow_patterns,
        max_workers=8                # 开启多线程下载
    )
    print(">>> 下载完成！")
except Exception as e:
    print(f"!!! 下载出错: {e}")