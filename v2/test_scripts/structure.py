import os
from transformers import AutoConfig, AutoModelForCausalLM

# 确保路径是绝对路径，防止相对路径找不到文件
model_path = os.path.abspath("BiDeltaDiff/trained_models")

print(f"尝试从以下路径加载配置: {model_path}")

try:
    # 1. 先只加载配置 (Config)
    config = AutoConfig.from_pretrained(
        model_path, 
        trust_remote_code=True
    )
    
    # 2. 使用配置初始化模型 (不加载权重，使用的是随机初始化权重)
    # 这样速度极快，且不会因为权重文件损坏或内存不足而失败
    model = AutoModelForCausalLM.from_config(
        config, 
        trust_remote_code=True
    )
    
    # 3. 打印结构
    print("----- 模型结构 -----")
    print(model)
    print("----- 结束 -----")

except Exception as e:
    print(f"发生错误: {e}")