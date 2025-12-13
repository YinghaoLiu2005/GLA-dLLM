import sys
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import numpy as np
import random

# Add GLADream to Python path to allow for dynamic registration
gla_dream_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if gla_dream_path not in sys.path:
    sys.path.insert(0, gla_dream_path)

try:
    from gla_dream.configuration_gla_dream import GLADreamConfig
    from gla_dream.modeling_gla_dream import GLADreamModel

    # Register the custom model with the AutoModel classes
    # This allows from_pretrained to work with the "GLADream" model type
    AutoConfig.register("GLADream", GLADreamConfig)
    AutoModelForCausalLM.register(GLADreamConfig, GLADreamModel)
    print("[INFO] Successfully registered GLADream custom model.")
except ImportError as e:
    print(f"[ERROR] Failed to import or register GLADream. Make sure the gla_dream directory is accessible.")
    print(f"[ERROR] Details: {e}")
    sys.exit(1)


# --- Use the local checkpoint ---
model_name = "/data/yinghaoliu/GLA-dLLM/trained_models/finetuning_gla_dream/step-26000"

print(f"[INFO] Loading model from: {model_name}")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="cuda:0",
    trust_remote_code=True  # Important for custom models
)

print(f"[INFO] Loading tokenizer from: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"[INFO] pad_token_id not set, using eos_token_id: {tokenizer.eos_token_id}")
# Initialize conversation
messages = []

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

fix_seed(42)
print("\nChatbot started! Type 'exit' to quit the conversation.")
print("Type 'clear' to clear conversation history.")
print("-" * 50)

while True:
    # Get user input
    user_input = input("User: ").strip()
    
    # Check if exit
    if user_input.lower() == "exit":
        print("Goodbye!")
        break
    
    # Check if clear conversation history
    if user_input.lower() == "clear":
        messages = messages[:1] if messages and messages[0]["role"] == "system" else []
        print("Conversation history cleared!")
        continue
    
    if not user_input:
        continue
    
    messages.append({"role": "user", "content": user_input})
    
    # Use the tokenizer's chat template
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as e:
        print(f"[ERROR] Failed to apply chat template: {e}")
        # Fallback for models without a proper chat template
        text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages]) + "\nassistant:"

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    input_ids = model_inputs["input_ids"]
    
    # 1. 计算并对齐输入长度 (Pad Input Length)
    # 这是为了确保 (max_length - input_length) 可以是 block_size 的倍数
    block_size = model.config.block_size  # 从模型配置中获取，通常是 32
    input_length = input_ids.shape[1]
    
    # 计算需要填充多少个 token 才能达到 block_size 的下一个倍数
    pad_len = (block_size - (input_length % block_size)) % block_size
    
    # 如果需要，手动进行填充
    if pad_len > 0:
        padding_tensor = torch.full(
            (1, pad_len), tokenizer.pad_token_id, dtype=input_ids.dtype, device=input_ids.device
        )
        # 注意：GLA-Dream 的架构似乎是处理整个序列，所以我们把填充放在前面
        # （如果效果不好，可以尝试放在后面 torch.cat([input_ids, padding_tensor], dim=1)）
        padded_input_ids = torch.cat([padding_tensor, input_ids], dim=1)
    else:
        padded_input_ids = input_ids
        
    padded_input_length = padded_input_ids.shape[1]

    # 2. 计算对齐的生成长度 (Align Generation Length)
    max_new_tokens_target = 2048  # 期望生成的最大 token 数
    # 向下取整，确保 gen_length 是 block_size 的倍数
    gen_length = (max_new_tokens_target // block_size) * block_size

    # 3. 计算最终的 max_length
    # 此时，padded_input_length 和 gen_length 都是 block_size 的倍数
    final_max_length = padded_input_length + gen_length

    print(f"[DEBUG] Original length: {input_length}, Padded length: {padded_input_length}, Generation length: {gen_length}, Final max_length: {final_max_length}")

    # 4. 使用计算好的参数调用 generate 函数
    generated_ids = model.generate(
        inputs=padded_input_ids,
        max_length=final_max_length, # 关键：使用计算好的总长度
        threshold=0.9,
        # 移除不再需要的或可能引起冲突的参数
        # tokenizer=tokenizer,
        # max_new_tokens=2048,
    )

    # 5. 解码时，跳过我们手动添加的 padding 和原始输入
    response_start_position = pad_len + input_length
    response = tokenizer.decode(generated_ids[0][response_start_position:], skip_special_tokens=True)
    
    print(f"AI: {response}")
    
    # Add AI response to conversation history
    messages.append({"role": "assistant", "content": response})
    
    print("-" * 50)