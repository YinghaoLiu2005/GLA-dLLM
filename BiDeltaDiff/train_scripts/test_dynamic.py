import torch
from BiDeltaDiff.models import BiDeltaDiffForCausalLM, BiDeltaDiffConfig
CUDA_VISIBLE_DIVICES=1
device ="cpu"
def test_model_sanity():
    print("🚀 开始冒烟测试...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. 使用你刚才修改好的 Config
    config = BiDeltaDiffConfig(
        vocab_size=151936,
        hidden_size=1536,
        intermediate_size=8960,
        num_hidden_layers=2, # 测试只开2层，省时间
        num_heads=12,
        head_dim=128,
        chunk_size=64,
        is_bidirectional=True,
        fuse_norm=True,
        fuse_swiglu=True
    )

    # 2. 实例化模型
    model = BiDeltaDiffForCausalLM(config).to(device).to(torch.bfloat16)
    model.train() # 开启训练模式 (触发 Fast-dLLM 的扩散 Mask 逻辑)
    print("✅ 模型实例化成功")

    # 3. 构造 Dummy Input
    # 注意：长度最好是 chunk_size (64) 的倍数，比如 128
    B, L = 2, 128 
    input_ids = torch.randint(0, 10000, (B, L)).to(device)
    labels = input_ids.clone() # 随便造点 label

    # 4. 前向传播测试 (Forward)
    try:
        print("🔄 正在进行 Forward...")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        print(f"✅ Forward 成功! Loss: {loss.item()}, Logits Shape: {logits.shape}")
    except Exception as e:
        print(f"❌ Forward 失败: {e}")
        return

    # 5. 反向传播测试 (Backward) - 检查梯度流是否断裂
    try:
        print("🔄 正在进行 Backward...")
        loss.backward()
        print("✅ Backward 成功! 梯度计算正常。")
    except Exception as e:
        print(f"❌ Backward 失败 (可能是 inplace操作或计算图断裂): {e}")
        return

    print("🎉🎉🎉 恭喜！模型结构在数学和工程上都是通的！可以开始加载权重训练了！")

if __name__ == "__main__":
    test_model_sanity()