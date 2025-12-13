import torch
from fla.layers.kda import KimiDeltaAttention

def test_raw_kda():
    print("🧪 测试纯净 KDA 层...")
    device = "cuda"
    dtype = torch.bfloat16
    
    B, L, H, D = 2, 128, 12, 128 # 模拟你的参数
    hidden_size = H * D
    
    # 实例化单层
    layer = KimiDeltaAttention(
        mode='chunk',
        hidden_size=hidden_size,
        head_dim=D,
        num_heads=H,
        num_v_heads=H, # GQA
        chunk_size=64,
        use_short_conv=True
    ).to(device).to(dtype)
    
    x = torch.randn(B, L, hidden_size, device=device, dtype=dtype).contiguous()
    
    try:
        y, _, _ = layer(x)
        print("✅ KDA 单层正向通过")
        y.sum().backward()
        print("✅ KDA 单层反向通过")
    except Exception as e:
        print(f"❌ KDA 单层崩溃: {e}")
        print("结论：这是 fla 库的问题，不是你代码的问题。")

if __name__ == "__main__":
    test_raw_kda()