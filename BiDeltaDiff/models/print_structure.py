import sys
import os

# 1. 获取当前脚本所在目录的"上上级"目录（即项目根目录 /data/yinghaoliu/GLA-dLLM/）
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))

# 2. 将项目根目录加入 Python 搜索路径
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 3. 使用完整的包路径进行导入 (Absolute Import)
# 这样 modeling.py 就会知道自己属于 BiDeltaDiff.models 包，从而使内部的 relative import 生效
from BiDeltaDiff.models.configuration import BiDeltaDiffConfig
from BiDeltaDiff.models.modeling import BiDeltaDiffForCausalLM

# 4. 初始化并打印
config = BiDeltaDiffConfig()
model = BiDeltaDiffForCausalLM(config)
print(model)