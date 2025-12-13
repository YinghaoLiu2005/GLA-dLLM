from configuration import BiDeltaDiffConfig
from modeling import BiDeltaDiffForCausalLM

config = BiDeltaDiffConfig()
model = BiDeltaDiffForCausalLM(config)
print(model)