
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from kernel.models.gated_deltanet.configuration_gated_deltanet import GatedDeltaNetConfig
from kernel.models.gated_deltanet.modeling_gated_deltanet import GatedDeltaNetForCausalLM, GatedDeltaNetModel

AutoConfig.register(GatedDeltaNetConfig.model_type, GatedDeltaNetConfig, exist_ok=True)
AutoModel.register(GatedDeltaNetConfig, GatedDeltaNetModel, exist_ok=True)
AutoModelForCausalLM.register(GatedDeltaNetConfig, GatedDeltaNetForCausalLM, exist_ok=True)

__all__ = ['GatedDeltaNetConfig', 'GatedDeltaNetForCausalLM', 'GatedDeltaNetModel']
