#!/usr/bin/env python
# coding=utf-8
import os,sys
from transformers import AutoTokenizer, AutoConfig
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))
from BiDeltaDiff.models import BiDeltaDiffForCausalLM
from .base_model import BaseModel

# modeling_ bideltadiff

class BiDeltaDiffForCausalLM(BaseModel):
    def __init__(self, model_args, *args, **kwargs):
        super().__init__(model_args, *args, **kwargs)
        
        # Load configuration
        self.config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
        
        # Load model
        self.backend_model = BiDeltaDiffForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=self.config,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.backend_model.config.pad_token_id = self.tokenizer.eos_token_id

    def get_model(self):
        return self.backend_model

    def get_tokenizer(self):
        return self.tokenizer
