#!/usr/bin/env python3
"""
Script to initialize GLA Dream model from Qwen3
"""

import argparse
from gla_dream.modeling_gla_dream import GLADreamModel
from gla_dream.configuration_gla_dream import GLADreamConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen3_path", type=str, required=True, help="Qwen3 model path")
    parser.add_argument("--output_path", type=str, required=True, help="GLA Dream model output path")
    parser.add_argument("--gla_config", type=str, help="GLA Dream config file path")
    
    args = parser.parse_args()
    
    # Load GLA Dream config
    gla_config = GLADreamConfig.from_pretrained(args.gla_config) if args.gla_config else GLADreamConfig()
    
    # Initialize GLA Dream model from Qwen3
    gla_model = GLADreamModel.from_qwen3_pretrained(
        qwen3_model_path=args.qwen3_path,
        gla_config=gla_config
    )
    
    # Save model
    gla_model.save_pretrained(args.output_path)
    print(f"GLA Dream model has been saved to: {args.output_path}")

if __name__ == "__main__":
    main()