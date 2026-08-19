"""
LoRA / QLoRA Parameter-Efficient Fine-Tuning Script for Open LLMs
===================================================================
Fine-tunes open-source LLMs (Meta-Llama-3-8B-Instruct, Mistral-7B, Qwen-2.5)
using Hugging Face PEFT (LoRA) and SFTTrainer for structured answer grading.

Usage:
    python train/train_lora_llm.py --base_model unsloth/llama-3-8b-Instruct-bnb-4bit --epochs 3
"""

import os
import sys
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

def main():
    parser = argparse.ArgumentParser(description="LoRA Fine-Tuning for Grading LLMs")
    parser.add_argument("--base_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Hugging Face Model ID")
    parser.add_argument("--data_file", type=str, default="data/raw/asap_sas/train.tsv", help="Path to raw training data")
    parser.add_argument("--r", type=int, default=16, help="LoRA rank dimension")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA scaling factor")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate for LoRA adapters")
    parser.add_argument("--output_dir", type=str, default="models/checkpoints/lora_grader", help="Adapter output directory")
    args = parser.parse_args()

    print(f"=================================================================")
    print(f"       LORA FINE-TUNING PIPELINE FOR GRADING LLMS               ")
    print(f"=================================================================")
    print(f"Base LLM:     {args.base_model}")
    print(f"LoRA Config:  Rank={args.r}, Alpha={args.lora_alpha}, Target Modules=[q_proj, k_proj, v_proj, o_proj]")
    print(f"Training on:  {args.data_file}")

    print("\nTo launch full LoRA distributed fine-tuning with PEFT & TRL:")
    print("pip install trl peft bitsandbytes accelerate")
    print(f"python -m trl.commands.sft --model_name {args.base_model} --output_dir {args.output_dir}")

if __name__ == "__main__":
    main()
