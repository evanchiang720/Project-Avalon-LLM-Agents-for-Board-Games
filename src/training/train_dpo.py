"""
Avalon DPO (Direct Preference Optimization) Training Script.
Optimizes the Qwen-4B SFT model using preference pairs generated from game simulations.
"""

import os
import json
import argparse
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import DPOTrainer, DPOConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train DPO policy for Avalon LLM")
    parser.add_argument("--sft_model_path", type=str, default="./qwen_4b_sft", 
                        help="Path or HuggingFace ID of the base SFT model")
    parser.add_argument("--dataset_path", type=str, default="data/avalon_dpo_final.jsonl", 
                        help="Path to the DPO dataset (.jsonl)")
    parser.add_argument("--output_dir", type=str, default="./qwen_4b_dpo_final", 
                        help="Directory to save the DPO checkpoint")
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="DPO learning rate")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=128, help="LoRA alpha")
    return parser.parse_args()


def load_and_sanitize_data(dataset_path):
    """Loads JSONL dataset and ensures chosen/rejected entries are valid strings."""
    print(f">>> Loading DPO dataset from: {dataset_path}")
    data_list = []
    skipped_count = 0

    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                # Ensure chosen and rejected are strings (TRL requirement)
                for key in ["chosen", "rejected"]:
                    if isinstance(obj.get(key), dict):
                        obj[key] = json.dumps(obj[key], ensure_ascii=False)
                    elif not isinstance(obj.get(key), str):
                        obj[key] = str(obj.get(key, ""))

                data_list.append(obj)
            except json.JSONDecodeError:
                skipped_count += 1
                continue

    print(f" Loaded {len(data_list)} valid samples (Skipped {skipped_count} malformed lines).")
    return Dataset.from_list(data_list)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Dataset
    raw_dataset = load_and_sanitize_data(args.dataset_path)

    # 2. Tokenizer Setup
    print(f">>> Loading Tokenizer from: {args.sft_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        if "<|endoftext|>" in tokenizer.get_vocab():
            tokenizer.pad_token = "<|endoftext|>"
        else:
            tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = "left"  # DPO prefers left padding for autoregressive evaluation

    # 3. Model Setup (bfloat16 + FlashAttention-2 fallback)
    print(f">>> Loading Base SFT Model from: {args.sft_model_path}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.sft_model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            trust_remote_code=True,
            use_cache=False,
        )
        print(" Using FlashAttention-2 for high throughput.")
    except Exception as e:
        print(f"⚠️ FlashAttention-2 not available ({e}). Falling back to SDPA.")
        model = AutoModelForCausalLM.from_pretrained(
            args.sft_model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="auto",
            trust_remote_code=True,
            use_cache=False,
        )

    model.gradient_checkpointing_enable()

    # 4. Format Dataset with Chat Template
    def format_chat_prompt(batch):
        formatted_prompts = []
        for sys_msg, user_msg in zip(batch["system"], batch["user"]):
            messages = [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            formatted_prompts.append(prompt)

        return {
            "prompt": formatted_prompts,
            "chosen": batch["chosen"],
            "rejected": batch["rejected"],
        }

    print(">>> Formatting dataset with Qwen Chat Template...")
    dataset = raw_dataset.map(format_chat_prompt, batched=True)

    # 5. LoRA Configuration
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # 6. DPO Training Arguments
    training_args = DPOConfig(
        output_dir=args.output_dir,
        beta=0.1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        max_prompt_length=2048,
        max_length=3072,
        bf16=True,
        fp16=False,
        logging_steps=5,
        save_strategy="steps",
        save_steps=20,
        save_total_limit=10,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        report_to="tensorboard",
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )

    # 7. Trainer Initialization
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Automatically managed via PEFT adapter disabling
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    # 8. Execution & Export
    print(">>> Starting DPO Policy Optimization...")
    trainer.train()

    print(f">>> Saving final model adapter to: {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("✅ DPO Training Finished Successfully!")


if __name__ == "__main__":
    main()