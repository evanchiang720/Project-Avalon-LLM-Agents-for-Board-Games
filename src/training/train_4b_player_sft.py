import torch
import json
import os
import glob
import re
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training  # 關鍵修正：引入這個函數
)
from trl import SFTTrainer, SFTConfig

# ================= 設定區 =================
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
DATA_DIR = "preference_data" 
OUTPUT_DIR = "./avalon_llama_sft_checkpoints"

# 訓練參數
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 2
MAX_LENGTH = 4096

# ================= System Prompt =================
LITE_SYSTEM_PROMPT = (
    "[System] You are an expert AI playing the social deduction game \"The Resistance: Avalon\".\n"
    "[Rules] Good Team (Merlin, Percival, Servant) wants Quest Success. Evil Team (Morgana, Assassin) wants Quest Fail.\n"
    "[Task] Be strategic, logical, and deceptive if necessary.\n\n"
)

# ================= 資料處理函數 =================
def clean_vote_response(text):
    text = re.sub(r"'Your \w+'\s*\n", "", text)
    text = re.sub(r"^player-\d+:\s*", "", text, flags=re.MULTILINE)
    lower_text = text.strip().lower()
    if lower_text in ['yes', 'no', 'yes.', 'no.']:
        return lower_text.replace('.', '')
    return text.strip()

def clean_dialogue_response(text):
    pattern = r"('Your speech'\s*\n)player-\d+:\s*"
    return re.sub(pattern, r"\1", text, flags=re.IGNORECASE)

def load_dataset_from_folder(folder_path):
    all_data = []
    file_paths = glob.glob(os.path.join(folder_path, "*.json"))
    
    print(f"找到 {len(file_paths)} 個資料檔案")
    
    for file_path in file_paths:
        try:
            filename = os.path.basename(file_path)
            is_vote_file = "_team_vote" in filename
            is_human = not filename[0].isdigit()

            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
                if isinstance(content, dict):
                    content = [content]
                
                valid_entries = []
                for entry in content:
                    if 'prompt' in entry and 'chosen' in entry:
                        prompt_text = LITE_SYSTEM_PROMPT + entry['prompt']
                        
                        original_chosen = entry['chosen']
                        if is_vote_file:
                            chosen_text = clean_vote_response(original_chosen)
                        else:
                            chosen_text = clean_dialogue_response(original_chosen)

                        # 構建 Chat 格式
                        messages = [
                            {"role": "user", "content": prompt_text},
                            {"role": "assistant", "content": chosen_text}
                        ]

                        valid_entries.append({
                            "messages": messages,
                            "source": "human" if is_human else "gemini",
                            "type": "vote" if is_vote_file else "dialogue"
                        })

                # 真人對話加權（複製3次）
                if is_human and not is_vote_file:
                    for _ in range(3): 
                        all_data.extend(valid_entries)
                else:
                    all_data.extend(valid_entries)
                    
        except Exception as e:
            print(f"讀取 {file_path} 錯誤: {e}")

    print(f"總訓練樣本數: {len(all_data)}")
    return Dataset.from_list(all_data)

# ================= 1. 載入資料 =================
print("正在載入資料集...")
dataset = load_dataset_from_folder(DATA_DIR)

# ================= 2. 載入模型與 Tokenizer =================
print(f"正在載入模型: {MODEL_NAME}...")

# 4-bit 量化設定
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 載入模型
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

# 基本模型設定
model.config.use_cache = False
model.config.pretraining_tp = 1

# [關鍵修正] 使用 PEFT 官方函數準備模型
# 這會自動處理 gradient checkpointing 和量化層的梯度問題
print("正在準備模型進行 k-bit 訓練...")
model = prepare_model_for_kbit_training(model)

# ================= 3. LoRA 設定 =================
print("套用 LoRA 設定...")
peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# ================= 4. 訓練參數 =================
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    
    # SFTConfig 參數
    max_length=MAX_LENGTH,
    packing=False,
    
    # 訓練超參數
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    
    # 精度與優化器
    fp16=False,
    bf16=True,
    optim="paged_adamw_32bit",
    
    # Checkpoints 與 Logging
    logging_steps=10,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=20,
    
    # 排程器
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    report_to="tensorboard",
    
    # [關鍵修正] 梯度檢查點設定
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False}, # 解決警告
    ddp_find_unused_parameters=False,
)

# ================= 5. 初始化 Trainer =================
# TRL 的 SFTTrainer 會自動處理 "messages" 格式的 dataset
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer,
    args=training_args,
    peft_config=peft_config, # 也可以在這裡傳入，雙重保險
)

# ================= 6. 開始訓練 =================
print("=" * 50)
print("開始訓練...")
print("請觀察 log 中的 'grad_norm'，它現在應該大於 0.0 了")
print("=" * 50)

trainer.train()

# ================= 7. 儲存模型 =================
print("訓練完成！儲存模型...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"模型已儲存至: {OUTPUT_DIR}")
