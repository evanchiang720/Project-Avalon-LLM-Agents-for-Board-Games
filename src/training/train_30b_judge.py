import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# ================= 設定區 =================
MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
DATA_FILE = "qwen_avalon_scoring_train.jsonl"
OUTPUT_DIR = "./qwen_30b_scoring_checkpoints"
MAX_SEQ_LENGTH = 4096  # 30B 建議 2k，避免 OOM

# ================= 載入資料 =================
print(">>> Loading dataset...")
dataset = load_dataset("json", data_files=DATA_FILE, split="train")

# ================= 載入模型 =================
print(">>> Loading model and tokenizer...")

# 4-bit 量化設定
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,  # 計算精度
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
    torch_dtype=torch.bfloat16, # 強制權重載入為 bf16 (除了量化層)
    attn_implementation="sdpa",
)

# 基本設定
model.config.use_cache = False
model.config.pretraining_tp = 1

# 準備模型進行量化訓練
# 這步會自動將某些層（如 lm_head）轉為 float32 以求穩定，但會導致 Dtype 衝突
print(">>> Preparing model for training...")
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# ================= LoRA 設定 =================
print(">>> Applying LoRA...")
peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    # 針對 Qwen3 MoE 架構建議的層
    target_modules=["q_proj", "v_proj", "o_proj", "k_proj"], 
    init_lora_weights=True,
)

model = get_peft_model(model, peft_config)

# ================= 【核心修正：解決 Dtype 衝突】 =================
# 因為計算精度是 bfloat16，但 lm_head 會被 prepare_model_for_kbit_training 轉成 float32
# 我們必須手動將其轉回 bfloat16，否則矩陣相乘時會報錯
print(">>> Fixing dtype mismatch for lm_head and embed_tokens...")
for name, module in model.named_modules():
    if "lm_head" in name or "embed_tokens" in name:
        module.to(torch.bfloat16)

# 確保所有可訓練參數（LoRA 層）也是 bfloat16
for name, param in model.named_parameters():
    if param.requires_grad:
        param.data = param.data.to(torch.bfloat16)
# ==============================================================

model.print_trainable_parameters()

# ================= 訓練參數 =================
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    max_length=MAX_SEQ_LENGTH,
    packing=False,
    
    # 訓練超參數
    num_train_epochs=3,
    per_device_train_batch_size=4,  # 30B 必須為 1
    gradient_accumulation_steps=16,
    learning_rate=1e-5,
    
    # 精度設定 (關鍵：必須與 compute_dtype 一致)
    fp16=False,
    bf16=True, 
    optim="paged_adamw_8bit",
    
    # Checkpoint 與 Logging
    logging_steps=5,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=10,
    
    # 學習率排程
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    report_to="none",
    
    # 梯度檢查點
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": True},
    
    # 分佈式與效能
    ddp_find_unused_parameters=False,
)

# ================= 訓練 =================
print(">>> Starting training...")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer,
    args=training_args,
    peft_config=peft_config,
)

# 開始訓練
trainer.train()

# ================= 儲存 =================
print(">>> Saving model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ Training complete! Model saved to: {OUTPUT_DIR}")