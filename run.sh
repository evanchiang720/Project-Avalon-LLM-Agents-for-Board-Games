#!/bin/bash

# 設定使用的 GPU ID (如果你只有 4 張卡，通常是 0,1,2,3)
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 設定主要通訊埠 (防止跟其他人衝突)
MASTER_PORT=29500

# 顯示當前設定
echo "Starting training on 4 GPUs..."
echo "Config: DeepSpeed ZeRO-2"
echo "Model: Qwen3-4B"

# 啟動指令
# --nproc_per_node=4 : 表示使用當前節點的 4 張卡
# --master_port : 指定通訊埠
torchrun --nproc_per_node=4 --master_port=$MASTER_PORT train.py

# 如果你想保留訓練 Log 到檔案，可以用下面這行代替上面那行：
# torchrun --nproc_per_node=4 --master_port=$MASTER_PORT train.py > training.log 2>&1 &