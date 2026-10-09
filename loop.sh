#!/bin/bash

# 無限迴圈
while true
do
    echo "---------------------------------------"
    echo "正在啟動 run_avalon.py ..."
    echo "時間: $(date)"
    echo "---------------------------------------"

    # 執行 Python 程式
    # 請確保 python3 是你正確的執行指令，或是使用虛擬環境的路徑
    python3 run_avalon.py

    # 當上面的 python 程式結束（無論是正常結束還是報錯 Crash），
    # 程式碼會繼續往下執行到這裡
    
    echo "程式已停止運行。"
    echo "將在 1 秒後重新啟動..."
    
    # 暫停 1 秒，避免如果程式瞬間崩潰導致 CPU 被無限迴圈佔滿
    sleep 1
done