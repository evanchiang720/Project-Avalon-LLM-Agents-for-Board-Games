import json
import random

# 定義輸入和輸出檔案的名稱
input_filename = 'avalon_sft_dataset_augmented.jsonl'
output_filename = 'random_sample_100.jsonl'
# 定義要抽取的樣本數量
sample_size = 100

# 讀取所有的資料行
try:
    with open(input_filename, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
except FileNotFoundError:
    print(f"錯誤：找不到檔案 '{input_filename}'。請確認檔案名稱和路徑是否正確。")
    exit()

# 確認資料筆數是否足夠
if len(all_lines) < sample_size:
    print(f"警告：資料總筆數 ({len(all_lines)}) 少于要抽取的數量 ({sample_size})。")
    print("將會使用所有可用的資料。")
    sample_size = len(all_lines)

# 隨機抽取指定數量的資料
random_sample_lines = random.sample(all_lines, sample_size)

# 將抽取的資料寫入新的檔案
with open(output_filename, 'w', encoding='utf-8') as f:
    for line in random_sample_lines:
        # 使用 strip() 來移除可能存在於行尾的空白或換行符，然後再手動加上換行
        # 這樣可以確保檔案格式的整潔
        f.write(line.strip() + '\n')

print(f"成功！已從 '{input_filename}' 中隨機抽取 {sample_size} 筆資料，並存儲至 '{output_filename}'。")