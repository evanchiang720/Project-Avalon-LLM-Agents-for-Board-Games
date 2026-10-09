import json
import random
import re
import sys

# 設定檔案路徑
INPUT_FILE = 'avalon_sft_dataset_fixed.jsonl'
OUTPUT_FILE = 'avalon_sft_dataset_augmented.jsonl'

# 設定玩家數量 (0 ~ 5)
MAX_PLAYER_ID = 5

# === 1. 前綴保護關鍵字 (Prefix Protection) ===
# 格式：關鍵字 (+ 冒號/連字號) + 數字
# 這些詞後面的數字「不是」玩家 ID
PROTECTED_PREFIXES = [
    r'Round', r'Quest', r'Mission', 
    r'Fails?', r'Failure',            # 失敗
    r'Succeeds?', r'Success(?:es)?',  # 成功 (修正你的問題)
    r'Need', r'Votes?', 
    r'Reject', r'Approve', 
    r'Turn', r'Stage', r'Size',
    r'Proposal', r'Team',             # 提案/隊伍編號通常不改
    r'Result', r'Score'
]

# === 2. 後綴保護關鍵字 (Suffix Protection) ===
# 格式：數字 + 關鍵字
# 這些詞前面的數字通常是「數量」而非玩家 ID
PROTECTED_SUFFIXES = [
    r'players?', r'people', r'persons?',
    r'good', r'evil', r'bad', 
    r'minions?', r'spies', r'resistance',
    r'fails?', r'failures?',          # 例如: "2 fails"
    r'succeeds?', r'success(?:es)?'   # 例如: "3 succeeds"
]

def get_random_mapping(max_id):
    """產生一個 0~max_id 的隨機對應表"""
    original = list(range(max_id + 1))
    shuffled = original[:]
    random.shuffle(shuffled)
    return {str(o): str(s) for o, s in zip(original, shuffled)}

def replace_numbers(text, mapping):
    """
    使用 Regex 替換文字中的玩家編號，同時避開各種保護規則。
    """
    
    # === 建構 Regex ===
    
    # Group A: 前綴保護 (Succeed: 1, Round 1)
    pattern_prefix_prot = r'\b(?:' + '|'.join(PROTECTED_PREFIXES) + r')\s*[:\-]?\s*\d+'
    
    # Group B: 後綴保護 (5 players, 2 fails)
    pattern_suffix_prot = r'\d+\s+(?:' + '|'.join(PROTECTED_SUFFIXES) + r')\b'
    
    # Group C: 連字號/結構保護 (1-1, 1-person, 1-Reject)
    pattern_hyphen_prot = r'\b\d+(?:-[a-zA-Z0-9]+)+\b'
    
    # Group D: 目標玩家編號 (要替換的)
    pattern_target = r'\b((?:player|p)\s*[-#]?\s*)?(\d+)\b'

    # 組合 Regex (忽略大小寫)
    full_pattern = re.compile(
        f"({pattern_prefix_prot})|({pattern_suffix_prot})|({pattern_hyphen_prot})|{pattern_target}",
        flags=re.IGNORECASE
    )

    def callback(match):
        # 如果是保護群組 (Group 1, 2, 3)，直接原樣返回
        if match.group(1): return match.group(1)
        if match.group(2): return match.group(2)
        if match.group(3): return match.group(3)
            
        # 否則替換玩家編號
        prefix = match.group(4) if match.group(4) else ""
        number = match.group(5)
        
        if number in mapping:
            return f"{prefix}{mapping[number]}"
        else:
            return match.group(0)

    return full_pattern.sub(callback, text)

def process_dataset():
    print(f"正在處理 {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as fin, \
             open(OUTPUT_FILE, 'w', encoding='utf-8') as fout:
            
            count = 0
            for line in fin:
                if not line.strip(): continue
                data = json.loads(line)
                
                # 資料增強倍數
                AUGMENT_FACTOR = 1
                
                for _ in range(AUGMENT_FACTOR):
                    mapping = get_random_mapping(MAX_PLAYER_ID)
                    
                    new_data = {}
                    new_data['instruction'] = replace_numbers(data.get('instruction', ''), mapping)
                    new_data['input'] = replace_numbers(data.get('input', ''), mapping)
                    new_data['output'] = replace_numbers(data.get('output', ''), mapping)
                    
                    fout.write(json.dumps(new_data, ensure_ascii=False) + '\n')
                
                count += 1
                if count % 100 == 0:
                    print(f"已處理 {count} 筆原始資料...")
                    
        print(f"完成！新檔案已儲存為 {OUTPUT_FILE}")
        
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 {INPUT_FILE}")

if __name__ == "__main__":
    # === 測試區 ===
    # 建立 0->1, 1->2 ... 5->0 的映射
    test_mapping = {str(i): str((i+1) % (MAX_PLAYER_ID + 1)) for i in range(MAX_PLAYER_ID + 1)}
    
    print(f"=== 測試 Regex (重點檢查 succeed) ===")
    test_cases = [
        "fail: 1",             # 預期: 不變
        "fails: 1",            # 預期: 不變
        "succeed: 1",          # 預期: 不變 (這次應該要成功了)
        "Success: 2",          # 預期: 不變
        "Result: 3 fails",     # 預期: 不變 (Result保護3, fails保護3)
        "Proposal 1 failed",   # 預期: 不變 (Proposal保護1)
        "Player 1 succeeded",  # 預期: Player 2 succeeded (這是玩家，要改)
        "P0 fails mission",    # 預期: P1 fails mission
    ]
    
    for t in test_cases:
        print(f"原句: {t:<20} -> 新句: {replace_numbers(t, test_mapping)}")
        
    print("\n開始處理檔案...")
    process_dataset()