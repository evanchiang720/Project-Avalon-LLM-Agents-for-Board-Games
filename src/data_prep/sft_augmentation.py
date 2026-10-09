import json
import random
import re
import os
from json import JSONDecoder

INPUT_FILENAME = "gemini_pro.jsonl"
OUTPUT_FILENAME = "qwen_avalon_scoring_train.jsonl"
AUGMENTATION_COUNT = 40

def permute_player_ids(data_obj):
    """資料增強：隨機置換玩家 ID"""
    if not isinstance(data_obj, dict): return None
    original_ids = list(range(6))
    shuffled_ids = list(range(6))
    random.shuffle(shuffled_ids)
    mapping = {old: new for old, new in zip(original_ids, shuffled_ids)}
    json_str = json.dumps(data_obj, ensure_ascii=False)
    def replace_match(match):
        old_id = int(match.group(1))
        new_id = mapping.get(old_id, old_id)
        return f"player-{new_id}"
    new_json_str = re.sub(r'[Pp]layer[-\s](\d)', replace_match, json_str)
    return json.loads(new_json_str)

def format_for_scoring_sft(scenario):
    """格式化為訓練資料"""
    if not isinstance(scenario, dict): return None
    # 寬鬆檢查：只要有 reason 和 score 就可以，或者嘗試修復
    if 'score' not in scenario or 'reason' not in scenario:
        return None

    system_prompt = (
        "You are an expert judge and strategic coach for the board game The Resistance: Avalon (6-player).\n"
        "Your task is to evaluate the quality of a player's move based on the game state.\n"
        "Input: Game State, History, and the Player's Move (Thought, Action, Speech).\n"
        "Output Format: JSON with 'reason' (detailed analysis) and 'score' (0-100 integer)." # <--- 這裡修改了
    )
    
    current_proposal_str = "None"
    if scenario.get('current_proposal'):
        cp = scenario['current_proposal']
        if isinstance(cp, dict):
            current_proposal_str = f"Leader: {cp.get('leader', 'Unknown')}, Team: {cp.get('team', [])}"

    target_move = {
        "thought": scenario.get('thought', ''),
        "action": scenario.get('action', None),
        "speech": scenario.get('speech', '')
    }

    user_content = f"""
Analyze the following Avalon game situation and evaluate the player's chosen move.

Current Game State:
- Round: {scenario.get('round', 'Unknown')}
- Phase: {scenario.get('phase', 'Unknown')}
- Role being evaluated: {scenario.get('role', 'Unknown')} ({scenario.get('identity', 'Unknown')})
- Private Knowledge: {json.dumps(scenario.get('private_knowledge', {}), ensure_ascii=False)}

Game History:
{json.dumps(scenario.get('game_history', []), ensure_ascii=False)}

Recent Dialogue:
{json.dumps(scenario.get('recent_dialogue', []), ensure_ascii=False)}

=== Player's Move to Evaluate ===
{json.dumps(target_move, ensure_ascii=False, indent=2)}

Provide your critique and a score from 0 to 100.  # <--- 這裡也修改了，確保 User Prompt 一致
"""

    assistant_content = json.dumps({
        "reason": scenario.get('reason', ''),
        "score": scenario.get('score', 0)
    }, ensure_ascii=False)

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content.strip()},
            {"role": "assistant", "content": assistant_content}
        ]
    }

def extract_json_objects_robust(content):
    """
    使用 raw_decode 進行串流解析，並智慧去除外層包裝。
    """
    data_list = []
    
    # 1. 前處理：移除 Markdown
    content = re.sub(r'^```(json)?\s*', '', content.strip())
    content = re.sub(r'\s*```$', '', content)
    
    # 2. 嘗試剝離外層 "scenarios": [ ... ] 結構
    start_pos = 0
    
    if content.startswith('{') and '"scenarios"' in content[:100]:
        print(">> Detected 'scenarios' wrapper. Attempting to peel it off...")
        match = re.search(r'"scenarios"\s*:\s*\[', content)
        if match:
            start_pos = match.end()
            print(f">> Found start of list at index {start_pos}")
    elif content.strip().startswith('['):
        start_pos = content.find('[') + 1
        print(">> Detected Top-level List. Starting after '['.")

    # 3. 使用 raw_decode 迴圈解析
    decoder = JSONDecoder()
    pos = start_pos
    total_length = len(content)
    
    while pos < total_length:
        # 跳過空白與逗號 (List 分隔符)
        while pos < total_length and (content[pos].isspace() or content[pos] == ','):
            pos += 1
        
        if pos >= total_length:
            break
        
        # 如果遇到 List 結尾 ']' 或 Object 結尾 '}' (在外層被剝離的情況下)，跳過或結束
        if content[pos] in [']', '}']:
            pos += 1
            continue

        try:
            obj, end_pos = decoder.raw_decode(content, idx=pos)
            
            # 成功解析出一個物件
            if isinstance(obj, dict):
                # 簡單驗證這是否像是一個場景資料
                if 'round' in obj or 'game_history' in obj or 'thought' in obj:
                    data_list.append(obj)
            
            pos = end_pos
        except json.JSONDecodeError:
            # 解析失敗，可能是檔案結尾截斷，或者格式錯亂
            # 嘗試往前找下一個 '{' 重新同步
            next_brace = content.find('{', pos + 1)
            if next_brace != -1:
                pos = next_brace
            else:
                break # 後面沒有 JSON 了
                
    return data_list

def main():
    if not os.path.exists(INPUT_FILENAME):
        print(f"Error: {INPUT_FILENAME} not found.")
        return

    # --- 診斷步驟 ---
    print(f"Reading from {INPUT_FILENAME}...")
    try:
        with open(INPUT_FILENAME, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        print("Error: File encoding is not UTF-8. Trying generic read...")
        with open(INPUT_FILENAME, 'r', errors='ignore') as f:
            content = f.read()

    print(f">> File Size: {len(content)} characters")
    if len(content) == 0:
        print("Error: File is empty!")
        return
    
    print(">> File Head (first 300 chars):")
    print("-" * 40)
    print(content[:300])
    print("-" * 40)
    
    # --- 執行解析 ---
    scenarios = extract_json_objects_robust(content)
    print(f"Successfully extracted {len(scenarios)} valid scenarios.")
    
    if len(scenarios) == 0:
        print("!! Critical Warning: Still found 0 scenarios.")
        print("Please check the 'File Head' above. Does it look like JSON?")
        return

    generated_count = 0
    skipped_count = 0
    
    with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f_out:
        for original_scenario in scenarios:
            # 1. 寫入原始資料
            sft_entry_original = format_for_scoring_sft(original_scenario)
            if sft_entry_original:
                f_out.write(json.dumps(sft_entry_original, ensure_ascii=False) + '\n')
                generated_count += 1
            else:
                skipped_count += 1
                continue

            # 2. 寫入 Augmentation 資料
            for _ in range(AUGMENTATION_COUNT):
                augmented_scenario = permute_player_ids(original_scenario)
                if not augmented_scenario: continue
                sft_entry_aug = format_for_scoring_sft(augmented_scenario)
                if sft_entry_aug:
                    f_out.write(json.dumps(sft_entry_aug, ensure_ascii=False) + '\n')
                    generated_count += 1
    
    print(f"Processing complete.")
    print(f"- Saved to: {OUTPUT_FILENAME}")
    print(f"- Total samples generated: {generated_count}")
    print(f"- Skipped original items (missing fields): {skipped_count}")

if __name__ == "__main__":
    main()