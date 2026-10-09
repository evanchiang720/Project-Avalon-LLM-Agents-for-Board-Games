import json
import re
import os
import glob

def parse_system_info(prompt_text):
    """
    解析玩家資訊 (保持不變)
    """
    id_match = re.search(r"You are player-(\d+)", prompt_text, re.IGNORECASE)
    player_id = id_match.group(1) if id_match else "?"

    role_match = re.search(r"role is (\w+)", prompt_text, re.IGNORECASE)
    role = role_match.group(1).capitalize() if role_match else "Unknown"

    knowledge_str = "None"
    if role == "Percival":
        percival_match = re.search(r"either (player-\d+) is merlin and (player-\d+) is morgana", prompt_text, re.IGNORECASE)
        if percival_match:
            c1, c2 = percival_match.group(1), percival_match.group(2)
            candidates = sorted([c1, c2]) 
            knowledge_str = f"Merlin Candidates: {candidates}"
    elif role == "Merlin":
        evil_match = re.search(r"know that (.*?) are on the Evil side", prompt_text, re.IGNORECASE)
        if evil_match:
            evils = re.findall(r"player-\d+", evil_match.group(1))
            knowledge_str = f"Known Evils: {sorted(evils)}"
    elif role in ["Morgana", "Assassin", "Minion", "Mordred", "Oberon"]:
        partners = re.findall(r"partner is (player-\d+)", prompt_text, re.IGNORECASE)
        if not partners:
            all_ids_in_prompt = re.findall(r"player-\d+", prompt_text)
            partners = [p for p in all_ids_in_prompt if p != f"player-{player_id}" and "player-" in p]
            partners = sorted(list(set(partners)))
        if partners:
            knowledge_str = f"Known Evils (Teammates): {partners}"

    return player_id, role, knowledge_str

def parse_game_history(history_text):
    """
    大幅增強：
    1. 準確捕捉 Round 變化
    2. 解析詳細投票紀錄 (YES/NO)
    """
    lines = history_text.split('\n')
    recent_dialogue_lines = []
    
    current_round = 1
    fails = 0
    logic_history = []
    
    # 暫存當前 Proposal 的資訊
    current_leader = "?"
    current_team = []
    
    for line in lines:
        line = line.strip()
        if not line: continue
        
        # 收集 Recent Dialogue
        recent_dialogue_lines.append(line)
        if len(recent_dialogue_lines) > 10:
            recent_dialogue_lines.pop(0)

        line_lower = line.lower()

        # --- 1. 修正 Round 偵測邏輯 ---
        # 你的資料有兩種格式: "quest 2 has begun" 或 "quest 1 requires 2 players"
        # 我們優先抓 "has begun" 因為它最準確表示新的一局
        round_match = re.search(r"quest (\d+) has begun", line_lower)
        if round_match:
            new_round = int(round_match.group(1))
            if new_round > current_round:
                current_round = new_round
        else:
            # Fallback: 如果只有 requires
            round_match_2 = re.search(r"quest (\d+) requires", line_lower)
            if round_match_2:
                r = int(round_match_2.group(1))
                if r > current_round:
                    current_round = r

        # --- 2. 捕捉隊長與提議 ---
        # system: player-3 proposed a party: player-3, player-0
        if "proposed a party" in line_lower:
            parts = line.split(":")
            # 抓隊長 (前面的 system: player-3 ...)
            leader_match = re.search(r"(player-\d+) proposed", line, re.IGNORECASE)
            if leader_match:
                current_leader = leader_match.group(1)
            
            # 抓隊伍
            members = re.findall(r"player-\d+", parts[-1], re.IGNORECASE)
            current_team = members

        # --- 3. 解析詳細投票 (新增功能) ---
        # system: party vote outcome: player-0: yes, player-1: yes...
        if "party vote outcome" in line_lower:
            # 解析每個人的票
            # 格式: player-0: yes
            votes = re.findall(r"(player-\d+):\s*(yes|no)", line, re.IGNORECASE)
            
            yes_votes = []
            no_votes = []
            
            for pid, vote in votes:
                if vote.lower() == 'yes':
                    yes_votes.append(pid.replace('player-', '')) # 只存數字比較簡潔
                else:
                    no_votes.append(pid.replace('player-', ''))
            
            # 判斷結果 (通常下一行會有 vote succeeded 或 failed，或是根據票數判斷)
            # 這裡簡單判定：如果 system 說 vote succeeded 或 failed
            # 為了簡單，我們先存下來，等下一行 system 訊息確認結果，或者直接顯示票數
            
            # 建立這一輪的詳細紀錄字串
            outcome_str = f"Round {current_round} Proposal:\n"
            outcome_str += f"- Leader: {current_leader}\n"
            outcome_str += f"- Team: {current_team}\n"
            outcome_str += f"- Votes (YES): {yes_votes}\n"
            outcome_str += f"- Votes (NO):  {no_votes}"
            
            logic_history.append(outcome_str)

        # --- 4. 捕捉結果 (Approved/Rejected/Mission Fail) ---
        if "vote succeeded" in line_lower:
            # 這是指組隊投票通過
            if logic_history: logic_history[-1] += "\n- Result: APPROVED"
            
        elif "vote failed" in line_lower:
            # 這是指組隊投票失敗
            if logic_history: logic_history[-1] += "\n- Result: REJECTED"

        # --- 5. 捕捉任務成敗 (Quest Result) ---
        if "quest succeeded" in line_lower or "mission succeeded" in line_lower:
            logic_history.append(f"Round {current_round} Mission: PASSED")
            
        elif "quest failed" in line_lower or "mission failed" in line_lower:
            # 避免重複 (有些 log 會重複寫)
            if not (logic_history and "Mission: FAILED" in logic_history[-1]):
                fails += 1
                logic_history.append(f"Round {current_round} Mission: FAILED")

    formatted_history = "\n\n".join(logic_history) if logic_history else "Round 1 just started. No history yet."
    formatted_recent = "\n".join(recent_dialogue_lines)
    
    return current_round, fails, formatted_history, formatted_recent

def extract_thought_and_speech(chosen_text):
    """
    分離 <thinking> 和 <speech>，並移除 speech 開頭的 'player-X:'
    """
    thought = ""
    speech = ""
    
    if "'Your speech'" in chosen_text:
        parts = chosen_text.split("'Your speech'")
        
        raw_thought = parts[0]
        thought = raw_thought.replace("'Your Thought'", "").strip()
        
        raw_speech = parts[1].strip()
        speech = re.sub(r"^player-\d+:\s*", "", raw_speech, flags=re.IGNORECASE)
        
    else:
        speech = re.sub(r"^player-\d+:\s*", "", chosen_text, flags=re.IGNORECASE)

    return thought, speech

def process_files(input_folder, output_file):
    json_files = glob.glob(os.path.join(input_folder, "*.json"))
    print(f"Found {len(json_files)} JSON files in '{input_folder}'")

    all_sft_data = []

    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
                if isinstance(raw_data, dict): raw_data = [raw_data]

                for item in raw_data:
                    prompt = item.get('prompt', '')
                    chosen = item.get('chosen', '')
                    if not prompt or not chosen: continue

                    # 1. System
                    pid, role, private_knowledge = parse_system_info(prompt)
                    
                    # 2. History
                    if "'Dialogue history'" in prompt:
                        history_part = prompt.split("'Dialogue history'")[1]
                    else:
                        history_part = prompt
                    
                    curr_round, fails, logic_hist, recent_diag = parse_game_history(history_part)
                    
                    # 3. Output
                    thought, speech = extract_thought_and_speech(chosen)

                    # 4. Assemble
                    input_text = f"""[System]
Role: {role} (Player-{pid})
Private Knowledge: {private_knowledge}
Current State: Round {curr_round}
Fails: {fails} (Need 3 to lose)

[Game Logic History]
{logic_hist}

[Recent Dialogue]
{recent_diag}"""

                    output_text = f"""<thinking>
{thought}
</thinking>

<speech>
{speech}
</speech>"""

                    all_sft_data.append({
                        "instruction": "You are playing a game of Avalon. Analyze the current situation and speak.",
                        "input": input_text,
                        "output": output_text
                    })

        except Exception as e:
            print(f"Skipping {file_path}: {e}")

    print(f"Total processed samples: {len(all_sft_data)}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in all_sft_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    # 請修改你的路徑
    input_folder = "discuss"
    output_filename = "avalon_sft_dataset_fixed.jsonl"
    
    if not os.path.exists(input_folder):
        os.makedirs(input_folder, exist_ok=True)
    
    process_files(input_folder, output_filename)