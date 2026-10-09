import json
import random
import requests
import re
import time
import os
from typing import List, Dict, Any

# 建議安裝這些庫: pip install sentence-transformers scikit-learn requests
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

# ==========================================
# 1. 基礎設定與參數
# ==========================================

# 模型 API 設定 (請根據你的實際環境修改)
# 建議: PLAYER 使用較快的小模型 (如 Qwen-4B/7B), SCORING 使用邏輯強的大模型 (如 Qwen-32B/72B)
PLAYER_URL = "http://localhost:8001/v1/chat/completions"
SCORING_URL = "http://localhost:8000/v1/chat/completions"
PLAYER_MODEL = "./qwen_4b_sft"
SCORING_MODEL = "./qwen_30b_scoring"

# 輸出檔案
OUTPUT_FILE = "avalon_dpo_dataset_v4_strict.jsonl"

# Embedding 模型 (用於候選回答的多樣性篩選)
print("Loading Embedding Model...")
try:
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
except Exception as e:
    print(f"Warning: Failed to load embedding model. Diversity filtering will be disabled. ({e})")
    embed_model = None

# 人格 Prompt (用於生成不同風格的回答)
persona_prompts = [
    "Adopt a hostile and confrontational tone. Attack the logic of previous speakers.",
    "Be skeptical and distrusting. Demand that others prove their loyalty.",
    "Maintain a cold, calculating tone. Focus purely on voting patterns and factual history.",
    "Focus entirely on logical consistency. Point out contradictions in previous turns.",
    "Adopt a vulnerable, emotional, and pleading tone. Ask for the group's trust.",
    "Emphasize the importance of 'Unity' and 'Trust'. Try to de-escalate conflicts.",
    "Project absolute confidence and authority. Speak as the de facto leader.",
    "Be impatient and decisive. Urge the group to vote immediately to avoid stalling.",
    "Express hesitation and uncertainty. Use words like 'maybe', 'might', and 'unsure'.",
    "State that you are relying on your 'gut feeling' or 'vibes' rather than complex data.",
    "Be extremely concise. Keep it under 15 words. Be blunt.",
    "Use a storytelling tone. Narrate the current state of the kingdom/mission."
]

# 玩家 System Prompt (強調事實查核)
SYSTEM_PROMPT = (
    "You are a strategic AI playing The Resistance: Avalon (6-player setup).\n"
    "Your Goal: Win by completing 3 Missions (Good) or causing 3 Fails/Assassinating Merlin (Evil).\n\n"
    "*** GAME RULES (6-Player) ***\n"
    "1. Roles: Good (Merlin, Percival, 2 Servants) vs Evil (Morgana, Assassin).\n"
    "2. Mission Team Sizes: [2, 3, 4, 3, 4].\n"
    "3. Voting: 5 rejected proposals in a single round = Evil wins immediately.\n"
    "4. Knowledge:\n"
    "   - Evil (Morgana, Assassin) know each other.\n"
    "   - Merlin knows both Evil players.\n"
    "   - Percival sees Merlin & Morgana but does NOT know which is which.\n"
    "   - Servants know nothing.\n\n"
    "*** STRATEGIC GUIDELINES ***\n"
    "1. AVOID THE 5TH PROPOSAL: Reaching the 5th proposal (Hammer) is dangerous for Good. It forces a blind 'Approve'.\n"
    "2. OPTIMAL TIMING: Good players should try to approve a clean team on Proposal 2 or 3.\n"
    "3. HAMMER LOGIC: If you are on Proposal 5, you MUST Approve (unless you are Evil and want to win by default).\n\n"
    "*** ABSOLUTE PROHIBITIONS ***\n"
    "1. NEVER say 'I am Evil', 'I am Morgana', or 'I am Assassin'. Pretend to be Good.\n"
    "2. NEVER act as the Leader if it is not your turn.\n\n"
    "*** RESPONSE FORMAT ***\n"
    "JSON with 'thought' (internal strategy), 'action', and 'speech' (public persona)."
)

# 評分 System Prompt (嚴格邏輯裁判)
SCORING_SYSTEM_PROMPT = (
    "You are a harsh and critical strategy referee for Avalon.\n"
    "Your job is to rank player moves based on strategic depth. Most moves are AVERAGE.\n\n"
    "*** SCORING RUBRIC (Normal Distribution) ***\n"
    "0: Hallucination / Fact Error / Game Breaking (e.g., self-doxxing).\n"
    "40-59 (FAIL/POOR): Generic filler text (e.g., 'I agree', 'Need more info', 'Let's see'). No new information added. Safe but useless.\n"
    "60-75 (PASS/AVERAGE): Logical but standard moves. Reacts correctly to the board state but lacks creativity.\n"
    "76-89 (GOOD): Strong reasoning. Identifies specific logical traps, points out contradictions in voting history, or uses role-specific strategies effectively.\n"
    "90-100 (EXCELLENT): Game-winning insight. Manipulates other players perfectly, or catches a subtle logical error that others missed.\n\n"
    "*** INSTRUCTIONS ***\n"
    "1. START at a score of 60.\n"
    "2. DEDUCT points for: Vague statements, repeating what others said, being too passive.\n"
    "3. ADD points for: Specific references to voting history, clever deception, identifying valid paradoxes.\n"
    "4. BE STINGY. Do not give 85+ easily. Most legitimate moves should be between 60 and 75.\n\n"
    "Output JSON: {\"ground_truth_verification\": {\"is_factually_consistent\": bool, \"error_details\": str}, \"strategic_analysis\": {...}, \"reasoning\": str, \"score\": int}"
)

# ==========================================
# 2. Python 硬邏輯過濾器 (Hard Filters)
# ==========================================

def validate_history_consistency(game_state_prompt, response_json):
    """檢查玩家是否產生與歷史結果相反的幻覺"""
    thought = (response_json.get('thought') or '').lower()
    speech = (response_json.get('speech') or '').lower()
    full_text = thought + " " + speech
    
    try:
        if "Game History (Fact):" in game_state_prompt:
            history_section = game_state_prompt.split("Game History (Fact):")[1].split("Current Proposal")[0]
        else:
            return True # 無法解析歷史，跳過此檢查
    except IndexError:
        return True

    # 檢查 Round 結果的一致性
    if "Mission: SUCCESS" in history_section:
        if "mission failed" in full_text or "previous fail" in full_text:
            print(">> [FILTER] Hallucination: History=Success, Text=Fail")
            return False
    elif "Mission: FAIL" in history_section:
        if "mission succeeded" in full_text or "clean record" in full_text:
            print(">> [FILTER] Hallucination: History=Fail, Text=Success")
            return False
            
    return True

def validate_proposal_facts(current_proposal, response_json, player_id):
    if not current_proposal:
        return True
        
    thought = (response_json.get('thought') or '').lower()
    speech = (response_json.get('speech') or '').lower()
    full_text = thought + " " + speech
    
    leader = current_proposal.get('leader', '').lower()
    team = [m.lower() for m in current_proposal.get('team', [])]
    
    # 1. 檢查「隊長是否在隊伍中」
    leader_in_team = leader in team
    
    # [修正] 只有當事實不符時才攔截
    # 如果文本說 "leader is on the team"
    if "leader" in full_text and ("on the team" in full_text or "in the team" in full_text or "with yourself" in full_text):
        # 如果事實上隊長不在，這才是幻覺
        if not leader_in_team:
             # 允許玩家說 "I wish the leader was on the team" 這種例外很難用簡單規則抓，
             # 但為了保險，我們只抓明確錯誤。
             if "is on the team" in full_text or "putting yourself" in full_text:
                print(f">> [FILTER] Hallucination: Claimed Leader ({leader}) is on team, but they are NOT.")
                return False

    # 如果文本說 "leader is NOT on the team"
    if "leader" in full_text and ("not on the team" in full_text or "excluded yourself" in full_text):
        # 如果事實上隊長在，這是幻覺
        if leader_in_team:
            print(f">> [FILTER] Hallucination: Claimed Leader ({leader}) is NOT on team, but they ARE.")
            return False

    # 2. 檢查「我」是否在隊伍中 (這個保持嚴格，因為玩家最容易搞混自己)
    am_i_in = player_id.lower() in team
    if am_i_in:
        if "i am not on" in full_text or "left me out" in full_text:
            print(">> [FILTER] Hallucination: In team but claimed out.")
            return False
    else:
        # [修正] 避免誤殺 "I am voting on this team" 變成 "I am on this team"
        # 檢查是否明確說是成員
        if "i am on this team" in full_text or "i'm on it" in full_text or "picked me" in full_text:
            print(">> [FILTER] Hallucination: Out of team but claimed in.")
            return False
            
    # 3. 檢查是誰提的案
    if player_id.lower() != leader:
        if "my proposal" in full_text or "i proposed" in full_text:
             print(f">> [FILTER] Hallucination: Claimed ownership of team but leader is {leader}")
             return False

    return True

def validate_self_doxxing(response_json, role):
    """禁止在 Speech 中自爆邪惡身分"""
    speech = (response_json.get('speech') or '').lower()
    
    # 禁止詞彙
    forbidden = ["i am morgana", "i am assassin", "i am evil", "my partner is"]
    
    for term in forbidden:
        if term in speech:
            print(f">> [FILTER] Game Breaking: Self-doxxing in speech ('{term}').")
            return False
            
    return True

def validate_action_consistency(response_json):
    """檢查 Action 和 Speech 的一致性"""
    action = str(response_json.get('action', '')).upper()
    speech = (response_json.get('speech') or '').lower()
    
    if "VOTE_APPROVE" in action or "YES" in action:
        # Action 是贊成，Speech 卻說要反對
        if "vote reject" in speech or "voting reject" in speech or "reject this team" in speech:
             print(">> [FILTER] Inconsistency: Action=APPROVE, Speech=REJECT")
             return False
                 
    if "VOTE_REJECT" in action or "NO" in action:
        # Action 是反對，Speech 卻說要贊成
        if "vote approve" in speech or "voting approve" in speech or "approve this team" in speech:
             print(">> [FILTER] Inconsistency: Action=REJECT, Speech=APPROVE")
             return False
             
    return True

def validate_identity_constraints(response_json, role_str):
    """檢查 Servant 是否在 Thought 中產生擁有資訊的幻覺"""
    thought = (response_json.get('thought') or '')
    if "Servant" in role_str:
        # Servant 不應該知道誰是 Evil (除非是根據推理，但不能說 'I know' + 'Private Knowledge')
        if "private knowledge" in thought.lower() and "evil" in thought.lower():
             print(">> [FILTER] Hallucination: Servant claiming private knowledge.")
             return False
    return True

def validate_third_person_ref(response_json, player_id):
    """防止玩家用第三人稱稱呼自己 (例如 Player-2 說 'I agree with player-2')"""
    speech = (response_json.get('speech') or '').lower()
    thought = (response_json.get('thought') or '').lower()
    p_id_str = player_id.lower() # e.g. "player-2"
    
    # 檢查 speech 中是否包含自己的名字
    if p_id_str in speech:
        # 允許自我介紹 "I am player-2" 或 "As player-2"
        allowed_phrases = [
            f"i am {p_id_str}", 
            f"as {p_id_str}", 
            f"my name is {p_id_str}",
            f"this is {p_id_str}"
        ]
        
        # 暫時把允許的片語移除，看看還剩下什麼
        temp_speech = speech
        for phrase in allowed_phrases:
            temp_speech = temp_speech.replace(phrase, "")
            
        # 如果移除後還有自己的名字，通常就是語病
        if p_id_str in temp_speech:
            # 再檢查一種情況：引用別人的話 (比較難判斷，但為了嚴謹先攔截)
            print(f">> [FILTER] Third-person reference: {player_id} referred to themselves in speech.")
            return False
            
    return True

# ==========================================
# 3. 輔助工具函式
# ==========================================

def call_llm(url, model, system, user, temperature):
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": temperature,
        "response_format": {"type": "json_object"},
        "max_tokens": 4096
    }
    try:
        response = requests.post(url, json=payload, timeout=60)
        content = response.json()['choices'][0]['message']['content']
        return content
    except Exception as e:
        print(f"API Error: {e}")
        return json.dumps({"thought": "Error generating response", "action": None, "speech": "..."})

def get_score_with_analysis(prompt, move_json):
    """回傳 (score, analysis_json)"""
    user_input = f"Evaluate this move based on the game state.\n\n=== GAME STATE ===\n{prompt}\n\n=== PLAYER MOVE ===\n{move_json}"
    
    # 簡單重試機制 (最多 2 次)
    for _ in range(2):
        res_str = call_llm(SCORING_URL, SCORING_MODEL, SCORING_SYSTEM_PROMPT, user_input, temperature=0.1)
        try:
            res_json = json.loads(res_str)
            score = res_json.get('score', 0)
            
            ground_truth = res_json.get('ground_truth_verification', {})
            if ground_truth.get('is_factually_consistent') is False:
                # 這裡可以把 error_details 印出來方便除錯，但在 DPO 訓練時不需要
                # print(f">> [JUDGE] Fact Check Failed: {ground_truth.get('error_details')}")
                score = 0
            
            return score, res_json
        except json.JSONDecodeError:
            print("   [Judge Error] JSON Parse failed, retrying...")
            continue
        except Exception as e:
            print(f"   [Judge Error] {e}")
            break
            
    return 0, {}

def select_diverse_candidates(candidates: List[str], k=3):
    """使用 K-Means 選擇語義不同的候選回答"""
    if not embed_model or len(candidates) <= k:
        return candidates

    valid_cands = []
    texts_for_embed = []
    
    for c in candidates:
        try:
            # 只 Embedding speech 部分，因為那是差異最大的地方
            obj = json.loads(c)
            speech = obj.get('speech', '') or obj.get('thought', '')
            if speech:
                valid_cands.append(c)
                texts_for_embed.append(speech)
        except: continue
            
    if len(valid_cands) <= k: return valid_cands
    
    embeddings = embed_model.encode(texts_for_embed)
    kmeans = KMeans(n_clusters=k, n_init=10).fit(embeddings)
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, embeddings)
    return [valid_cands[i] for i in closest]

def save_dpo_data(user_prompt, scored_candidates):
    """儲存 DPO 數據，僅當 Chosen 分數夠高且與 Rejected 有差距時"""
    if not scored_candidates: return

    # 按照分數排序 (高到低)
    sorted_cands = sorted(scored_candidates, key=lambda x: x['score'], reverse=True)
    if len(sorted_cands) < 2: return

    best = sorted_cands[0]
    worst = sorted_cands[-1]

    # DPO 嚴格門檻
    MIN_CHOSEN_SCORE = 75
    MIN_SCORE_DIFF = 10

    if best['score'] >= MIN_CHOSEN_SCORE and (best['score'] - worst['score'] >= MIN_SCORE_DIFF):
        dpo_entry = {
            "system": SYSTEM_PROMPT,
            "user": user_prompt,
            "chosen": best['text'],
            "rejected": worst['text'],
            "metadata": {
                "chosen_score": best['score'],
                "rejected_score": worst['score'],
                "chosen_analysis": best.get('analysis', {}), 
                "rejected_analysis": worst.get('analysis', {})
            }
        }
        
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(dpo_entry, ensure_ascii=False) + "\n")
        print(f"   [SAVED] DPO Pair Saved. Scores: {best['score']} vs {worst['score']}")
    else:
        print(f"   [SKIP] Quality threshold not met. Best: {best['score']}, Diff: {best['score'] - worst['score']}")

# ==========================================
# 4. 遊戲引擎
# ==========================================

class AvalonGame:
    def __init__(self):
        # 6人局配置
        roles = ['Merlin', 'Morgana', 'Percival', 'Servant', 'Servant', 'Assassin']
        random.shuffle(roles)
        self.players = {f"player-{i}": roles[i] for i in range(6)}
        self.history = []
        self.dialogue = []
        self.mission_results = [] 
        self.round = 1
        self.sub_round = 1
        self.leader_idx = random.randint(0, 5)
        self.mission_sizes = [2, 3, 4, 3, 4] # 6人局任務人數

    def get_private_knowledge(self, p_id):
        role = self.players[p_id]
        evils = [k for k, v in self.players.items() if v in ['Morgana', 'Assassin']]
        if role == 'Merlin': return {"evils": evils}
        if role == 'Percival':
            merlin_cands = [k for k, v in self.players.items() if v in ['Merlin', 'Morgana']]
            random.shuffle(merlin_cands)
            return {"merlin_candidates": merlin_cands}
        if role in ['Morgana', 'Assassin']: return {"evils": evils}
        return {"evils": []}

    def format_prompt(self, p_id, phase, current_proposal=None):
        """構建輸入給 LLM 的 Prompt，包含 Fact 標註"""
        processed_history = []
        for entry in self.history:
            processed_history.append(entry)

        history_str = json.dumps(processed_history, indent=None) if processed_history else "[]"

        # 優化 Current Proposal 的顯示方式，加入 Note 提示
        proposal_text = "None"
        if current_proposal:
            leader = current_proposal['leader']
            team = current_proposal['team']
            is_leader_in = leader in team
            
            # Explicit Hint to prevent hallucination
            leader_hint = "(The Leader IS on the team)" if is_leader_in else "(The Leader is NOT on the team)"
            
            proposal_text = json.dumps(current_proposal) + f"\nNote: {leader_hint}"

        # 獲取 Recent Dialogue
        recent_dialogue = self.dialogue[-6:] if self.dialogue else []

        # === 新增邏輯：第 5 次提案的致命警告 ===
        critical_warning = ""
        if self.sub_round == 5:
            critical_warning = (
                "\n\n*** CRITICAL WARNING (HAMMER ROUND) ***\n"
                "This is the 5th proposal of the round. 4 proposals have already been rejected.\n"
                "IF THIS PROPOSAL IS REJECTED, EVIL WINS IMMEDIATELY.\n"
                "Strategy: If you are Good, you MUST VOTE APPROVE (even if the team looks bad) to avoid instant loss.\n"
                "If you are Evil, you might want to Approve to blend in, or Reject to win now if you have the votes."
            )
        # ==========================================

        return (
            f"Current Game State:\n- Round: {self.round}\n- Phase: {phase}\n"
            f"- Proposal Number: {self.sub_round}/5{critical_warning}\n"  # 這裡插入警告
            f"- Your Role: {p_id} (Refer to yourself as 'I')\n"
            f"- Your Identity: {self.players[p_id]}\n"
            f"- Private Knowledge: {json.dumps(self.get_private_knowledge(p_id))}\n\n"
            f"Game History (Fact): \n{history_str}\n\n"
            f"Current Proposal (Fact): \n{proposal_text}\n\n"
            f"Recent Dialogue:\n{json.dumps(recent_dialogue, indent=None)}\n\n"
        )

    def generate_move_with_dpo(self, p_id, phase, current_proposal=None, candidate_num=8, top_k=3):
        base_prompt = self.format_prompt(p_id, phase, current_proposal)
        
        # 根據階段添加具體指令 (增加 One-Shot Example)
        instruction = ""
        if phase == "Proposal":
            size = self.mission_sizes[self.round-1]
            instruction = (
                f"It is your turn to propose a team of {size} players.\n"
                f"Action Example: [\"player-1\", \"player-2\"]\n" # 明確範例
                f"Please propose {size} members for round {self.round}."
            )
        elif phase == "Team Vote":
            instruction = "Please decide whether to Approve or Reject the current proposal.\nAction Example: \"VOTE_APPROVE\" or \"VOTE_REJECT\""
        elif phase == "Mission Vote":
            instruction = "Please decide whether to succeed or fail the mission.\nAction Example: \"MISSION_SUCCESS\" or \"MISSION_FAIL\""
        
        selected_personas = random.sample(persona_prompts, min(candidate_num, len(persona_prompts)))
        candidates = []
        
        # 1. 生成多個候選
        for persona in selected_personas:
            user_input = base_prompt + instruction + f"\nConstraint: {persona}"
            raw_res = call_llm(PLAYER_URL, PLAYER_MODEL, SYSTEM_PROMPT, user_input, temperature=1.0)
            candidates.append(raw_res)
        
        # 2. 多樣性篩選
        diverse_candidates = select_diverse_candidates(candidates, k=top_k)

        # 3. 過濾與評分
        scored = []
        for cand_str in diverse_candidates:
            try:
                cand_json = json.loads(cand_str)
                
                # =========== 新增：針對 Proposal 階段的嚴格檢查 ===========
                if phase == "Proposal":
                    # 1. 安全獲取 action
                    raw_action = cand_json.get('action', '')
                    
                    # 2. 強制轉為字串 (解決 AttributeError: 'list' object has no attribute 'replace')
                    # 無論模型回傳的是 list ["p1", "p2"] 還是字串 "PROPOSE: [p1, p2]"，轉字串後都能被 regex 抓到
                    action_str = str(raw_action)
                    
                    # 3. 使用正則表達式提取 player-id
                    team_match = re.findall(r"player-\d", action_str)
                    required_size = self.mission_sizes[self.round-1]
                    
                    # 4. 如果提議人數不對，直接過濾
                    # 使用 set() 去重，避免模型輸出 ['player-0', 'player-0']
                    if len(set(team_match)) != required_size:
                        # 可以在這裡 print log 方便除錯，但在大量跑時建議註解掉
                        # print(f">> [FILTER] Invalid Proposal Size: Got {len(set(team_match))}, Expected {required_size}")
                        continue
                # =======================================================
                
                # --- APPLY ALL HARD FILTERS ---
                if not validate_history_consistency(base_prompt, cand_json): continue
                if not validate_proposal_facts(current_proposal, cand_json, p_id): continue
                if not validate_identity_constraints(cand_json, self.players[p_id]): continue
                if not validate_self_doxxing(cand_json, self.players[p_id]): continue
                if not validate_action_consistency(cand_json): continue
                # 新增這行
                if not validate_third_person_ref(cand_json, p_id): continue
                
                # --- LLM Judge (32B) ---
                score, analysis = get_score_with_analysis(base_prompt, cand_str)
                
                # 只有分數 > 0 (通過事實查核) 才保留
                if score > 0:
                    scored.append({"text": cand_str, "score": score, "analysis": analysis})
                    
            except json.JSONDecodeError:
                continue

        # 4. 決策與儲存
        if not scored:
            print(f"   [WARNING] All candidates filtered for {p_id}. Model hallucinating.")
            # Fallback 動作，不存 DPO
            fallback_action = None
            if phase == "Team Vote": fallback_action = "VOTE_REJECT"
            elif phase == "Mission Vote": fallback_action = "MISSION_SUCCESS" # Default to success for safety
            
            return {"thought": "System fallback due to hallucinations.", "action": fallback_action, "speech": "I am unsure about the situation."}

        # 選擇最高分的作為遊戲動作
        max_entry = max(scored, key=lambda x: x['score'])
        
        # 嘗試儲存 DPO
        save_dpo_data(base_prompt, scored)

        print(f"   Selected Move Score: {max_entry['score']} (Player: {p_id})")
        return json.loads(max_entry['text'])

# ==========================================
# 5. 主程式
# ==========================================

def run_simulation():
    game = AvalonGame()
    print(f"--- Game Start (Leader: player-{game.leader_idx}) ---")
    
    # 遊戲結束條件: 3 Success 或 3 Fail
    while len([r for r in game.mission_results if r]) < 3 and len([r for r in game.mission_results if not r]) < 3:
        leader_id = f"player-{game.leader_idx}"
        print(f"\n=== Round {game.round}-{game.sub_round} (Leader: {leader_id}) ===")
        
        # Phase 1: Proposal
        # 稍微增加 candidate_num 讓模型有更多機會生成正確格式
        proposal_move = game.generate_move_with_dpo(leader_id, "Proposal", None, candidate_num=8, top_k=3)
        
        # 解析提議
        try:
            team_match = re.findall(r"player-\d", str(proposal_move.get('action', '')))
            team = list(set(team_match))
            req_size = game.mission_sizes[game.round-1]
            
            # 如果還是錯 (代表所有候選都爛，fallback 觸發)，這時候只能強制修，但這一輪的 DPO 資料可能就廢了
            if len(team) != req_size:
                print(f"   [CRITICAL FAIL] Model failed to propose valid team after retries. Fallback random.")
                pool = [f"player-{i}" for i in range(6) if f"player-{i}" != leader_id]
                team = random.sample(pool, req_size)
                # 重要：如果這裡強制修了，建議不要讓這筆資料進入訓練，或者修改 speech 為通用語句
                proposal_move['speech'] = f"I propose {team}." 
        except:
             team = [f"player-{i}" for i in range(6) if i != game.leader_idx][:game.mission_sizes[game.round-1]]
             proposal_move['speech'] = f"I propose {team}."

        current_prop = {"leader": leader_id, "team": team}
        speech = proposal_move.get('speech', '...')
        game.dialogue.append(f"{leader_id}: {speech}")
        print(f"{leader_id} proposes {team}: \"{speech}\"")

        # Phase 2: Discussion (修改：讓除隊長外的所有人都發言)
        print("--- Discussion ---")
        # 為了避免順序固定導致的偏差，可以隨機打亂發言順序，但確保每個人都輪到
        discussion_order = [f"player-{i}" for i in range(6) if f"player-{i}" != leader_id]
        
        for p_id in discussion_order:
            # 這裡可以稍微降低 candidate_num 以加快速度，因為人數變多了
            disc_move = game.generate_move_with_dpo(p_id, "Discussion", current_prop, candidate_num=8, top_k=3)
            
            speech = disc_move.get('speech', '...')
            # 如果 speech 是空的或 ...，可以選擇不加入 dialogue 以節省空間，這裡視需求而定
            if len(speech) > 5:
                game.dialogue.append(f"{p_id}: {speech}")
                print(f"{p_id}: \"{speech}\"")

        # Phase 3: Team Vote
        print("--- Voting ---")
        votes_yes = []
        votes_no = []
        for i in range(6):
            p_id = f"player-{i}"
            v_move = game.generate_move_with_dpo(p_id, "Team Vote", current_prop, candidate_num=5, top_k=2)
            action_str = str(v_move.get('action')).upper()
            if "APPROVE" in action_str or "YES" in action_str:
                votes_yes.append(p_id)
            else:
                votes_no.append(p_id)
        
        passed = len(votes_yes) > len(votes_no)
        result_tag = "APPROVED" if passed else "REJECTED"
        print(f"Vote Result: {result_tag} (YES: {len(votes_yes)}, NO: {len(votes_no)})")
        
        # === 修改部分開始 ===
        # 原本的邏輯是不管成功失敗都先組字串，最後統一 append
        # 現在改為：失敗保留詳細記錄；成功則清空該回合失敗記錄，只留最終摘要
        
        if passed:
            # Phase 4: Mission Execution
            print("--- Mission Execution ---")
            fails = 0
            for p_id in team:
                role = game.players[p_id]
                # 只有壞人需要決策，好人強制 Success
                if role in ['Morgana', 'Assassin']:
                    m_move = game.generate_move_with_dpo(p_id, "Mission Vote", current_prop, candidate_num=5, top_k=2)
                    if "FAIL" in str(m_move.get('action')).upper():
                        fails += 1
                else:
                    # Good players always succeed
                    pass
            
            success = fails == 0 
            game.mission_results.append(success)
            res_str = "SUCCESS" if success else f"FAIL ({fails} fail votes)"
            print(f"Mission Result: {res_str}")
            
            # --- 歷史紀錄清洗邏輯 ---
            # 1. 找出並移除當前 Round 之前的所有失敗提案 (格式為 "Round X-Y")
            #    這樣當回合結束時，Prompt 中就不會殘留 "Round 1-1: Rejected" 這種資訊
            current_round_prefix = f"Round {game.round}-"
            game.history = [h for h in game.history if not h.startswith(current_round_prefix)]

            # 2. 加入最終成功的紀錄 (格式改為 "Round X: ...")
            #    這就是你要的簡潔格式，沒有 sub_round 編號
            final_entry = (f"Round {game.round}: Leader {leader_id} proposed {team}. "
                           f"Result: APPROVED (Yes: {len(votes_yes)}, No: {len(votes_no)})"
                           f" -> Mission: {res_str}.")
            game.history.append(final_entry)
            
            game.round += 1
            game.sub_round = 1
        else:
            # 提案被拒絕：
            # 暫時保留詳細格式 "Round X-Y"，這樣在同回合的下一次提案時，
            # 玩家還能看到剛才誰投了反對票。等到該回合有隊伍通過後，這些就會被上面的程式碼清除。
            fail_entry = (f"Round {game.round}-{game.sub_round}: Leader {leader_id} proposed {team}. "
                          f"Result: REJECTED (Yes: {len(votes_yes)}, No: {len(votes_no)})")
            game.history.append(fail_entry)

            game.sub_round += 1
            if game.sub_round > 5:
                print("Hammer rejected 5 times. Evil wins by default.")
                game.mission_results = [False, False, False] # Force end
        
        # 隊長輪替
        game.leader_idx = (game.leader_idx + 1) % 6

    # Phase 5: Assassination (If Good won 3 missions)
    if len([r for r in game.mission_results if r]) >= 3:
        print("\n--- Assassination Phase ---")
        assassin_id = next(k for k, v in game.players.items() if v == 'Assassin')
        ass_move = game.generate_move_with_dpo(assassin_id, "Assassination", None, candidate_num=5, top_k=2)
        
        # 解析刺殺目標
        target_match = re.search(r"player-\d", str(ass_move.get('action', '')))
        target_id = target_match.group(0) if target_match else "player-0"
        
        print(f"Assassin ({assassin_id}) assassinates {target_id}...")
        if game.players[target_id] == 'Merlin':
            print(f"SUCCESS! Target was Merlin. Evil Wins.")
        else:
            print(f"FAIL! Target was {game.players[target_id]}. Good Wins.")
    else:
        print("\nEvil Wins by 3 Mission Fails.")

if __name__ == "__main__":
    # 清空或創建輸出文件
    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            pass
    run_simulation()