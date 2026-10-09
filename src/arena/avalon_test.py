import json
import os
from openai import OpenAI
from tqdm import tqdm

#Config 設定
#INPUT_FILE = '1_test.json'      # 你的輸入檔案名稱
#OUTPUT_FILE = '1_result.json'  # 輸出的檔案名稱

# vLLM 的設定
# 請確認你的 vLLM 啟動參數，預設通常是 localhost:8000
VLLM_API_URL = "http://localhost:8001/v1" 
VLLM_API_KEY = "EMPTY" # vLLM 本地端通常不需要 Key，填 EMPTY 即可

# 初始化 OpenAI Client (連線到 vLLM)
client = OpenAI(
    base_url=VLLM_API_URL,
    api_key=VLLM_API_KEY,
)

# 為了確保模型名稱正確，我們抓取 server 上正在跑的模型名稱
try:
    models = client.models.list()
    MODEL_NAME = models.data[0].id
    print(f"正在使用模型: {MODEL_NAME}")
except Exception as e:
    print(f"無法自動偵測模型名稱，請手動設定。錯誤: {e}")
    MODEL_NAME = "your-model-name-here" # 如果自動偵測失敗，請手動填入

# 定義你的系統提示詞 (System Prompt / Rules)
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
    "3. HAMMER LOGIC: If you are on Proposal 5, you MUST Approve (unless you are Evil and want to win by default).\n"
    "4. MISSION VOTING: Good players MUST always vote SUCCESS. Evil players can choose SUCCESS or FAIL.\n\n"
    "*** ABSOLUTE PROHIBITIONS ***\n"
    "1. NEVER say 'I am Evil', 'I am Morgana', or 'I am Assassin'. Pretend to be Good.\n"
    "2. NEVER act as the Leader if it is not your turn.\n\n"
    "*** RESPONSE FORMAT ***\n"
    "JSON with 'thought' (internal strategy), 'action' (null), and 'speech' (public statement)."
)

def generate_avalon_response(user_prompt):
    """
    呼叫 LLM 生成回應
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7, # 可根據需要調整創造力
            max_tokens=1024   # 預留足夠的長度給 Thought 和 Speech
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"\n生成時發生錯誤: {e}")
        return None

def main():
    # 1. 讀取輸入檔案
    if not os.path.exists(INPUT_FILE):
        print(f"找不到檔案: {INPUT_FILE}")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"讀取到 {len(data)} 筆資料，開始處理...")

    processed_data = []

    # 2. 迴圈處理每一筆資料
    for item in tqdm(data):
        current_prompt = item
        #print(item)
        #print(type(item))
        
        # 呼叫 LLM
        llm_output = generate_avalon_response(current_prompt)
        
        if llm_output:
            # 3. 將結果加入資料中
            # 我們建立一個新欄位 'completion' 存放生成的結果
            prompt = item
            item = {}
            item['prompt'] = prompt
            item['completion'] = llm_output
            
            # 如果你想把結果直接拚接在 prompt 後面存成一個長字串，可以用下面這行：
            # item['prompt_with_completion'] = current_prompt + "\n" + llm_output
        else:
            item['completion'] = "Error generating response"

        processed_data.append(item)

    # 4. 寫入新的檔案
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)

    print(f"\n處理完成！結果已儲存至 {OUTPUT_FILE}")

if __name__ == "__main__":
    #對於processed_testdata資料夾中的檔案進行處理
    if not os.path.exists('original_test'):
        os.makedirs('original_test')
    for file_name in os.listdir('converted_output'):
        if file_name.endswith('.json'):
            INPUT_FILE = os.path.join('converted_output', file_name)
            OUTPUT_FILE = os.path.join('original_test', f'result_{file_name}')
        main()