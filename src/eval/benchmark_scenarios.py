"""
Qualitative Scenario Benchmark for Avalon Checkpoints.
Evaluates agent reasoning under iconic social deduction dilemmas (e.g. Merlin's Dilemma, Percival's Deduction).
"""

import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_PATH = "./qwen_4b_sft"
DPO_CHECKPOINT_DIR = "./qwen_4b_dpo_final"
CHECKPOINTS_TO_TEST = [60, 70, 80, 90]

SCENARIOS = [
    {
        "name": "Case 1: Merlin's Dilemma (Concealed Knowledge)",
        "desc": "Merlin knows Player-4 is Evil and included in the proposal. Must oppose without revealing his true identity.",
        "system": (
            "You are a strategic AI playing The Resistance: Avalon (6-player setup).\n"
            "Your Goal: Win by completing 3 Missions (Good).\n"
            "*** IMPORTANT ***\n"
            "You are Merlin. You know the Evils. You MUST hide your identity. Do NOT directly state that you know who is Evil."
        ),
        "user": (
            "Current Game State:\n"
            "- Round: 1\n- Phase: Team Vote\n- Proposal: 1/5\n- Your Role: player-2\n- Your Identity: Merlin\n"
            '- Private Knowledge: {"evils": ["player-4", "player-5"]}\n\n'
            'Current Proposal:\n{"leader": "player-0", "team": ["player-0", "player-4"]}\n\n'
            "Action Needed: Decide to Approve or Reject, and provide your speech."
        )
    },
    {
        "name": "Case 2: Assassin's Camouflage",
        "desc": "Assassin is included in the team. Must camouflage as Good to get the team approved so they can fail the quest.",
        "system": (
            "You are a strategic AI playing The Resistance: Avalon (6-player setup).\n"
            "Your Goal: Fail 3 Missions or Assassinate Merlin.\n"
            "*** IMPORTANT ***\n"
            "You are the Assassin. Pretend to be a loyal Servant of Arthur."
        ),
        "user": (
            "Current Game State:\n"
            "- Round: 1\n- Phase: Team Vote\n- Proposal: 2/5\n- Your Role: player-5\n- Your Identity: Assassin\n"
            '- Private Knowledge: {"evils": ["player-4"], "merlin_candidates": ["player-1", "player-2"]}\n\n'
            'Current Proposal:\n{"leader": "player-3", "team": ["player-3", "player-5"]}\n\n'
            "Note: You (player-5) are on the team!\nAction Needed: Decide to Approve or Reject, and provide speech."
        )
    },
    {
        "name": "Case 3: Percival's Deduction",
        "desc": "Percival sees two candidates (Merlin & Morgana) giving conflicting opinions. Must deduce the true Merlin.",
        "system": (
            "You are a strategic AI playing The Resistance: Avalon (6-player setup).\n"
            "Your Goal: Win by completing 3 Missions.\n"
            "*** IMPORTANT ***\n"
            "You are Percival. You see Merlin and Morgana but don't know which is which. You must deduce the true Merlin."
        ),
        "user": (
            "Current Game State:\n"
            "- Round: 2\n- Phase: Team Vote\n- Proposal: 3/5\n- Your Role: player-0\n- Your Identity: Percival\n"
            '- Private Knowledge: {"merlin_candidates": ["player-1", "player-4"]}\n\n'
            'Recent Dialogue:\n- player-1: "I trust this team. Let\'s approve it."\n- player-4: "This team looks very suspicious. Rejecting."\n\n'
            'Current Proposal:\n{"leader": "player-2", "team": ["player-2", "player-3", "player-5"]}\n\n'
            "Action Needed: Decide to Approve or Reject, and explain reasoning."
        )
    }
]

def main():
    print("Loading base tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    for step in CHECKPOINTS_TO_TEST:
        adapter_path = f"{DPO_CHECKPOINT_DIR}/checkpoint-{step}"
        print(f"\n{'='*50}\nEvaluating Checkpoint: {adapter_path}\n{'='*50}")

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        try:
            model = PeftModel.from_pretrained(base_model, adapter_path)
            model.eval()
        except Exception as e:
            print(f"Skipping Checkpoint {step} (Error: {e})")
            continue

        for scen in SCENARIOS:
            print(f"\n--- {scen['name']} ---")
            messages = [{"role": "system", "content": scen["system"]}, {"role": "user", "content": scen["user"]}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([text], return_tensors="pt").to("cuda")

            with torch.no_grad():
                out = model.generate(inputs.input_ids, max_new_tokens=512, temperature=0.7, do_sample=True, top_p=0.9)

            resp = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
            print(resp.split("assistant\n")[-1] if "assistant\n" in resp else resp)

        del model, base_model
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()