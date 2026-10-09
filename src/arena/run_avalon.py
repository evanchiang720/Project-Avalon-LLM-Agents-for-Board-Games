"""
Multi-Agent Avalon Game Arena.
Simulates full 6-player games between two LLMs (Good vs. Evil) via OpenAI-compatible endpoints.
"""

import json
import random
import requests
import re
import os
import argparse
from typing import List, Dict, Any

HISTORY_FILE = "dpo_vs_dpo_match_history.jsonl"

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
    "1. AVOID THE 5TH PROPOSAL: Reaching proposal 5 (Hammer) forces a blind 'Approve' for Good.\n"
    "2. OPTIMAL TIMING: Good players should try to approve a clean team on Proposal 2 or 3.\n"
    "3. HAMMER LOGIC: On Proposal 5, Good MUST Approve to avoid default defeat.\n"
    "4. MISSION VOTING: Good players MUST always vote SUCCESS. Evil can vote SUCCESS or FAIL.\n\n"
    "*** RESPONSE FORMAT ***\n"
    "JSON with 'thought' (internal strategy), 'action', and 'speech' (public statement)."
)


def call_llm(config, system, user, temperature):
    payload = {
        "model": config["model"],
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": temperature,
        "response_format": {"type": "json_object"},
        "max_tokens": 1024,
    }
    try:
        response = requests.post(config["url"], json=payload, timeout=60)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return json.dumps({"thought": f"API Error {response.status_code}", "action": None, "speech": "..."})
    except Exception as e:
        return json.dumps({"thought": f"Connection Error: {e}", "action": None, "speech": "..."})


class AvalonGame:
    def __init__(self, good_config, evil_config):
        self.good_config = good_config
        self.evil_config = evil_config
        roles = ["Merlin", "Morgana", "Percival", "Servant", "Servant", "Assassin"]
        random.shuffle(roles)
        self.players = {f"player-{i}": roles[i] for i in range(6)}
        self.history = []
        self.dialogue = []
        self.mission_results = []
        self.round = 1
        self.sub_round = 1
        self.leader_idx = random.randint(0, 5)
        self.mission_sizes = [2, 3, 4, 3, 4]

    def get_faction(self, p_id):
        return "Good" if self.players[p_id] in ["Merlin", "Percival", "Servant"] else "Evil"

    def get_private_knowledge(self, p_id):
        role = self.players[p_id]
        evils = [k for k, v in self.players.items() if v in ["Morgana", "Assassin"]]
        if role == "Merlin":
            return {"evils": evils}
        if role == "Percival":
            merlin_cands = [k for k, v in self.players.items() if v in ["Merlin", "Morgana"]]
            random.shuffle(merlin_cands)
            return {"merlin_candidates": merlin_cands}
        if role in ["Morgana", "Assassin"]:
            return {"evils": evils}
        return {"evils": []}

    def format_prompt(self, p_id, phase, current_proposal=None):
        processed_history = json.dumps(self.history) if self.history else "[]"
        proposal_text = json.dumps(current_proposal) if current_proposal else "None"
        recent_dialogue = self.dialogue[-6:] if self.dialogue else []
        critical_warning = (
            "\n*** CRITICAL WARNING: Proposal 5. IF REJECTED, EVIL WINS. MUST APPROVE IF GOOD ***\n"
            if self.sub_round == 5
            else ""
        )

        return (
            f"Current Game State:\n- Round: {self.round}\n- Phase: {phase}\n"
            f"- Proposal: {self.sub_round}/5{critical_warning}\n"
            f"- Your Role: {p_id} ({self.players[p_id]})\n"
            f"- Private Knowledge: {json.dumps(self.get_private_knowledge(p_id))}\n\n"
            f"Game History (Fact):\n{processed_history}\n\n"
            f"Current Proposal (Fact):\n{proposal_text}\n\n"
            f"Recent Dialogue:\n{json.dumps(recent_dialogue)}\n\n"
        )

    def generate_move(self, p_id, phase, current_proposal=None):
        base_prompt = self.format_prompt(p_id, phase, current_proposal)
        instructions = {
            "Proposal": f"Propose a team of {self.mission_sizes[self.round-1]} players. Action Example: [\"player-1\", \"player-2\"]",
            "Discussion": "Analyze the situation and discuss who to trust.",
            "Team Vote": "Decide whether to Approve or Reject. Action Example: \"VOTE_APPROVE\" or \"VOTE_REJECT\"",
            "Mission Vote": "Vote on the mission. Action Example: \"MISSION_SUCCESS\" or \"MISSION_FAIL\"",
            "Assassination": "Assassin: Pick a player to assassinate. Action Example: \"player-1\"",
        }

        user_input = base_prompt + instructions.get(phase, "")
        target_config = self.good_config if self.get_faction(p_id) == "Good" else self.evil_config

        for attempt in range(3):
            temp = 0.7 if attempt == 0 else 0.9
            raw_res = call_llm(target_config, SYSTEM_PROMPT, user_input, temperature=temp)
            try:
                cand_json = json.loads(raw_res)
                if phase == "Proposal":
                    raw_action = str(cand_json.get("action", ""))
                    team_match = list(set([f"player-{m}" for m in re.findall(r"\d", raw_action)]))
                    if len(team_match) == self.mission_sizes[self.round - 1]:
                        cand_json["action"] = str(team_match)
                        return cand_json
                    continue
                return cand_json
            except json.JSONDecodeError:
                continue

        fallback_action = "VOTE_REJECT" if phase == "Team Vote" else ("MISSION_SUCCESS" if phase == "Mission Vote" else None)
        return {"thought": "Fallback response.", "action": fallback_action, "speech": "I am thinking..."}


def run_simulation(good_conf, evil_conf, game_num):
    print(f"\n{'='*50}\nGAME {game_num} | Good: {good_conf['name']} vs Evil: {evil_conf['name']}\n{'='*50}")
    game = AvalonGame(good_conf, evil_conf)
    winner, end_reason = "Unknown", "Unknown"

    while True:
        leader_id = f"player-{game.leader_idx}"
        print(f"\n--- Round {game.round}.{game.sub_round} (Leader: {leader_id}) ---")

        # 1. Proposal Phase
        prop_move = game.generate_move(leader_id, "Proposal", None)
        team = [m for m in re.findall(r"player-\d", str(prop_move.get("action", "")))]
        req_size = game.mission_sizes[game.round - 1]
        if len(team) != req_size:
            pool = [f"player-{i}" for i in range(6) if f"player-{i}" != leader_id]
            team = random.sample(pool, req_size)

        current_prop = {"leader": leader_id, "team": team}
        speech = prop_move.get("speech", "...")
        game.dialogue.append(f"{leader_id}: {speech}")
        print(f"Proposal by {leader_id}: {team} | \"{speech}\"")

        # 2. Discussion Phase
        for p_id in [f"player-{i}" for i in range(6) if f"player-{i}" != leader_id]:
            d_move = game.generate_move(p_id, "Discussion", current_prop)
            if d_move.get("speech"):
                game.dialogue.append(f"{p_id}: {d_move['speech']}")

        # 3. Team Vote Phase
        votes_yes = [p for p in [f"player-{i}" for i in range(6)] if "APPROVE" in str(game.generate_move(p, "Team Vote", current_prop).get("action")).upper()]
        passed = len(votes_yes) > 3
        print(f"Vote Result: {'APPROVED' if passed else 'REJECTED'} ({len(votes_yes)}/6 Approved)")

        if passed:
            fails = sum(1 for p in team if game.players[p] in ["Morgana", "Assassin"] and "FAIL" in str(game.generate_move(p, "Mission Vote", current_prop).get("action")).upper())
            success = (fails == 0)
            game.mission_results.append(success)
            res_str = "SUCCESS" if success else f"FAIL ({fails} fail cards)"
            game.history.append(f"Round {game.round}: Proposal {team} -> {res_str}")
            game.round += 1
            game.sub_round = 1
        else:
            game.history.append(f"Round {game.round}.{game.sub_round}: Proposal {team} -> REJECTED")
            game.sub_round += 1
            if game.sub_round > 5:
                winner, end_reason = "Evil", "5 Failed Proposals (Hammer Rejection)"
                break

        game.leader_idx = (game.leader_idx + 1) % 6

        # Check Termination
        if game.mission_results.count(False) >= 3:
            winner, end_reason = "Evil", "3 Failed Missions"
            break
        elif game.mission_results.count(True) >= 3:
            assassin = next(k for k, v in game.players.items() if v == "Assassin")
            target_match = re.search(r"player-\d", str(game.generate_move(assassin, "Assassination", None).get("action", "")))
            target = target_match.group(0) if target_match else "player-0"
            if game.players[target] == "Merlin":
                winner, end_reason = "Evil", "Assassination Success (Merlin Killed)"
            else:
                winner, end_reason = "Good", "Assassination Failed (Good Victory)"
            break

    print(f"\n🏆 Winner: {winner} ({end_reason})")
    record = {
        "game_id": game_num,
        "winner_faction": winner,
        "winning_model": good_conf["name"] if winner == "Good" else evil_conf["name"],
        "end_reason": end_reason,
        "good_model": good_conf["name"],
        "evil_model": evil_conf["name"],
        "roles": game.players,
        "history": game.history,
        "dialogue": game.dialogue,
    }
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return winner


if __name__ == "__main__":
    MODEL_A = {"url": "http://localhost:8001/v1/chat/completions", "model": "Qwen3-4B-DPO", "name": "Qwen3-4B-DPO"}
    MODEL_B = {"url": "http://localhost:8001/v1/chat/completions", "model": "Qwen3-30B", "name": "Qwen3-30B"}
    run_simulation(MODEL_A, MODEL_B, game_num=1)