"""
Avalon Evaluation Metric Analyzer.
Aggregates DeepEval scores (Strategy, Hallucination, Diversity) across experimental runs.
"""

import os
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

# 自動抓取專案根目錄 (avalon_clean_repo)
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent  # src/eval -> src -> avalon_clean_repo


def analyze_scores(folder_path):
    """Analyzes a folder containing DeepEval JSON results."""
    folder = Path(folder_path)
    if not folder.exists():
        print(f"❌ 資料夾不存在: {folder.resolve()}")
        return None

    # 同時支援 score_*.json 以及 result_*.json
    file_pattern = re.compile(r"^(score|result)_.*\.json$")
    all_scores = defaultdict(list)
    file_count = 0

    json_files = list(folder.glob("*.json"))
    if not json_files:
        print(f"⚠️ 資料夾內沒有找到任何 .json 檔案: {folder.resolve()}")
        return None

    for file_path in json_files:
        if file_pattern.match(file_path.name):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                    # 嘗試抓取 evaluation 內容
                    items = []
                    if isinstance(data, dict):
                        if "evaluation" in data:
                            items = data["evaluation"]
                        else:
                            # 有些格式可能直接將 items 存在外層字典
                            items = list(data.values())
                    elif isinstance(data, list):
                        items = data

                    found_valid_entry = False
                    for entry in items:
                        if not isinstance(entry, dict):
                            continue
                        for category in ["strategy", "hallucination", "diversity"]:
                            if category in entry:
                                cat_data = entry[category]
                                if isinstance(cat_data, dict) and "score" in cat_data:
                                    try:
                                        val = float(cat_data["score"])
                                        all_scores[f"{category}_score"].append(val)
                                        found_valid_entry = True
                                    except (ValueError, TypeError):
                                        continue

                    if found_valid_entry:
                        file_count += 1

            except Exception as e:
                print(f"  [!] 讀取 {file_path.name} 錯誤: {e}")

    stats_result = {}
    for key in ["strategy_score", "hallucination_score", "diversity_score"]:
        vals = all_scores[key]
        stats_result[key] = {
            "mean": statistics.mean(vals) if vals else 0.0,
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "count": len(vals),
        }
    stats_result["file_count"] = file_count
    return stats_result


def print_summary(all_results):
    """Prints a markdown/CLI compatible table of the results."""
    metrics = [
        ("Strategy Score (↑)", "strategy_score"),
        ("Hallucination Score (↓)", "hallucination_score"),
        ("Diversity Score (↑)", "diversity_score"),
    ]

    print("\n" + "=" * 85)
    print(f"{'Model Condition':<25} | {'Metric':<26} | {'Mean':<8} | {'Std Dev':<8} | {'Count':<5}")
    print("-" * 85)

    for folder_name, stats in all_results.items():
        if not stats or all(m["count"] == 0 for m in stats.values() if isinstance(m, dict)):
            print(f"{folder_name:<25} | No valid score entries found")
            print("-" * 85)
            continue

        first_row = True
        label = folder_name.replace("eval_results/", "").upper()
        for display_name, key in metrics:
            m = stats[key]
            folder_display = label if first_row else ""
            print(f"{folder_display:<25} | {display_name:<26} | {m['mean']:>8.2f} | {m['std']:>8.2f} | {m['count']:>5}")
            first_row = False
        print("-" * 85)


if __name__ == "__main__":
    # 使用相對於 REPO_ROOT 的絕對路徑
    target_folders = {
        "ORIGINAL": REPO_ROOT / "eval_results" / "original",
        "SFT": REPO_ROOT / "eval_results" / "sft",
        "SFT_DPO": REPO_ROOT / "eval_results" / "sft_dpo",
    }

    summary_data = {}

    print(f">>> 專案根目錄定位至: {REPO_ROOT}")
    print(">>> Aggregating Avalon DeepEval Evaluation Metrics...")
    
    for label, folder in target_folders.items():
        result = analyze_scores(folder)
        summary_data[label] = result
        if result and result["strategy_score"]["count"] > 0:
            print(f" Analyzed {label}: {result['file_count']} files, {result['strategy_score']['count']} samples")
        else:
            print(f"⚠️ Warning: No valid evaluation data found in {label} ({folder})")

    print_summary(summary_data)