# 🛡️ Avalon-LLM-DPO: Optimizing Strategic Social Deduction Agents with Limited Data and DPO

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![TRL](https://img.shields.io/badge/TRL-DPO%20%26%20SFT-green.svg)](https://github.com/huggingface/trl)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Evaluation of Small Models Trained with Limited Data for *The Resistance: Avalon***  
> **Advisor**: Prof. I-Chen Wu  
> **Team**: Chia-Tung Chiang, Si-Kai Zhang

---

## 📖 Introduction

*The Resistance: Avalon* is a complex hidden-role social deduction board game requiring strategic deception, belief revision, and multi-agent coordination. While massive frontier models perform well in such environments, running them is computationally expensive.

This project investigates **how to train small-parameter language models (e.g., Qwen-4B) to play Avalon effectively under scarce human gameplay data and constrained resources.** 

### Key Contributions
* **Synthetic Data & CoT Enrichment**: Overcame data scarcity by synthesizing game logs using larger LLMs and injecting internal **Chain-of-Thought (CoT)** reasoning into historical dialogues.
* **Persona & Embedding-driven Preference Pairs**: Sampled diverse actions across **12 distinct player personas**, filtered via text embedding distances, and ranked by a fine-tuned **30B LLM-as-a-Judge** to construct high-quality `(chosen, rejected)` pairs.
* **Direct Preference Optimization (DPO)**: Applied DPO with LoRA and FlashAttention-2, boosting winning rates by **nearly 2x** and significantly lowering hallucination rates without requiring a reinforcement learning reward model.

---

## 🏛️ Pipeline Architecture

The overall framework follows a three-stage pipeline:

```
[Phase 0: SFT] ───────────────► [Phase 1: Preference Generation] ────────► [Phase 2: DPO]
• 20 Human + 100 Synthetic Logs  • 12 Diverse Personas               • Target: Qwen3-4B
• CoT Injection (Thought + Speech)• Embedding Diversity Selection     • FlashAttention-2
• Trained 30B Judge & 4B Base    • 30B Model Trust Scoring (Chosen/Rej)• LoRA (r=64, a=128)
```

1. **Phase 0 (SFT)**: 
   - Supervised fine-tuning of `Qwen3-4B` and `Qwen3-30B-A3B` on dialogue logs with human game data weighted 3x.
2. **Phase 1 (Data Generation)**: 
   - SFT `Qwen3-4B` generates 12 candidate plays per game step based on 12 personas.
   - Computes text embeddings to extract 3 distinct candidates with maximum distance.
   - A fine-tuned `Qwen3-30B` Judge model rates simulated player trust scores to assign `chosen` (highest trust) and `rejected` (lowest trust).
3. **Phase 2 (DPO)**: 
   - Directly optimizes the 4B policy to favor strategic, deceptive, and context-aligned responses.

---

## 📊 Experimental Results

### 1. Quantitative Evaluation (DeepEval LLM-as-a-Judge)
Evaluated across **Strategy**, **Factual Consistency (Hallucination)**, and **Linguistic Diversity** (0–100 scale, judged by `gemini-2.5-flash`):

| Model Variant | Strategy Score (↑) | Hallucination Score (↓) | Diversity Score (↑) |
| :--- | :---: | :---: | :---: |
| **Original Qwen3-4B** | 56.65 (±37.12) | 40.71 (±38.19) | **61.80** (±19.98) |
| **SFT Model** | 58.55 (±35.85) | 46.03 (±42.47) | 41.01 (±27.98) |
| **SFT + DPO (Ours)** | **62.88** (±33.27) | **39.93** (±39.23) | 47.52 (±22.55) |

> 📌 **Key Takeaway**: DPO substantially improves tactical decision-making while mitigating hallucinations caused by hallucinated game states. Language diversity converges slightly toward optimal, concise gameplay conventions.

---

### 2. Head-to-Head Arena Win Rates (vs. Qwen3-30B)
Win rates across **150 simulated games** per faction playing against `Qwen3-30B-A3B-Instruct`:

| Model Condition | Good Faction (Loyal Servants, Merlin) | Evil Faction (Minions, Assassin) |
| :--- | :---: | :---: |
| **Original (Un-tuned)** | 20.7% | 36.0% |
| **SFT Only** | 31.3% | 45.3% |
| **SFT + DPO (Ours)** | **40.7%** *(~2x improvement)* | **49.3%** |

---

## 🔍 Case Studies

### Case 1: Logical Traitor Deduction (Servant Role)
* **History**: Mission 1 succeeded with `[P1, P2]`. Mission 2 failed when `P3` was added (`[P1, P2, P3]`). P3 now proposes `[P1, P3, P4]`.
* **Original Model**: *"I agree with Player 3's logic... The team composition seems safe enough. I vote approve."* ❌ *(Fails to connect failure to P3)*
* **SFT+DPO Model**: *"Quest 1 succeeded with P1 and P2... Quest 2 failed with all three... This strongly indicates Player 3 is the Evil... unacceptable risk. I vote NO."* ✅ *(Accurate logical deduction)*

### Case 2: Hidden Knowledge Concealment (Merlin Role)
* **Situation**: Merlin knows `P2` is Evil. `P4` proposes `[P4, P2]`. Merlin must block P2 without exposing his identity to the Assassin.
* **Original Model**: Proposes suspicious circular questions without taking a firm stance.
* **SFT+DPO Model**: *"I strongly advise against this team. There is no reason to approve a team that includes a player we have zero information on."* ✅ *(Tactically blocks Evil using "lack of information" as plausible deniability)*

---

## 📁 Repository Structure

```text
avalon-llm-dpo/
├── configs/
│   └── ds_config.json              # DeepSpeed ZeRO configuration
├── data/
│   ├── samples/                    # Sample JSONL datasets for reproduction
│   └── avalon_dpo_final.jsonl      # Final preference training pairs
├── eval_results/                   # Test inference logs & LLM-judge scores
│   ├── original/
│   ├── sft/
│   └── sft_dpo/
├── src/
│   ├── arena/                      # Multi-agent Avalon game simulation
│   │   ├── avalon.py               # Core game engine rules
│   │   ├── run_avalon.py           # Head-to-head match runner (via vLLM/OpenAI API)
│   │   └── avalon_test.py          # Benchmark inference script
│   ├── data_prep/                  # CoT augmentation & data processing
│   │   └── sft_augmentation.py
│   ├── training/                   # Model alignment scripts
│   │   ├── train_4b_player_sft.py  # Stage 0: Qwen-4B SFT training
│   │   ├── train_30b_judge.py      # Stage 0: Qwen-30B Judge training
│   │   └── train_dpo.py            # Stage 2: LoRA + FlashAttention-2 DPO
│   └── eval/                       # Metrics & Qualitative Benchmarks
│       ├── calc_scores.py          # Aggregates DeepEval metrics table
│       ├── evaluate_llm_judge.py   # LLM-as-a-Judge auto-evaluator via Gemini API
│       └── benchmark_scenarios.py  # Checkpoint verification across classic dilemmas
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/<your-username>/avalon-llm-dpo.git
cd avalon-llm-dpo
pip install -r requirements.txt
```

### 2. Training
#### Stage 1: SFT Target Model
```bash
python src/training/train_4b_player_sft.py
```

#### Stage 2: DPO Alignment (Optimized for FlashAttention-2 & bfloat16)
```bash
python src/training/train_dpo.py \
    --sft_model_path ./qwen_4b_sft \
    --dataset_path data/avalon_dpo_final.jsonl \
    --output_dir ./qwen_4b_dpo_final \
    --lora_r 64 \
    --lora_alpha 128
```

### 3. Running Multi-Agent Game Arena
Launch a local OpenAI-compatible inference server (e.g., vLLM):
```bash
vllm serve ./qwen_4b_dpo_final --port 8001
```
Simulate live 6-player matches:
```bash
python src/arena/run_avalon.py
```

### 4. Evaluation
Aggregate DeepEval evaluation statistics:
```bash
python src/eval/calc_scores.py
```
Score model outputs using Gemini as an automated judge:
```bash
export GEMINI_API_KEY="your_api_key"
python src/eval/evaluate_llm_judge.py --input_folder eval_results/sft --model gemini-1.5-flash
```

---

## 📜 References
* [1] *Long-Horizon Dialogue Understanding for Role Identification in the Game of Avalon with Large Language Models* (arXiv:2311.05720)
* [2] *AvalonBench: Evaluating LLMs Playing the Game of Avalon* (arXiv:2310.05036)
* [3] *Learning Strategic Language Agents in the Werewolf Game with Iterative Latent Space Policy Optimization* (arXiv:2502.04686)
