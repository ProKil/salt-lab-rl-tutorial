# Agentic RL from Scratch: GRPO on Tic-Tac-Toe

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ProKil/salt-lab-rl-tutorial/blob/main/agentic_rl_grpo_tictactoe.ipynb)

A minimal, from-scratch **GRPO** (Group Relative Policy Optimization) tutorial: a small
Qwen model is dropped into a tic-tac-toe environment and trained until it stops losing.
The LLM *is* the policy — it reads a board as text and writes back a move. Everything
(advantages, log-probs, the clipped surrogate + KL loss, the training loop) is implemented
by hand in a single notebook.

- **Notebook (fill-in-the-blank):** [`agentic_rl_grpo_tictactoe.ipynb`](agentic_rl_grpo_tictactoe.ipynb) ([open in Colab](https://colab.research.google.com/github/ProKil/salt-lab-rl-tutorial/blob/main/agentic_rl_grpo_tictactoe.ipynb) — use a T4 GPU runtime)
- **Notebook (solutions):** [`agentic_rl_grpo_tictactoe_solutions.ipynb`](agentic_rl_grpo_tictactoe_solutions.ipynb) ([open in Colab](https://colab.research.google.com/github/ProKil/salt-lab-rl-tutorial/blob/main/agentic_rl_grpo_tictactoe_solutions.ipynb)) — every blank filled in; Run All passes end to end with no edits. Use it to verify correctness.
- **Policy:** `Qwen/Qwen3.5-0.8B`, trained with **LoRA** adapters (the frozen base = reference policy for KL)
- **Target:** vs the deterministic `weak_opponent`, win rate → ~1.0

## Hardware

A single GPU is enough (a free Colab T4 works; any AWS GPU instance — `g4dn`/`g5`/`p3` — is
plenty). Generations are tiny. CPU-only works for the pure-Python parts and a smoke test,
but real training wants a GPU.

## Setup (AWS GPU box)

```bash
git clone https://github.com/ProKil/salt-lab-rl-tutorial.git
cd salt-lab-rl-tutorial
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Headless (executes every cell, fails loudly on any error):

```bash
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 agentic_rl_grpo_tictactoe.ipynb
```

Or interactively:

```bash
jupyter lab   # open the notebook and Run All
```

## Knobs (in the Part 8 "Run training" cell)

| Setting | Meaning |
|---------|---------|
| `SMOKE_TEST = True` | A few steps to prove the plumbing runs. **Flip to `False` for real training.** |
| `OPPONENT = weak_opponent` | Deterministic/exploitable → reachable 100% wins. Swap `random_opponent` for the harder "0% losses" goal. |
| `steps = 400` | Max training steps; the loop early-stops once the rolling window is ≥95% wins with 0 losses (vs `weak_opponent` that happens well before 100 steps). |
| `GROUP_SIZE = 8` | Games sampled per GRPO group. Don't raise to 16 on a 16GB T4: full-length games OOM in float32. |
| `LR = 1e-4` | LoRA learning rate. |

## Notes on correctness

The core moves that make training actually converge (rather than spew illegal moves):

1. **`enable_thinking=False`** in the prompt template — Qwen3.x reasons by default and would
   spend the whole `max_new_tokens` budget inside `<think>…</think>`, never emitting a digit,
   so every move parsed as illegal.
2. **`parse_move` reads the first legal *digit*, not the first integer** — the untrained model
   glues its answer to junk (`"20/\n# 20"` = cell 2 + garbage), so integer-parsing rejects
   ~88% of moves as illegal. Worse, once *every* game in a group is illegal, all rewards tie
   at `-1`, the group advantage is exactly zero, and so is the gradient: training can never
   start. Digit-parsing keeps ~88% of untrained moves legal and gives GRPO reward variance
   to bootstrap from.
3. **`max_new_tokens=4`** (sampling *and* greedy eval) — a move is one digit; a longer budget
   only buys trailing junk after the answer.
4. **Illegal moves are credited** — `play_game` records the offending turn *before* the legality
   check, so an illegal generation actually receives the `-1` gradient that teaches the format.
5. **LoRA learning rate** raised to `1e-4` (`1e-6` was too small to move the adapters).
6. **`GROUP_SIZE=8`** — 16 exceeds a 16GB T4's memory in float32 once games run full length
   (the full-vocab log-softmax over every turn in the group is the big allocation).
