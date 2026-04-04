# OthelloRL

A high-performance Othello reinforcement learning environment written in C, with a PPO + LSTM agent that beats classical minimax search after 136M training steps.

The environment is open — clone it, train your own agent, and benchmark it against negamax.

---

## Results

| Opponent | Win Rate |
|---|---|
| Negamax depth 1 | 100% |
| Negamax depth 2 | 100% |
| Negamax depth 3 | 100% |
| Negamax depth 5 | 89.5% |

Evaluated with `checkpoint_016600` at `global_step=135,987,200`. Full training run tracked on [W&B](https://wandb.ai/sriramanasuri-georgia-institute-of-technology/Connect4RL-othello/runs/1on75tog).

---

## How It Works

**Environment** — The Othello game engine is written in C and compiled as a Python extension via ctypes. It runs up to 512 parallel games simultaneously, which keeps the environment from being the training bottleneck.

**Policy** — PPO with an LSTM backbone (256 hidden units). The LSTM captures temporal structure across moves — positional commitments made early in a game influence the endgame in ways a feedforward net misses.

**Curriculum** — Training escalates through six phases based on `total_timesteps`:

| Phase | Opponent | Starts at |
|---|---|---|
| 0 | Random | 0% |
| 1 | Negamax depth 1 | 10% |
| 2 | Negamax depth 2 | 25% |
| 3 | Negamax depth 3 | 40% |
| 4 | Negamax depth 5 | 60% |
| 5 | Self-play | 75% |

The agent is never handed strategy. Corner control, forced passes, endgame conversion — all emerged from competitive pressure.

---

## Prerequisites

- Python 3.10+
- gcc (macOS: ships with Xcode Command Line Tools — `xcode-select --install`)
- [raylib](https://www.raylib.com/) *(optional — only needed for visual rendering)*
  ```bash
  # macOS
  brew install raylib
  ```

---

## Setup

```bash
git clone https://github.com/sanasuri101/OthelloRL.git
cd OthelloRL

# Install Python dependencies
pip install pufferlib>=3.0 gymnasium numpy torch

# Build the C environment
cd othello && make && cd ..
```

To build without rendering support (no raylib required):

```bash
cd othello && NO_RENDER=1 make && cd ..
```

---

## Run the Pre-trained Agent

Download `checkpoint_016600.pt` from the [Releases page](https://github.com/sanasuri101/OthelloRL/releases) and place it anywhere.

```bash
# Ladder benchmark — win rate vs negamax at all depths
python othello/run_eval.py --ladder --checkpoint path/to/checkpoint_016600.pt

# Watch it play (requires raylib)
python othello/run_eval.py --render --checkpoint path/to/checkpoint_016600.pt

# Play against it yourself (requires raylib)
python othello/run_eval.py --human --checkpoint path/to/checkpoint_016600.pt

# Headless — 100 games, win/loss/draw stats
python othello/run_eval.py --episodes 100 --checkpoint path/to/checkpoint_016600.pt
```

---

## Train Your Own Agent

```bash
# Train with default config (CPU, 150M steps)
python -m othello.train

# Train with W&B tracking
python -m othello.train --wandb --wandb_project your-project-name

# Use GPU if available
python -m othello.train --train.device cuda

# Shorter run to test your setup (5M steps)
python -m othello.train --train.total_timesteps 5000000

# Resume from a checkpoint
python -m othello.train --load_checkpoint experiments/othello_ppo/checkpoint_001000.pt
```

All hyperparameters are in `othello/config.ini` and can be overridden from the CLI with `--section.key value`.

### Run a Hyperparameter Sweep

```bash
# Requires wandb
pip install wandb
wandb sweep sweep.yaml
wandb agent <sweep-id>
```

The sweep config uses Bayesian optimization over learning rate, entropy coefficient, BPTT horizon, gamma, and hidden size.

---

## Benchmark Your Agent

Once you have a checkpoint, run the full ladder:

```bash
python othello/run_eval.py --ladder --checkpoint experiments/othello_ppo/best/checkpoint_016600.pt
```

Output:
```
Evaluation Results
--------------------------------
  vs negamax d1: 100.0%  ████████████████████
  vs negamax d2: 100.0%  ████████████████████
  vs negamax d3: 100.0%  ████████████████████
  vs negamax d5:  89.5%  █████████████████▊
```

For programmatic evaluation in your own code:

```python
from othello.eval import evaluate
from othello.othello import Othello
from othello.train import make_policy
import torch

env = Othello(num_envs=1)
policy = make_policy(env, hidden_size=256)
env.close()

ckpt = torch.load("path/to/checkpoint.pt", map_location="cpu", weights_only=False)
policy.load_state_dict(ckpt["policy_state_dict"])

results = evaluate(policy, depths=[1, 2, 3, 5], n_games=100)
for depth, win_rate in results.items():
    print(f"vs negamax d{depth}: {win_rate:.1%}")
```

---

## Project Structure

```
OthelloRL/
├── othello/
│   ├── binding.c       # C game engine (compile with make)
│   ├── othello.h       # Board logic, legal moves, flipping
│   ├── negamax.h       # Minimax search implementation
│   ├── render.h        # Raylib visual rendering
│   ├── othello.py      # Python wrapper (vectorized env)
│   ├── train.py        # PPO training loop
│   ├── eval.py         # evaluate() function used during training
│   ├── run_eval.py     # CLI eval: render, ladder, human, episodes
│   ├── curriculum.py   # Phase scheduler
│   └── config.ini      # All hyperparameters
├── sweep.yaml          # W&B Bayesian sweep config
├── colab_sweep.ipynb   # Run sweeps on Google Colab
└── video/              # Remotion demo video source
```

---

## Tests

```bash
cd othello
pip install pytest
pytest tests/
```

---

## Notes

- Training on CPU is supported and how the main run was done. A GPU speeds up the policy forward pass but the C environment is the throughput driver.
- The `.so` compiled extension is platform-specific — you must run `make` for your own machine.
- Checkpoints save every 200 gradient updates by default (`checkpoint_interval` in config.ini).
- W&B run for `checkpoint_016600`: [view on W&B](https://wandb.ai/sriramanasuri-georgia-institute-of-technology/Connect4RL-othello/runs/1on75tog)
