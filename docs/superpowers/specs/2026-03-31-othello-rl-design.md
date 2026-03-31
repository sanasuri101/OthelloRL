# Othello RL — Design Specification

**Date:** 2026-03-31
**Goal:** Build an Othello (Reversi) RL environment with PufferLib, using a curriculum training pipeline (random → negamax → self-play) and raylib rendering. Structured as a PufferLib ocean-compatible environment.

---

## Architecture: Hybrid C Engine + Python Opponent Layer

Game logic and board operations in C (bitboard, move generation, flipping). Opponent selection and curriculum managed in Python. Self-play calls the policy network from Python; negamax opponent implemented in C but invoked from Python.

### Project Structure

```
othello/
├── othello.h         # C game engine: bitboard, move gen, flip, scoring
├── negamax.h         # C negamax opponent with alpha-beta pruning
├── render.h          # Raylib rendering: board, pieces, animations
├── binding.c         # CPython bindings (env_binding.h pattern)
├── othello.py        # PufferEnv wrapper, obs/action spaces
├── curriculum.py     # Opponent scheduling: random → negamax(1,2,3,5) → self-play
├── train.py          # Training loop with PuffeRL + curriculum hooks
├── run_eval.py       # Evaluation with rendering
├── Makefile          # Compile C → shared lib
└── config.ini        # PufferLib hyperparameters
```

---

## Board Representation: Bitboards

Two `uint64_t` values encode the entire 8x8 board — one for black pieces, one for white. Enables fast move generation and flipping via bitwise operations.

### Bit-Index-to-Square Mapping

Bit `i` corresponds to row `i / 8`, column `i % 8`, where row 0 is the top row (rank 1) and column 0 is the left column (file a). This is row-major, top-left origin:

```
bit  0 = a1 (row 0, col 0)    bit  7 = h1 (row 0, col 7)
bit  8 = a2 (row 1, col 0)    bit 15 = h2 (row 1, col 7)
...
bit 56 = a8 (row 7, col 0)    bit 63 = h8 (row 7, col 7)
```

This mapping is used consistently across observations (planes 0-2), actions (index 0-63), and rendering.

Starting position:
- `black = 0x0000000810000000` (bits 28 and 35 → e4 and d5)
- `white = 0x0000001008000000` (bits 27 and 36 → d4 and e5)

---

## Observation Space

`Box(low=0, high=1, shape=(192,), dtype=float32)` — three 8x8 planes flattened:

| Plane | Indices | Content |
|-------|---------|---------|
| 0 | [0:64] | Current player's pieces (1.0 = piece, 0.0 = empty) |
| 1 | [64:128] | Opponent's pieces (1.0 = piece, 0.0 = empty) |
| 2 | [128:192] | Valid moves mask (1.0 = legal, 0.0 = illegal) |

Rationale: Separate player/opponent planes avoid sign ambiguity. Valid moves plane lets the network learn legality without trial-and-error. Matches AlphaGo/AlphaZero conventions.

---

## Action Space

`Discrete(65)` — 64 board squares (index 0-63) + 1 pass action (index 64).

Pass is required when no legal moves exist. Invalid moves (playing on a non-legal square) result in -1.0 reward and terminal=True (same penalty as PufferLib Connect4).

---

## Reward Structure (Sparse)

| Reward | Condition |
|--------|-----------|
| +1.0 | Win (more pieces when game ends) |
| -1.0 | Loss (fewer pieces) or invalid move |
| 0.0 | Draw (equal pieces) or in-progress |

No dense shaping. Lesson from flappy-rl: sparse rewards outperform dense rewards with survival/alignment/streak bonuses.

---

## Opponent Curriculum

Six phases, linearly ramped over 200M total timesteps. Depth 4 is skipped because the jump from d=3 (positional awareness) to d=5 (deep tactics) provides a more meaningful difficulty step than d=3→d=4→d=5.

| Phase | Opponent | Steps | Purpose |
|-------|----------|-------|---------|
| Random | Uniform random legal moves | 0-10M (0-5%) | Learn basic rules, piece flipping |
| Negamax d=1 | 1-ply greedy capture | 10-30M (5-15%) | Avoid obvious blunders |
| Negamax d=2 | 2-ply search | 30-60M (15-30%) | Think one move ahead |
| Negamax d=3 | 3-ply + alpha-beta | 60-100M (30-50%) | Positional strategy (corners, edges) |
| Negamax d=5 | 5-ply + alpha-beta + eval fn | 100-130M (50-65%) | Deep tactical play |
| Self-Play | Frozen policy pool | 130-200M (65-100%) | Novel strategies beyond negamax |

### Self-Play Mechanism

The opponent is sampled uniformly from a pool of the **last 5 frozen policy snapshots**, refreshed every 5M steps. This prevents overfitting to a single opponent and provides diversity in play styles. The pool starts with 1 snapshot and grows to 5 as training progresses.

### Curriculum-Environment Integration

The curriculum scheduler runs in `train.py` and communicates the current opponent configuration to environments via a `multiprocessing.Value` (shared float for difficulty level) plus a `multiprocessing.Array` (shared buffer for frozen policy weights during self-play). Opponent transitions happen only between episodes — never mid-game. The environment's `step()` checks the shared difficulty value at episode reset and selects the appropriate opponent for the new episode.

### Negamax Evaluation Function

For depths >= 3, uses classic positional weights:

```
+100  -20  +10   +5   +5  +10  -20  +100
 -20  -50   -2   -2   -2   -2  -50   -20
 +10   -2   -1   -1   -1   -1   -2   +10
  +5   -2   -1   -1   -1   -1   -2    +5
  +5   -2   -1   -1   -1   -1   -2    +5
 +10   -2   -1   -1   -1   -1   -2   +10
 -20  -50   -2   -2   -2   -2  -50   -20
+100  -20  +10   +5   +5  +10  -20  +100
```

Corners (+100) can't be flipped. X-squares (-50) adjacent to corners give them away. Edges (+5/10) are stable once captured.

---

## Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| optimizer | muon | PufferLib 3.0 default, outperforms Adam |
| total_timesteps | 200M | 2x flappy-rl; Othello has deeper strategy |
| learning_rate | 0.01 | High for Muon, cosine annealed |
| gamma | 0.99 | Long horizon for end-of-game rewards |
| use_rnn (LSTM) | True | Temporal reasoning across moves |
| hidden_size | 256 | Larger than flappy (128); more complex game |
| lstm_hidden_size | 256 | Match hidden size |
| num_envs (workers) | 8 | Matches Connect4 ocean config |
| sub-envs per worker | 512 | 4096 total parallel games |
| bptt_horizon | 32 | BPTT window; Othello games ~60 moves |
| ent_coef | 0.02 | Encourage exploration |
| minibatch_size | 32768 | Large batches for stability |
| anneal_lr | True | Cosine annealing |

### Policy Architecture

`MLP(192 → 256 → 256) + LSTM(256, 256) → action_head(65) + value_head(1)`

Approximate parameter count: MLP layers ~115K + LSTM ~525K + heads ~17K = **~657K parameters**. LSTM captures game phase (opening → midgame → endgame) and opponent patterns across moves.

PufferLib's `LSTMWrapper` handles LSTM hidden state management automatically — resetting states on episode boundaries and managing batched hidden states across all 4096 parallel environments via `bptt_horizon`.

---

## Raylib Rendering

Window: 600x700 pixels.

### Visual Elements

- **Board:** Green felt background (classic Othello), 8x8 grid with subtle cell borders
- **Pieces:** 3D-shaded circles — black with dark shadow, white with light shadow
- **Valid moves:** Semi-transparent rings on legal squares
- **Last move:** Gold border highlight on most recently placed piece
- **Flip animation:** Brief color transition (0.2s) when pieces are captured
- **HUD header:** Move counter, opponent type, piece count (Black vs White)
- **HUD footer:** Curriculum phase, running win rate, ESC to quit

---

## Evaluation Pipeline

| Mode | Command | Purpose |
|------|---------|---------|
| Visual | `python run_eval.py --render` | Watch agent play at 2 moves/sec |
| Headless | `python run_eval.py --episodes 100` | Batch win/loss/draw rates |
| Ladder | `python run_eval.py --ladder` | Test vs negamax d=1,2,3,5 for Elo estimate |
| Human | `python run_eval.py --human` | Click to play against the trained agent |

---

## Logging & Metrics

**Performance:** Win rate, average game length, piece differential, invalid move rate.

**Strategy indicators:** Corner capture rate, X-square avoidance rate, edge ownership %, mobility (avg legal moves).

**Training:** Policy/value loss, entropy, steps/second, LR schedule position.

**Curriculum:** Win rate per opponent type, phase transition timestamps, self-play opponent age, Elo estimate.

---

## C Binding API (`binding.c`)

The CPython extension exposes these functions to Python:

| Function | Signature | Purpose |
|----------|-----------|---------|
| `vec_init` | `(obs, actions, rewards, terminals, truncations, num_envs, seed) → c_envs` | Allocate N game structs, slice into shared numpy buffers |
| `vec_reset` | `(c_envs) → None` | Reset all environments to starting position |
| `vec_step` | `(c_envs) → None` | Step all envs using actions buffer, write obs/rewards/terminals in-place |
| `vec_render` | `(c_envs, env_idx) → None` | Render a specific env via raylib |
| `vec_close` | `(c_envs) → None` | Free all C memory |
| `negamax_move` | `(c_envs, env_idx, depth) → int` | Compute negamax best move for opponent in a specific env |
| `vec_log` | `(c_envs) → dict` | Aggregate stats (win_rate, avg_game_length, corner_captures, etc.) |

All board state lives in C. Python never touches bitboards directly — only the flattened observation planes in the shared numpy buffer.

---

## Color Assignment

The agent alternates between playing black and white each episode. On even episodes, agent plays black (moves first); on odd episodes, agent plays white. Observations are always encoded from the current player's perspective (plane 0 = my pieces, plane 1 = opponent pieces), so the network sees a consistent view regardless of color.

---

## Error Handling

- **Invalid moves:** -1.0 reward, terminal=True, auto-reset on next step. No C-level crash.
- **C errors:** `binding.c` validates all inputs (env_idx bounds, action range 0-64). Returns -1 error codes; Python raises `RuntimeError` with descriptive message.
- **Memory:** `vec_init` allocates, `vec_close` frees. Python `__del__` calls `vec_close` as safety net.
- **Negamax timeout:** For deep searches, negamax has a node count limit (1M nodes). If exceeded, returns the best move found so far.

---

## Build Requirements

```makefile
# Makefile
CC = gcc
CFLAGS = -O3 -shared -fPIC -Wall
RAYLIB_FLAGS = $(shell pkg-config --cflags --libs raylib)
PYTHON_FLAGS = $(shell python3-config --includes)
TARGET = binding$(shell python3-config --extension-suffix)

$(TARGET): binding.c othello.h negamax.h render.h
	$(CC) $(CFLAGS) $(PYTHON_FLAGS) $(RAYLIB_FLAGS) -o $@ binding.c
```

- **Compiler:** gcc or clang (C11)
- **Dependencies:** raylib >= 4.5 (via Homebrew on macOS: `brew install raylib`), Python 3.10+
- **Output:** `binding.cpython-3XX-darwin.so` (macOS) or `binding.cpython-3XX-linux-gnu.so` (Linux)
- **PufferLib:** `pip install pufferlib` (v3.0+)

---

## Testing

### C Unit Tests
- Move generation: verify legal moves for known positions (opening, corner scenarios, full board)
- Flipping: verify correct pieces flip in all 8 directions
- Terminal detection: verify game-over when board is full or both players pass
- Bitboard arithmetic: edge cases (row/column wrapping must NOT happen)

### Python Integration Tests
- Observation shape and bounds: verify (192,) float32 in [0, 1]
- Action validation: verify invalid moves produce -1.0 reward + terminal
- Pass action: verify pass is legal only when no other moves exist
- Episode lifecycle: verify reset produces valid starting position, step advances game state
- Reward correctness: verify +1/-1/0 at game end matches piece count

### Curriculum Tests
- Phase transitions: verify opponent type changes at correct step thresholds
- Self-play pool: verify snapshots accumulate and are sampled correctly
- Color alternation: verify agent plays both colors across episodes

---

## Key Design Decisions

1. **Bitboards over array** — O(1) move generation and flipping via bit manipulation, fits in CPU cache
2. **192-dim obs over 64-dim** — Three planes (mine/yours/legal) is cleaner than signed encoding
3. **Discrete(65) with pass** — Explicit pass action rather than auto-pass, gives the network agency
4. **Sparse rewards** — Learned from flappy-rl that dense shaping adds noise
5. **LSTM** — Learned from flappy-rl that temporal memory is essential for sequential games
6. **Hybrid C+Python** — C for hot path (board ops), Python for flexibility (curriculum, self-play)
7. **Frozen self-play pool** — 5 recent snapshots refreshed every 5M steps prevents overfitting to single opponent
8. **Color alternation** — Agent plays both black and white, matching AlphaZero convention
9. **Config format** — PufferLib INI format (matches ocean convention, e.g. `pufferlib/config/ocean/connect4.ini`)
