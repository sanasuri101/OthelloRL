# Design: Eval Script + GitHub CI

**Date:** 2026-03-31

## 1. Eval Script (`othello/eval.py`)

### Purpose
Benchmark a trained policy against fixed negamax opponents at multiple depths. Provides a curriculum-independent signal of actual agent strength.

### Interface
```python
evaluate(policy, depths=[1,2,3,5], n_games=100, device="cpu") -> dict[int, float]
# Returns {depth: win_rate} e.g. {1: 0.72, 2: 0.51, 3: 0.33, 5: 0.18}
```

### Integration with train.py
- `eval_interval = 50_000` steps (added to `config.ini` under `[train]`)
- Called every `eval_interval` global steps during training
- Logs to wandb as `eval/win_rate_vs_negamax_d1`, `eval/win_rate_vs_negamax_d3`, etc.
- No-grad, greedy (argmax) policy during eval

### Standalone CLI
```bash
python othello/eval.py --checkpoint model.pt [--wandb] [--depths 1,2,3,5] [--n_games 100]
```

### Implementation notes
- Uses existing `step_negamax()` split-step path
- Single-env evaluation (n_games sequential games) — no vectorization needed
- LSTM state zeroed at each episode start
- Reports win_rate per depth + overall summary to stdout

## 2. GitHub + CI

### Remote
`https://github.com/sanasuri101/OthelloRL` — push existing master branch.

### GitHub Actions workflow (`.github/workflows/ci.yml`)
Triggers: push to `master`, PRs to `master`

Steps:
1. Checkout
2. Install `uv` (official installer action)
3. `uv sync` — install Python deps
4. `make` — recompile C extension from source
5. `uv run python -m pytest othello/tests/ -v`

### Key decisions
- `.so` stays gitignored — CI always rebuilds from source
- Python 3.12 (matches local dev)
- No matrix build — single platform (ubuntu-latest)
- macOS-specific Makefile flags (`-arch arm64`, `pkg-config raylib`) need Linux equivalents

