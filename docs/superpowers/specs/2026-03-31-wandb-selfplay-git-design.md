# Design: Wandb Observability, Self-Play Wiring, Hyperparameter Sweep, Git Hygiene

**Date:** 2026-03-31
**Status:** Approved

---

## 1. Problem Statement

Three gaps block production-ready training:

1. **No observability** — `train.py` only prints to stdout; no experiment tracking, no metric history, no sweep capability.
2. **Self-play not wired** — `CurriculumScheduler` snapshots policy weights into `SelfPlayPool` but `train.py` never loads them back; the C engine always plays a random opponent even during the self-play curriculum phase.
3. **Git state dirty** — `othello/uv.lock`, symlink resource dirs, and future wandb run dirs are untracked or need explicit gitignore rules.

---

## 2. Design

### 2.1 Self-Play Wiring (C + Python)

**Root cause:** `binding.c`'s `vec_step()` calls `play_random_opponent()` inline after every agent move — Python has no intercept point.

**Fix — split-step mode in `binding.c`:**

Add two new C functions alongside the existing `vec_step()`:

- **`vec_step_agent(actions)`** — applies only the agent's move for each env, then writes the opponent's board state into a new shared `opp_obs` numpy buffer (same dtype/shape as `obs`). Returns without applying opponent moves.
- **`vec_step_opponent(opp_actions)`** — applies the pre-computed opponent actions, runs the post-opponent checks (terminal, pass, etc.), writes agent observations + rewards + dones. Completes the step.

`vec_step()` is **unchanged** — all non-self-play phases continue using it.

**`train.py` rollout loop change:**

```
if opp_type == "self_play" and pool has snapshots:
    vec_step_agent(agent_actions)        # agent moves, fills opp_obs
    opp_actions = snapshot_policy(opp_obs)  # no_grad forward pass
    vec_step_opponent(opp_actions)       # opponent moves, fills agent obs/rewards
else:
    vec_step(agent_actions)              # existing path (random / negamax)
```

The snapshot policy is loaded from `SelfPlayPool.sample()` at the start of each update, refreshed via `maybe_refresh()`. If the pool is empty (first entry into self-play phase), fall back to `vec_step()` (random opponent) until a snapshot exists.

**`Othello` PufferEnv wrapper:** expose `opp_obs` buffer via `set_buffers` so `train.py` can read it directly without a copy.

### 2.2 Wandb Logging

**New CLI flags on `train.py`:**
- `--wandb` (bool toggle, default off)
- `--wandb-project` (str, default `"othello-rl"`)
- `--wandb-entity` (str, default `None` — uses wandb account default)

**`wandb.init()` call:** fires at training start when `--wandb` is set. Logs the full parsed config dict as `wandb.config`.

**Metrics logged every `log_interval` updates:**

| Key | Source |
|-----|--------|
| `loss/total`, `loss/policy`, `loss/value`, `loss/entropy` | PPO update |
| `train/clip_fraction` | PPO update |
| `train/learning_rate` | optimizer param group |
| `train/sps` | steps per second |
| `train/grad_norm` | before clip |
| `curriculum/phase` | 0–5 integer |
| `curriculum/opp_depth` | negamax depth or 0 |
| `episode/win_rate` | C engine `log()` |
| `episode/length` | C engine `log()` |
| `episode/invalid_moves` | C engine `log()` |
| `episode/corner_captures` | C engine `log()` |

Episode stats are logged whenever the C engine's `log()` call returns a non-empty dict (every `report_interval` steps inside the env).

**Checkpoints as wandb artifacts:** each saved `.pt` file is logged as a `model` artifact with metadata `{global_step, update}`.

### 2.3 Hyperparameter Sweep

**`sweep.yaml`** at project root:

```yaml
program: othello/train.py
method: bayes
metric:
  name: episode/win_rate
  goal: maximize
early_terminate:
  type: hyperband
  min_iter: 10
parameters:
  learning_rate:
    distribution: log_uniform_values
    min: 1e-4
    max: 1e-2
  gamma:
    values: [0.99, 0.995, 0.999]
  gae_lambda:
    distribution: uniform
    min: 0.9
    max: 0.99
  ent_coef:
    distribution: log_uniform_values
    min: 1e-4
    max: 1e-2
  bptt_horizon:
    values: [8, 16, 32]
  hidden_size:
    values: [128, 256, 512]
```

**Sweep integration in `train.py`:** when wandb is active and `wandb.config` contains sweep-injected keys, those values override `config.ini` values. This happens naturally since `wandb.config` is checked after `config.ini` is loaded.

**Usage:**
```bash
wandb sweep sweep.yaml          # registers sweep, prints sweep-id
wandb agent <entity>/<project>/<sweep-id>  # launches a worker
```

Multiple agents can run in parallel on different machines pointing at the same sweep.

### 2.4 Git Hygiene

**`.gitignore` additions:**
```
wandb/          # wandb local run dirs
othello/resources  # symlink into .venv
resources/      # symlink into .venv
othello/uv.lock  # not ours to track (pufferlib artifact)
```

**Track in git:**
- `sweep.yaml` — belongs in version control
- `othello/config.ini` — already committed
- `othello/train.py` — already committed

---

## 3. Files Changed

| File | Change |
|------|--------|
| `othello/binding.c` | Add `vec_step_agent()`, `vec_step_opponent()`, `opp_obs` buffer |
| `othello/Makefile` | Recompile after binding.c changes |
| `othello/othello.py` | Expose `opp_obs` buffer; add `step_agent()` / `step_opponent()` methods |
| `othello/train.py` | Wandb init/log, split-step self-play loop, sweep config override |
| `sweep.yaml` | New file — wandb sweep definition |
| `.gitignore` | Add `wandb/`, `othello/resources`, `resources/`, `othello/uv.lock` |

---

## 4. Out of Scope

- Tensorboard (wandb covers this)
- Multi-GPU / distributed training
- Modifying `run_eval.py` (evaluation against wandb-stored checkpoints is a separate task)
- Changing curriculum phase boundaries
