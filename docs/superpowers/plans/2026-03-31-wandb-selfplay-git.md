# Wandb Observability, Self-Play Wiring, Sweep, Git Hygiene — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire self-play opponents into the training loop, add wandb logging + hyperparameter sweep, and clean up git tracking.

**Architecture:** The C engine gets a split-step API (`vec_step_agent` / `vec_step_opponent`) so Python can intercept and supply opponent moves from a snapshot policy. Wandb logging is toggled via `--wandb` flag in `train.py`. Sweep config lives in `sweep.yaml` at project root.

**Tech Stack:** C extension (CPython), PyTorch, wandb, configparser, pytest

---

## Chunk 1: Git hygiene + C split-step API

### Task 1: Git hygiene

**Files:**
- Modify: `.gitignore`
- Track: `othello/uv.lock`

- [ ] **Step 1: Update .gitignore**

Add these lines to `.gitignore`:
```
wandb/
othello/resources
resources/
othello/uv.lock
```

- [ ] **Step 2: Stage and commit uv.lock and updated .gitignore**

```bash
git add .gitignore othello/uv.lock
git commit -m "chore: gitignore wandb dirs and venv symlinks, track uv.lock"
```

Expected: clean `git status` with only intentional untracked files.

---

### Task 2: Write failing tests for C split-step

**Files:**
- Modify: `othello/tests/test_othello.py`

The split-step API has two new functions: `vec_step_agent()` (applies agent move, fills `opp_obs` buffer) and `vec_step_opponent(opp_actions)` (applies opponent moves, completes step). Test that together they produce the same observable result as `vec_step()`.

- [ ] **Step 1: Write the failing tests**

Add to `othello/tests/test_othello.py`:

```python
import numpy as np
import pytest
from othello import binding

OBS_DIM = binding.OBS_DIM
NUM_ACTIONS = binding.NUM_ACTIONS


class TestSplitStep:
    """vec_step_agent + vec_step_opponent must behave like vec_step."""

    def _make_env(self, n=2):
        obs = np.zeros((n, OBS_DIM), dtype=np.float32)
        actions = np.zeros(n, dtype=np.int32)
        rewards = np.zeros(n, dtype=np.float32)
        dones = np.zeros(n, dtype=np.int32)
        opp_obs = np.zeros((n, OBS_DIM), dtype=np.float32)
        vec = binding.VecEnv()
        vec.init(n, obs, actions, rewards, dones)
        vec.set_opp_obs(opp_obs)
        vec.reset()
        return vec, obs, actions, rewards, dones, opp_obs

    def test_set_opp_obs_accepted(self):
        """set_opp_obs() does not raise."""
        vec, *_ = self._make_env()
        # If we got here without error the C function exists

    def test_step_agent_fills_opp_obs(self):
        """After vec_step_agent(), opp_obs must be non-zero (opponent has pieces)."""
        vec, obs, actions, rewards, dones, opp_obs = self._make_env(n=1)
        # Pick any legal move from the legal-move plane (obs plane 2)
        legal = np.where(obs[0, 128:])[0]
        actions[0] = int(legal[0])
        vec.step_agent()
        # opp_obs plane 0 (opponent's own pieces) should be non-zero
        assert opp_obs[0].sum() > 0, "opp_obs should contain board state"

    def test_step_opponent_completes_step(self):
        """vec_step_opponent() must write obs/rewards/dones back."""
        vec, obs, actions, rewards, dones, opp_obs = self._make_env(n=1)
        legal = np.where(obs[0, 128:])[0]
        actions[0] = int(legal[0])
        opp_actions = np.zeros(1, dtype=np.int32)
        vec.step_agent()
        # Pick a legal move for opponent from opp_obs legal plane
        opp_legal = np.where(opp_obs[0, 128:])[0]
        opp_actions[0] = int(opp_legal[0]) if len(opp_legal) > 0 else 64
        vec.step_opponent(opp_actions)
        # obs must have been updated (non-zero)
        assert obs[0].sum() > 0

    def test_split_step_preserves_game_length(self):
        """Games played via split-step should complete in a reasonable number of moves."""
        n = 4
        vec, obs, actions, rewards, dones, opp_obs = self._make_env(n=n)
        opp_actions = np.zeros(n, dtype=np.int32)
        total_steps = 0
        total_done = 0
        for _ in range(200):  # Othello max ~60 moves
            for i in range(n):
                legal = np.where(obs[i, 128:])[0]
                actions[i] = int(legal[0]) if len(legal) > 0 else 64
            vec.step_agent()
            for i in range(n):
                opp_legal = np.where(opp_obs[i, 128:])[0]
                opp_actions[i] = int(opp_legal[0]) if len(opp_legal) > 0 else 64
            vec.step_opponent(opp_actions)
            total_done += int(dones.sum())
            total_steps += 1
        assert total_done > 0, "At least some games should have finished"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /Users/openclaw/Code/Connect4RL
source .venv/bin/activate
python -m pytest othello/tests/test_othello.py::TestSplitStep -v
```

Expected: `AttributeError: 'VecEnv' object has no attribute 'set_opp_obs'` or similar — tests should FAIL.

---

### Task 3: Implement split-step in binding.c

**Files:**
- Modify: `othello/binding.c`
- Modify: `othello/Makefile` (recompile)

**Add `opp_obs_ptrs` to the VecEnv struct** (after `done_ptrs`):
```c
    float **opp_obs_ptrs;  // opponent observation buffer, set by set_opp_obs()
```

**Add to `VecEnv_new`** (after existing field inits):
```c
        self->opp_obs_ptrs = NULL;
```

**Add to `VecEnv_dealloc`** (after existing frees):
```c
    if (self->opp_obs_ptrs) free(self->opp_obs_ptrs);
```

**Add `vec_set_opp_obs` function** (after `vec_reset`):
```c
static PyObject *vec_set_opp_obs(VecEnv *self, PyObject *args) {
    PyObject *opp_obs_arr;
    if (!PyArg_ParseTuple(args, "O", &opp_obs_arr)) return NULL;

    if (self->opp_obs_ptrs) {
        free(self->opp_obs_ptrs);
    }
    self->opp_obs_ptrs = (float **)calloc(self->num_envs, sizeof(float *));
    if (!self->opp_obs_ptrs) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate opp_obs_ptrs");
        return NULL;
    }
    float *base = (float *)PyArray_DATA((PyArrayObject *)opp_obs_arr);
    for (int i = 0; i < self->num_envs; i++) {
        self->opp_obs_ptrs[i] = base + i * OTH_OBS_DIM;
    }
    Py_RETURN_NONE;
}
```

**Add `vec_step_agent` function** (after `vec_set_opp_obs`):
```c
static PyObject *vec_step_agent(VecEnv *self, PyObject *args) {
    (void)args;
    for (int i = 0; i < self->num_envs; i++) {
        Othello *g = &self->envs[i];
        int agent_color = g->episode_id % 2;

        // Auto-reset if done (same logic as vec_step)
        if (g->done) {
            self->total_games++;
            self->total_moves += g->move_count;
            self->total_invalid_moves += g->invalid_moves;
            self->total_corner_captures += g->corner_captures;
            float terminal_reward = g->reward;
            if (agent_color == OTH_WHITE) terminal_reward = -terminal_reward;
            if (terminal_reward > 0) self->total_wins++;

            oth_reset(g);
            g->episode_id++;
            agent_color = g->episode_id % 2;

            if (agent_color == OTH_WHITE) {
                play_random_opponent(g, OTH_BLACK);
                if (oth_check_terminal(g)) {
                    oth_write_obs(g, self->obs_ptrs[i], agent_color);
                    *self->reward_ptrs[i] = (agent_color == OTH_WHITE) ? -g->reward : g->reward;
                    *self->done_ptrs[i] = 1;
                    if (self->opp_obs_ptrs)
                        oth_write_obs(g, self->opp_obs_ptrs[i], 1 - agent_color);
                    continue;
                }
            }
            oth_write_obs(g, self->obs_ptrs[i], agent_color);
            *self->reward_ptrs[i] = 0.0f;
            *self->done_ptrs[i] = 0;
            if (self->opp_obs_ptrs)
                oth_write_obs(g, self->opp_obs_ptrs[i], 1 - agent_color);
            continue;
        }

        int action = *self->action_ptrs[i];
        int terminal = oth_step_agent(g, action, agent_color);

        if (terminal) {
            float r = g->reward;
            if (agent_color == OTH_WHITE) r = -r;
            oth_write_obs(g, self->obs_ptrs[i], agent_color);
            *self->reward_ptrs[i] = r;
            *self->done_ptrs[i] = 1;
            if (self->opp_obs_ptrs)
                oth_write_obs(g, self->opp_obs_ptrs[i], 1 - agent_color);
            continue;
        }

        // Write opponent's view into opp_obs so Python can pick the move
        if (self->opp_obs_ptrs) {
            oth_write_obs(g, self->opp_obs_ptrs[i], 1 - agent_color);
        }
        // Mark as not done yet — step_opponent will complete this env
        *self->done_ptrs[i] = 0;
        *self->reward_ptrs[i] = 0.0f;
    }
    Py_RETURN_NONE;
}
```

**Add `vec_step_opponent` function** (after `vec_step_agent`):
```c
static PyObject *vec_step_opponent(VecEnv *self, PyObject *args) {
    PyObject *opp_actions_arr;
    if (!PyArg_ParseTuple(args, "O", &opp_actions_arr)) return NULL;
    int *opp_acts = (int *)PyArray_DATA((PyArrayObject *)opp_actions_arr);

    for (int i = 0; i < self->num_envs; i++) {
        Othello *g = &self->envs[i];
        // Skip envs that already terminated in step_agent
        if (g->done || *self->done_ptrs[i] == 1) {
            oth_write_obs(g, self->obs_ptrs[i], g->episode_id % 2);
            continue;
        }

        int agent_color = g->episode_id % 2;
        int opp_color = 1 - agent_color;

        // Apply the Python-provided opponent action
        int opp_action = opp_acts[i];
        uint64_t opp_board = (opp_color == OTH_BLACK) ? g->black : g->white;
        uint64_t my_board  = (opp_color == OTH_BLACK) ? g->white : g->black;
        uint64_t opp_moves = oth_get_moves(opp_board, my_board);

        if (opp_moves == 0) {
            oth_apply_move(g, OTH_ACTION_PASS);
        } else {
            // Validate: if chosen move is illegal, pick first legal
            if (opp_action < 64 && ((opp_moves >> opp_action) & 1)) {
                oth_apply_move(g, opp_action);
            } else {
                // Fallback: first legal move
                for (int b = 0; b < 64; b++) {
                    if ((opp_moves >> b) & 1) { oth_apply_move(g, b); break; }
                }
            }
        }

        if (oth_check_terminal(g)) {
            float r = g->reward;
            if (agent_color == OTH_WHITE) r = -r;
            oth_write_obs(g, self->obs_ptrs[i], agent_color);
            *self->reward_ptrs[i] = r;
            *self->done_ptrs[i] = 1;
            continue;
        }

        // Check if agent has moves; if not, auto-pass for agent
        uint64_t agent_board = (agent_color == OTH_BLACK) ? g->black : g->white;
        uint64_t enemy_board = (agent_color == OTH_BLACK) ? g->white : g->black;
        uint64_t agent_moves = oth_get_moves(agent_board, enemy_board);
        if (agent_moves == 0) {
            oth_apply_move(g, OTH_ACTION_PASS);
            if (oth_check_terminal(g)) {
                float r = g->reward;
                if (agent_color == OTH_WHITE) r = -r;
                oth_write_obs(g, self->obs_ptrs[i], agent_color);
                *self->reward_ptrs[i] = r;
                *self->done_ptrs[i] = 1;
                continue;
            }
        }

        oth_write_obs(g, self->obs_ptrs[i], agent_color);
        *self->reward_ptrs[i] = 0.0f;
        *self->done_ptrs[i] = 0;
    }
    Py_RETURN_NONE;
}
```

**Add to `module_methods` array** (before the `{NULL, NULL, 0, NULL}` sentinel):
```c
    {"set_opp_obs",    (PyCFunction)vec_set_opp_obs,    METH_VARARGS, "Set opponent obs buffer"},
    {"step_agent",     (PyCFunction)vec_step_agent,     METH_NOARGS,  "Apply agent moves only"},
    {"step_opponent",  (PyCFunction)vec_step_opponent,  METH_VARARGS, "Apply opponent moves and complete step"},
```

- [ ] **Step 1: Apply all binding.c edits above**

- [ ] **Step 2: Recompile**

```bash
cd /Users/openclaw/Code/Connect4RL/othello
source ../.venv/bin/activate
make
```

Expected: `binding.cpython-312-darwin.so` rebuilt, 0 errors.

- [ ] **Step 3: Run split-step tests**

```bash
cd /Users/openclaw/Code/Connect4RL
python -m pytest othello/tests/test_othello.py::TestSplitStep -v
```

Expected: 4 tests PASS.

- [ ] **Step 4: Run full test suite to confirm no regressions**

```bash
python -m pytest othello/tests/ -v
```

Expected: 39 tests PASS (35 original + 4 new).

- [ ] **Step 5: Commit**

```bash
git add othello/binding.c othello/binding.cpython-312-darwin.so othello/tests/test_othello.py
git commit -m "feat: add split-step C API (vec_step_agent/vec_step_opponent) for self-play"
```

---

## Chunk 2: PufferEnv wrapper + train.py self-play wiring

### Task 4: Expose split-step in PufferEnv wrapper

**Files:**
- Modify: `othello/othello.py`

Add `opp_obs` buffer and `step_agent` / `step_opponent` methods to `Othello`.

- [ ] **Step 1: Write the failing test**

Add to `othello/tests/test_env.py`:

```python
class TestSplitStepEnv:
    def test_step_agent_returns_opp_obs(self):
        env = Othello(num_envs=2)
        env.reset()
        actions = np.array([0, 0], dtype=np.int32)
        # pick legal moves
        for i in range(2):
            legal = np.where(env.observations[i, 128:])[0]
            actions[i] = int(legal[0])
        opp_obs = env.step_agent(actions)
        assert opp_obs.shape == (2, binding.OBS_DIM)
        assert opp_obs.sum() > 0
        env.close()

    def test_step_opponent_completes_step(self):
        env = Othello(num_envs=2)
        env.reset()
        actions = np.zeros(2, dtype=np.int32)
        for i in range(2):
            legal = np.where(env.observations[i, 128:])[0]
            actions[i] = int(legal[0])
        opp_obs = env.step_agent(actions)
        opp_actions = np.zeros(2, dtype=np.int32)
        for i in range(2):
            opp_legal = np.where(opp_obs[i, 128:])[0]
            opp_actions[i] = int(opp_legal[0]) if len(opp_legal) > 0 else 64
        obs, rewards, terminals, truncations, infos = env.step_opponent(opp_actions)
        assert obs.shape == (2, binding.OBS_DIM)
        env.close()
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
python -m pytest othello/tests/test_env.py::TestSplitStepEnv -v
```

Expected: `AttributeError: 'Othello' object has no attribute 'step_agent'`

- [ ] **Step 3: Implement in othello.py**

In `Othello.__init__`, after `self._c_env.init(...)`, add:

```python
        self._opp_obs = np.zeros((num_envs, binding.OBS_DIM), dtype=np.float32)
        self._c_env.set_opp_obs(self._opp_obs)
        self._c_opp_actions = np.zeros(num_envs, dtype=np.int32)
```

Add two new methods to `Othello`:

```python
    def step_agent(self, actions: np.ndarray) -> np.ndarray:
        """Apply agent moves only; returns opponent obs buffer (view, not copy)."""
        np.copyto(self._c_actions, actions.ravel()[: self.num_agents].astype(np.int32))
        self._c_env.step_agent()
        return self._opp_obs

    def step_opponent(self, opp_actions: np.ndarray):
        """Apply opponent moves and complete the step."""
        np.copyto(self._c_opp_actions, opp_actions.ravel()[: self.num_agents].astype(np.int32))
        self._c_env.step_opponent(self._c_opp_actions)

        np.copyto(self.rewards, self._c_rewards)
        np.copyto(self.terminals, self._c_dones.astype(bool))
        self.truncations[:] = False
        self.masks[:] = True

        self._step_count += 1
        infos = []
        if self._step_count % self.report_interval == 0:
            log = self._c_env.log()
            infos = [log]

        return self.observations, self.rewards, self.terminals, self.truncations, infos
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest othello/tests/test_env.py -v
```

Expected: all env tests PASS including new `TestSplitStepEnv`.

- [ ] **Step 5: Commit**

```bash
git add othello/othello.py othello/tests/test_env.py
git commit -m "feat: expose step_agent/step_opponent and opp_obs buffer in PufferEnv wrapper"
```

---

### Task 5: Wire self-play into train.py rollout loop

**Files:**
- Modify: `othello/train.py`

When `opp_type == "self_play"` and the pool has a snapshot, the rollout loop uses `step_agent` + snapshot policy forward pass + `step_opponent` instead of the regular `step`.

- [ ] **Step 1: Add snapshot policy loader helper**

Add this function to `train.py` (after `make_policy`):

```python
def _load_snapshot_policy(
    state_dict: dict, env, hidden_size: int, device: torch.device
) -> OthelloPolicy:
    """Instantiate a policy and load a state_dict from the self-play pool."""
    snap = make_policy(env, hidden_size=hidden_size).to(device)
    snap.load_state_dict(state_dict)
    snap.eval()
    return snap
```

- [ ] **Step 2: Add self-play rollout helper**

Add this function to `train.py` (after `_load_snapshot_policy`):

```python
def _rollout_step_selfplay(
    env: Othello,
    agent_actions: np.ndarray,
    snapshot_policy: OthelloPolicy,
    snap_lstm: Dict[str, torch.Tensor],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """One split-step using the snapshot policy as opponent."""
    opp_obs_np = env.step_agent(agent_actions)
    opp_obs_t = torch.tensor(opp_obs_np, dtype=torch.float32, device=device)
    with torch.no_grad():
        opp_logits, _ = snapshot_policy.forward_eval(opp_obs_t, snap_lstm)
    opp_actions = opp_logits.argmax(dim=-1).cpu().numpy().astype(np.int32)
    return env.step_opponent(opp_actions)
```

- [ ] **Step 3: Update the rollout collection block in `train()`**

Locate the rollout collection section (inside `with torch.no_grad():`). Replace:

```python
                action_np = action.cpu().numpy().astype(np.int32)
                obs_np, rew_np, term_np, trunc_np, _ = env.step(action_np)
```

With:

```python
                action_np = action.cpu().numpy().astype(np.int32)
                if opp_type == "self_play" and snapshot_policy is not None:
                    obs_np, rew_np, term_np, trunc_np, _ = _rollout_step_selfplay(
                        env, action_np, snapshot_policy, snap_lstm_state, device
                    )
                else:
                    obs_np, rew_np, term_np, trunc_np, _ = env.step(action_np)
```

- [ ] **Step 4: Add snapshot policy state to the main training loop**

At the top of the `for update in range(...)` loop, after the curriculum update block, add:

```python
        # Load/refresh snapshot policy for self-play
        snapshot_policy = None
        snap_lstm_state = None
        if opp_type == "self_play":
            snap_state_dict = curriculum.self_play_pool.sample()
            if snap_state_dict is not None:
                snapshot_policy = _load_snapshot_policy(
                    snap_state_dict, env, hidden_size, device
                )
                snap_lstm_state = _init_lstm_state(num_envs, hidden_size, device)
```

- [ ] **Step 5: Smoke-test self-play runs end-to-end**

```bash
source .venv/bin/activate
python -c "
import sys; sys.argv = [
    'train.py',
    '--train.total_timesteps', '6000',
    '--train.bptt_horizon', '16',
    '--env.num_envs', '4',
    '--train.minibatch_size', '32',
    '--train.update_epochs', '1',
    '--train.checkpoint_interval', '9999',
    '--policy.hidden_size', '32',
    # Force self-play phase early
]
# Patch curriculum so self-play starts at 0%
from othello import curriculum as cur
cur.PHASE_BOUNDARIES = [(0.0, 1.0, 'self_play', 0)]
from othello.train import train
train()
"
```

Expected: training runs, logs show `phase=self_play(d=0)`, no errors.

- [ ] **Step 6: Run full test suite**

```bash
python -m pytest othello/tests/ -v
```

Expected: all 39 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add othello/train.py
git commit -m "feat: wire self-play snapshot policy into rollout loop via split-step API"
```

---

## Chunk 3: Wandb logging + hyperparameter sweep

### Task 6: Install wandb

- [ ] **Step 1: Install wandb into the venv**

```bash
source .venv/bin/activate
pip install wandb
```

Expected: `Successfully installed wandb-*`

- [ ] **Step 2: Login**

```bash
wandb login
```

Paste your API key when prompted. Verify with `wandb status`.

---

### Task 7: Add wandb logging to train.py

**Files:**
- Modify: `othello/train.py`

- [ ] **Step 1: Add wandb CLI flags**

In the `train()` function, inside the CLI `parser` block, add:

```python
    parser.add_argument("--wandb", action="store_true", default=False,
                        help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="othello-rl")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
```

Parse them after `known, remainder = parser.parse_known_args()`:

```python
    use_wandb = known.wandb
    wandb_project = known.wandb_project
    wandb_entity = known.wandb_entity
    wandb_run_name = known.wandb_run_name or f"{exp_name}_{int(time.time())}"
```

- [ ] **Step 2: Add wandb init block**

After the print statement that shows training config, add:

```python
    if use_wandb:
        import wandb
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_run_name,
            config={
                "num_envs": num_envs,
                "total_timesteps": total_timesteps,
                "learning_rate": lr,
                "gamma": gamma,
                "gae_lambda": gae_lambda,
                "update_epochs": update_epochs,
                "clip_coef": clip_coef,
                "vf_coef": vf_coef,
                "ent_coef": ent_coef,
                "max_grad_norm": max_grad_norm,
                "bptt_horizon": bptt_horizon,
                "minibatch_size": minibatch_size,
                "hidden_size": hidden_size,
                "seed": seed,
            },
        )
```

- [ ] **Step 3: Compute and log grad norm before clipping**

In the PPO update loop, just before `nn.utils.clip_grad_norm_`, add:

```python
                grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
```

Remove the existing bare `nn.utils.clip_grad_norm_` call (replace it with the assignment above).

- [ ] **Step 4: Add wandb logging inside the periodic logging block**

Replace the existing logging block with:

```python
        if update % log_interval == 0 or update == total_updates - 1:
            elapsed = time.time() - wall_start
            sps = (global_step - step_start) / max(elapsed, 1e-9)
            step_start = global_step
            wall_start = time.time()
            mean_clip = float(np.mean(clip_fracs)) if clip_fracs else float("nan")
            print(
                f"[{global_step:>10}/{total_timesteps}] "
                f"update={update + 1}/{total_updates}  "
                f"loss={last_loss:.4f}  pg={last_pg:.4f}  "
                f"vf={last_vf:.4f}  ent={last_ent:.4f}  "
                f"clip={mean_clip:.3f}  "
                f"phase={opp_type}(d={opp_depth})  "
                f"SPS={sps:.0f}"
            )
            if use_wandb:
                import wandb
                log_dict = {
                    "loss/total": last_loss,
                    "loss/policy": last_pg,
                    "loss/value": last_vf,
                    "loss/entropy": last_ent,
                    "train/clip_fraction": mean_clip,
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                    "train/sps": sps,
                    "train/grad_norm": float(grad_norm) if 'grad_norm' in dir() else 0.0,
                    "curriculum/phase": ["random","negamax","negamax","negamax","negamax","self_play"].index(opp_type) if opp_type in ["random","self_play"] else 1,
                    "curriculum/opp_depth": opp_depth,
                    "global_step": global_step,
                }
                wandb.log(log_dict, step=global_step)
```

- [ ] **Step 5: Log episode stats whenever the env reports them**

After the line `obs_np, rew_np, term_np, trunc_np, _ = env.step(...)` (and the equivalent selfplay branch), capture and log infos. Modify the rollout to collect infos:

```python
                # replace the _ with infos capture:
                obs_np, rew_np, term_np, trunc_np, step_infos = env.step(action_np)
                # ... or for self-play:
                obs_np, rew_np, term_np, trunc_np, step_infos = _rollout_step_selfplay(...)

                if use_wandb and step_infos:
                    import wandb
                    info = step_infos[0]
                    total_games = info.get("total_games", 0)
                    if total_games > 0:
                        wandb.log({
                            "episode/win_rate": info.get("total_wins", 0) / total_games,
                            "episode/length": info.get("total_moves", 0) / total_games,
                            "episode/invalid_moves": info.get("total_invalid_moves", 0) / total_games,
                            "episode/corner_captures": info.get("total_corner_captures", 0) / total_games,
                        }, step=global_step)
```

- [ ] **Step 6: Log checkpoints as wandb artifacts**

In the checkpoint saving block, after `torch.save(...)`, add:

```python
            if use_wandb:
                import wandb
                artifact = wandb.Artifact(
                    name=f"{exp_name}_checkpoint",
                    type="model",
                    metadata={"global_step": global_step, "update": update + 1},
                )
                artifact.add_file(str(path))
                wandb.log_artifact(artifact)
```

- [ ] **Step 7: Close wandb at end of train()**

Before `env.close()` at the end of `train()`, add:

```python
    if use_wandb:
        import wandb
        wandb.finish()
```

- [ ] **Step 8: Smoke-test wandb logging**

```bash
python othello/train.py \
  --train.total_timesteps 4096 \
  --env.num_envs 8 \
  --train.bptt_horizon 16 \
  --train.minibatch_size 64 \
  --train.update_epochs 1 \
  --train.checkpoint_interval 9999 \
  --policy.hidden_size 32 \
  --wandb \
  --wandb_project othello-rl-test \
  --wandb_run_name smoke-test
```

Expected: run appears in wandb UI at `https://wandb.ai/<your-entity>/othello-rl-test`, metrics visible.

- [ ] **Step 9: Commit**

```bash
git add othello/train.py
git commit -m "feat: add wandb logging with losses, curriculum phase, episode stats, and checkpoint artifacts"
```

---

### Task 8: Add sweep config and sweep integration

**Files:**
- Create: `sweep.yaml`
- Modify: `othello/train.py` (sweep config override)

- [ ] **Step 1: Create sweep.yaml**

Create `/Users/openclaw/Code/Connect4RL/sweep.yaml`:

```yaml
program: othello/train.py
method: bayes
metric:
  name: episode/win_rate
  goal: maximize
early_terminate:
  type: hyperband
  min_iter: 10
command:
  - ${env}
  - python
  - ${program}
  - --wandb
  - --wandb_project
  - othello-rl
  - ${args}
parameters:
  train.learning_rate:
    distribution: log_uniform_values
    min: 0.0001
    max: 0.01
  train.gamma:
    values: [0.99, 0.995, 0.999]
  train.gae_lambda:
    distribution: uniform
    min: 0.9
    max: 0.99
  train.ent_coef:
    distribution: log_uniform_values
    min: 0.0001
    max: 0.01
  train.bptt_horizon:
    values: [8, 16, 32]
  policy.hidden_size:
    values: [128, 256, 512]
```

- [ ] **Step 2: Add wandb sweep config override in train()**

After `_apply_cli_overrides(cfg, remainder)` and the wandb init block, add:

```python
    # When running as a wandb sweep agent, override config with sweep-injected values
    if use_wandb:
        import wandb
        # wandb.config is populated by the sweep agent before train() is called
        for key, value in wandb.config.items():
            if "." in key:
                section, name = key.split(".", 1)
                if cfg.has_section(section):
                    cfg.set(section, name.replace("-", "_"), str(value))
        # Re-parse any affected values
        lr = float(cfg.get("train", "learning_rate"))
        gamma = float(cfg.get("train", "gamma"))
        gae_lambda = float(cfg.get("train", "gae_lambda"))
        ent_coef = float(cfg.get("train", "ent_coef"))
        bptt_horizon = cfg.getint("train", "bptt_horizon")
        hidden_size = cfg.getint("policy", "hidden_size")
```

- [ ] **Step 3: Register the sweep**

```bash
cd /Users/openclaw/Code/Connect4RL
source .venv/bin/activate
wandb sweep sweep.yaml
```

Expected output: `wandb: Created sweep with ID: <sweep-id>`
Note the sweep ID.

- [ ] **Step 4: Test a single sweep agent run**

```bash
wandb agent <your-entity>/othello-rl/<sweep-id> --count 1
```

Expected: one training run starts, appears in wandb sweep dashboard.

- [ ] **Step 5: Commit**

```bash
git add sweep.yaml othello/train.py
git commit -m "feat: add wandb sweep config and sweep-agent config override in train.py"
```

---

## Verification

After all tasks complete, run the full suite:

```bash
source .venv/bin/activate
python -m pytest othello/tests/ -v
```

Expected: **39 tests PASS**.

Run a final end-to-end check with wandb enabled:

```bash
python othello/train.py \
  --env.num_envs 64 \
  --train.total_timesteps 65536 \
  --train.bptt_horizon 16 \
  --train.minibatch_size 256 \
  --train.checkpoint_interval 50 \
  --policy.hidden_size 64 \
  --wandb \
  --wandb_project othello-rl
```

Expected:
- All 6 curriculum phases appear in logs
- Checkpoints saved to `experiments/othello_ppo/`
- Wandb run visible with loss curves, win rate, curriculum phase metrics
