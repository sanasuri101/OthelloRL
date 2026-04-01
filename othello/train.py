"""PPO training script for the Othello RL environment.

Standalone CleanRL-style PPO with LSTM policy, curriculum learning, and
checkpointing.  Does not depend on pufferlib.pufferl (which requires a
torch/pufferlib version that supports torch.uint64).

Usage
-----
    # Train with default config.ini
    python train.py

    # Override individual settings
    python train.py --train.total_timesteps 5000000 --env.num_envs 256

    # Resume from a checkpoint
    python train.py --load_checkpoint experiments/othello_ppo/checkpoint_000200.pt

Config
------
All settings are read from config.ini in the same directory.  CLI flags of
the form ``--section.key value`` override the file values.

Checkpoints
-----------
Saved to ``{data_dir}/{exp_name}/checkpoint_{update:06d}.pt``.  Each file
contains::

    {
        "global_step": int,
        "policy_state_dict": OrderedDict,
        "optimizer_state_dict": dict,
    }
"""

from __future__ import annotations

import argparse
import configparser
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# ---------------------------------------------------------------------------
# Make `othello` importable when running from the project root or from inside
# the othello/ directory.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pufferlib

from othello.othello import Othello
from othello.curriculum import CurriculumScheduler
from othello.eval import evaluate as _evaluate

# ---------------------------------------------------------------------------
# Built-in config defaults (mirror config.ini so the script runs without it)
# ---------------------------------------------------------------------------
_CONFIG_DEFAULTS: dict = {
    "env": {
        "num_envs": "512",
    },
    "train": {
        "exp_name": "othello_ppo",
        "data_dir": "experiments",
        "seed": "42",
        "torch_deterministic": "True",
        "device": "cpu",
        "total_timesteps": "150000000",
        "learning_rate": "3e-4",
        "gamma": "0.995",
        "gae_lambda": "0.95",
        "update_epochs": "4",
        "clip_coef": "0.2",
        "vf_coef": "1.0",
        "ent_coef": "0.01",
        "max_grad_norm": "0.5",
        "anneal_lr": "True",
        "bptt_horizon": "32",
        "minibatch_size": "2048",
        "checkpoint_interval": "200",
        "eval_interval": "1000000",
        "eval_n_games": "200",
    },
    "policy": {
        "hidden_size": "256",
    },
}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _load_config(config_path: Optional[str] = None) -> configparser.ConfigParser:
    """Read config.ini, seeding with built-in defaults first."""
    cfg = configparser.ConfigParser()
    for section, values in _CONFIG_DEFAULTS.items():
        cfg[section] = values

    path = config_path or str(_SCRIPT_DIR / "config.ini")
    if os.path.exists(path):
        cfg.read(path)
    return cfg


def _apply_cli_overrides(
    cfg: configparser.ConfigParser, remainder: list[str]
) -> None:
    """Apply ``--section.key value`` overrides parsed from argv."""
    i = 0
    while i < len(remainder):
        flag = remainder[i]
        if not flag.startswith("--"):
            i += 1
            continue
        key = flag[2:].replace("-", "_")
        if "." not in key:
            i += 1
            continue
        section, name = key.split(".", 1)
        if i + 1 < len(remainder) and not remainder[i + 1].startswith("--"):
            value = remainder[i + 1]
            i += 2
        else:
            value = "True"
            i += 1
        if not cfg.has_section(section):
            cfg.add_section(section)
        cfg.set(section, name, value)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


class OthelloPolicy(nn.Module):
    """LSTM-backed actor-critic policy for Othello.

    Self-contained: does not require pufferlib.models (which has a
    torch.uint64 incompatibility on some torch versions).

    forward_eval(obs, state) → (logits, value)
      • obs:   (B, OBS_DIM) float32
      • state: dict with "lstm_h" and "lstm_c", each (1, B, H)
      Returns logits (B, NUM_ACTIONS), value (B, 1).

    Mutates state in-place so the caller's dict always holds the
    latest hidden state.
    """

    def __init__(self, env, hidden_size: int = 256) -> None:
        super().__init__()
        obs_size = int(np.prod(env.single_observation_space.shape))
        act_size = int(env.single_action_space.n)
        self.hidden_size = hidden_size

        self.encoder = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.lstm_cell = nn.LSTMCell(hidden_size, hidden_size)
        self.actor = nn.Linear(hidden_size, act_size)
        self.critic = nn.Linear(hidden_size, 1)

        # Orthogonal init on actor/critic heads
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward_eval(
        self,
        obs: torch.Tensor,
        state: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single-step inference (uses LSTMCell for efficiency)."""
        encoded = self.encoder(obs.float())  # (B, H)

        h = state["lstm_h"].squeeze(0)  # (B, H)
        c = state["lstm_c"].squeeze(0)  # (B, H)
        h, c = self.lstm_cell(encoded, (h, c))

        state["lstm_h"] = h.unsqueeze(0)
        state["lstm_c"] = c.unsqueeze(0)

        return self.actor(h), self.critic(h)


def make_policy(env, hidden_size: int = 256) -> OthelloPolicy:
    """Construct an LSTM-backed policy for the given env."""
    return OthelloPolicy(env, hidden_size=hidden_size)


def _load_snapshot_policy(
    state_dict: dict, env, hidden_size: int, device: torch.device
) -> OthelloPolicy:
    """Instantiate a policy and load a state_dict from the self-play pool."""
    snap = make_policy(env, hidden_size=hidden_size).to(device)
    snap.load_state_dict(state_dict)
    snap.eval()
    return snap


def _apply_action_mask(logits: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    """Mask logits for illegal actions using the legal-move plane in obs.

    obs layout: [my_pieces(64) | opp_pieces(64) | legal_squares(64)] — 192 dims total.
    Action space: 0-63 = board squares, 64 = pass.
    Pass (action 64) is legal only when no squares are legal.
    """
    legal = obs[:, 128:192]  # (B, 64) — 1.0 if that square is a legal move
    has_legal = legal.sum(dim=-1, keepdim=True) > 0  # (B, 1) bool
    pass_legal = (~has_legal).float()  # (B, 1) — 1.0 when no squares available
    full_mask = torch.cat([legal, pass_legal], dim=-1)  # (B, 65)
    return logits.masked_fill(full_mask < 0.5, float("-inf"))


def _rollout_step_selfplay(
    env,
    agent_actions: np.ndarray,
    snapshot_policy,
    snap_lstm: Dict[str, torch.Tensor],
    device: torch.device,
) -> tuple:
    """One split-step using the snapshot policy as opponent (with action masking)."""
    opp_obs_np = env.step_agent(agent_actions)
    opp_obs_t = torch.tensor(opp_obs_np, dtype=torch.float32, device=device)
    with torch.no_grad():
        opp_logits, _ = snapshot_policy.forward_eval(opp_obs_t, snap_lstm)
    # Mask illegal actions; sample (not argmax) for exploration diversity
    opp_masked = _apply_action_mask(opp_logits, opp_obs_t)
    opp_dist = Categorical(logits=opp_masked)
    opp_actions = opp_dist.sample().cpu().numpy().astype(np.int32)
    return env.step_opponent(opp_actions)


# ---------------------------------------------------------------------------
# LSTM state helpers
# ---------------------------------------------------------------------------


def _init_lstm_state(
    num_agents: int, hidden_size: int, device: torch.device
) -> Dict[str, torch.Tensor]:
    return {
        "lstm_h": torch.zeros(1, num_agents, hidden_size, device=device),
        "lstm_c": torch.zeros(1, num_agents, hidden_size, device=device),
    }


# ---------------------------------------------------------------------------
# Rollout storage
# ---------------------------------------------------------------------------


class RolloutBuffer:
    """Fixed-horizon rollout storage for vectorised agents."""

    def __init__(
        self,
        num_agents: int,
        obs_shape: tuple,
        horizon: int,
        hidden_size: int,
        device: torch.device,
    ) -> None:
        self.num_agents = num_agents
        self.horizon = horizon
        self.device = device

        self.observations = torch.zeros(
            horizon, num_agents, *obs_shape, device=device
        )
        self.actions = torch.zeros(
            horizon, num_agents, dtype=torch.long, device=device
        )
        self.log_probs = torch.zeros(horizon, num_agents, device=device)
        self.rewards = torch.zeros(horizon, num_agents, device=device)
        self.dones = torch.zeros(horizon, num_agents, device=device)
        self.values = torch.zeros(horizon, num_agents, device=device)
        self.lstm_h = torch.zeros(horizon, num_agents, hidden_size, device=device)
        self.lstm_c = torch.zeros(horizon, num_agents, hidden_size, device=device)

    def store(
        self,
        step: int,
        obs: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
        lstm_h: torch.Tensor,
        lstm_c: torch.Tensor,
    ) -> None:
        self.observations[step] = obs
        self.actions[step] = action
        self.log_probs[step] = log_prob
        self.dones[step] = done
        self.values[step] = value
        self.lstm_h[step] = lstm_h.squeeze(0)
        self.lstm_c[step] = lstm_c.squeeze(0)

    def compute_gae(
        self,
        last_value: torch.Tensor,
        last_done: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (advantages, returns) using Generalised Advantage Estimation."""
        advantages = torch.zeros_like(self.rewards)
        last_gae = torch.zeros(self.num_agents, device=self.device)

        for t in reversed(range(self.horizon)):
            if t == self.horizon - 1:
                next_val = last_value
                next_nonterminal = 1.0 - last_done.float()
            else:
                next_val = self.values[t + 1]
                next_nonterminal = 1.0 - self.dones[t + 1].float()

            delta = self.rewards[t] + gamma * next_val * next_nonterminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
            advantages[t] = last_gae

        returns = advantages + self.values
        return advantages, returns


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------


def train(
    config_path: Optional[str] = None,
    load_checkpoint: Optional[str] = None,
) -> None:
    """Run the full PPO training loop."""

    # ------------------------------------------------------------------
    # CLI + config
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--config", type=str, default=None, help="Path to config.ini")
    parser.add_argument(
        "--load_checkpoint", type=str, default=None, help="Resume from checkpoint"
    )
    parser.add_argument("--wandb", action="store_true", default=False,
                        help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="othello-rl")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    known, remainder = parser.parse_known_args()

    cfg = _load_config(config_path or known.config)
    _apply_cli_overrides(cfg, remainder)

    if load_checkpoint is None:
        load_checkpoint = known.load_checkpoint

    # ------------------------------------------------------------------
    # Parse config
    # ------------------------------------------------------------------
    num_envs = cfg.getint("env", "num_envs")

    exp_name = cfg.get("train", "exp_name")
    data_dir = cfg.get("train", "data_dir")
    seed = cfg.getint("train", "seed")
    torch_deterministic = cfg.getboolean("train", "torch_deterministic")
    device_str = cfg.get("train", "device")
    total_timesteps = int(
        float(cfg.get("train", "total_timesteps").replace("_", ""))
    )
    lr = float(cfg.get("train", "learning_rate"))
    gamma = float(cfg.get("train", "gamma"))
    gae_lambda = float(cfg.get("train", "gae_lambda"))
    update_epochs = cfg.getint("train", "update_epochs")
    clip_coef = float(cfg.get("train", "clip_coef"))
    vf_coef = float(cfg.get("train", "vf_coef"))
    ent_coef = float(cfg.get("train", "ent_coef"))
    max_grad_norm = float(cfg.get("train", "max_grad_norm"))
    anneal_lr = cfg.getboolean("train", "anneal_lr")
    bptt_horizon = cfg.getint("train", "bptt_horizon")
    minibatch_size = cfg.getint("train", "minibatch_size")
    checkpoint_interval = cfg.getint("train", "checkpoint_interval")
    eval_interval = cfg.getint("train", "eval_interval", fallback=50000)
    eval_depths   = [int(d.strip()) for d in cfg.get("train", "eval_depths", fallback="1,2,3,5").split(",")]
    eval_n_games  = cfg.getint("train", "eval_n_games", fallback=50)

    hidden_size = cfg.getint("policy", "hidden_size")

    use_wandb = known.wandb
    wandb_project = known.wandb_project
    wandb_entity = known.wandb_entity
    wandb_run_name = known.wandb_run_name or f"{exp_name}_{int(time.time())}"

    _wandb = None

    # ------------------------------------------------------------------
    # Reproducibility + device
    # ------------------------------------------------------------------
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic

    if device_str == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available — using CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------
    env = Othello(num_envs=num_envs, seed=seed)
    obs_shape = env.single_observation_space.shape

    batch_size = num_envs * bptt_horizon
    total_updates = total_timesteps // batch_size
    num_minibatches = max(1, batch_size // minibatch_size)

    print(
        f"[Train] exp={exp_name}  envs={num_envs}  horizon={bptt_horizon}  "
        f"batch={batch_size}  minibatches={num_minibatches}  "
        f"total_updates={total_updates}  device={device}"
    )

    if use_wandb:
        import wandb
        _wandb = wandb
        _wandb.init(
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
        # When running as a wandb sweep agent, override config with injected values
        for key, value in _wandb.config.items():
            if "." in str(key):
                section, name = str(key).split(".", 1)
                name = name.replace("-", "_")
                if cfg.has_section(section):
                    cfg.set(section, name, str(value))
        # Re-parse ALL sweep-affected values so every hyperparameter can be
        # varied from a wandb sweep config, not just the subset listed below.
        lr = float(cfg.get("train", "learning_rate"))
        gamma = float(cfg.get("train", "gamma"))
        gae_lambda = float(cfg.get("train", "gae_lambda"))
        ent_coef = float(cfg.get("train", "ent_coef"))
        clip_coef = float(cfg.get("train", "clip_coef"))
        vf_coef = float(cfg.get("train", "vf_coef"))
        max_grad_norm = float(cfg.get("train", "max_grad_norm"))
        update_epochs = cfg.getint("train", "update_epochs")
        minibatch_size = cfg.getint("train", "minibatch_size")
        bptt_horizon = cfg.getint("train", "bptt_horizon")
        hidden_size = cfg.getint("policy", "hidden_size")
        # Recalculate derived batch dimensions in case bptt_horizon was swept
        batch_size = num_envs * bptt_horizon
        total_updates = total_timesteps // batch_size
        num_minibatches = max(1, batch_size // minibatch_size)

    # ------------------------------------------------------------------
    # Curriculum — uses the real CurriculumScheduler API
    # ------------------------------------------------------------------
    self_play_pool_capacity = cfg.getint("curriculum", "self_play_pool_capacity", fallback=20)
    self_play_refresh_interval = cfg.getint("curriculum", "self_play_refresh_interval", fallback=50_000)
    # Phase split-points: fraction of total_timesteps where each phase begins.
    # phase_0 always starts at 0.0; these 5 values define the remaining boundaries.
    phase_fractions = [
        cfg.getfloat("curriculum", "phase_1_fraction", fallback=0.05),
        cfg.getfloat("curriculum", "phase_2_fraction", fallback=0.15),
        cfg.getfloat("curriculum", "phase_3_fraction", fallback=0.30),
        cfg.getfloat("curriculum", "phase_4_fraction", fallback=0.50),
        cfg.getfloat("curriculum", "phase_5_fraction", fallback=0.65),
    ]
    curriculum = CurriculumScheduler(
        total_timesteps=total_timesteps,
        self_play_pool_capacity=self_play_pool_capacity,
        self_play_refresh_interval=self_play_refresh_interval,
        phase_fractions=phase_fractions,
    )

    # ------------------------------------------------------------------
    # Policy + optimiser
    # ------------------------------------------------------------------
    policy = make_policy(env, hidden_size=hidden_size).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr, eps=1e-5)

    global_step = 0
    start_update = 0

    if load_checkpoint is not None:
        ckpt = torch.load(load_checkpoint, map_location=device)
        policy.load_state_dict(ckpt["policy_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        global_step = ckpt.get("global_step", 0)
        start_update = global_step // batch_size
        print(f"[Train] Resumed from {load_checkpoint}  global_step={global_step}")

    # ------------------------------------------------------------------
    # Checkpoint directory
    # ------------------------------------------------------------------
    ckpt_dir = Path(data_dir) / exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Rollout buffer
    # ------------------------------------------------------------------
    rollout = RolloutBuffer(
        num_agents=num_envs,
        obs_shape=obs_shape,
        horizon=bptt_horizon,
        hidden_size=hidden_size,
        device=device,
    )

    # ------------------------------------------------------------------
    # Initial reset
    # ------------------------------------------------------------------
    obs_np, _ = env.reset(seed=seed)
    obs = torch.tensor(obs_np, dtype=torch.float32, device=device)
    done = torch.zeros(num_envs, device=device)
    lstm_state = _init_lstm_state(num_envs, hidden_size, device)

    log_interval = max(1, total_updates // 100)
    wall_start = time.time()
    step_start = global_step

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    snapshot_policy = None
    snap_lstm_state = None
    _last_snap_sd = None  # track last loaded state_dict by identity

    for update in range(start_update, total_updates):

        # ---- LR annealing --------------------------------------------
        if anneal_lr:
            for g in optimizer.param_groups:
                g["lr"] = (1.0 - update / total_updates) * lr

        # ---- Curriculum ----------------------------------------------
        # update() returns phase dict AND manages self_play_pool snapshots
        phase = curriculum.update(global_step, policy)
        opp_type = phase["type"]
        opp_depth = phase["depth"]

        # Load/refresh snapshot policy for self-play (only when snapshot changes)
        if opp_type == "self_play":
            snap_sd = curriculum.self_play_pool.sample()
            if snap_sd is not None and snap_sd is not _last_snap_sd:
                snapshot_policy = _load_snapshot_policy(
                    snap_sd, env, hidden_size, device
                )
                snap_lstm_state = _init_lstm_state(num_envs, hidden_size, device)
                _last_snap_sd = snap_sd

        # ---- Rollout collection --------------------------------------
        policy.eval()
        with torch.no_grad():
            for step in range(bptt_horizon):
                # Zero LSTM state for freshly-done agents
                if done.any():
                    mask = done.bool()
                    lstm_state["lstm_h"][:, mask] = 0.0
                    lstm_state["lstm_c"][:, mask] = 0.0
                    if snap_lstm_state is not None:
                        snap_lstm_state["lstm_h"][:, mask] = 0.0
                        snap_lstm_state["lstm_c"][:, mask] = 0.0

                # Capture LSTM input state BEFORE forward_eval mutates it in-place.
                # Storing the pre-call state ensures the PPO update re-enters the
                # LSTM from the same hidden state used during rollout collection,
                # so old_log_prob and new_log_prob are consistent (ratio ≈ 1 at
                # epoch 0).  Storing the post-call state causes ratio ≠ 1 even
                # for unchanged parameters, breaking the PPO clipping guarantee.
                lstm_h_pre = lstm_state["lstm_h"].clone()
                lstm_c_pre = lstm_state["lstm_c"].clone()

                logits, value = policy.forward_eval(obs, lstm_state)
                masked_logits = _apply_action_mask(logits, obs)
                dist = Categorical(logits=masked_logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                rollout.store(
                    step=step,
                    obs=obs,
                    action=action,
                    log_prob=log_prob,
                    done=done,
                    value=value.squeeze(-1),
                    lstm_h=lstm_h_pre,
                    lstm_c=lstm_c_pre,
                )

                action_np = action.cpu().numpy().astype(np.int32)
                if opp_type == "self_play" and snapshot_policy is not None:
                    obs_np, rew_np, term_np, trunc_np, step_infos = _rollout_step_selfplay(
                        env, action_np, snapshot_policy, snap_lstm_state, device
                    )
                elif opp_type == "negamax" and opp_depth > 0:
                    obs_np, rew_np, term_np, trunc_np, step_infos = env.step_negamax(
                        action_np, opp_depth
                    )
                else:
                    obs_np, rew_np, term_np, trunc_np, step_infos = env.step(action_np)

                if step_infos:
                    info = step_infos[0]
                    games_played = info.get("games_played", 0)
                    if games_played > 0 and _wandb is not None:
                        _wandb.log(
                            {
                                "episode/win_rate": info.get("win_rate", 0.0),
                                "episode/length": info.get("avg_game_length", 0.0),
                                "episode/invalid_moves": info.get("invalid_move_rate", 0.0),
                                "episode/corner_captures": info.get("corner_captures", 0) / max(games_played, 1),
                            },
                            step=global_step,
                        )

                rollout.rewards[step] = torch.tensor(
                    rew_np, dtype=torch.float32, device=device
                )
                done = torch.tensor(
                    (term_np | trunc_np).astype(np.float32), device=device
                )
                obs = torch.tensor(obs_np, dtype=torch.float32, device=device)
                global_step += num_envs

            # Bootstrap value for last observation
            if done.any():
                mask = done.bool()
                lstm_state["lstm_h"][:, mask] = 0.0
                lstm_state["lstm_c"][:, mask] = 0.0
                if snap_lstm_state is not None:
                    snap_lstm_state["lstm_h"][:, mask] = 0.0
                    snap_lstm_state["lstm_c"][:, mask] = 0.0
            _, last_value = policy.forward_eval(obs, lstm_state)
            last_value = last_value.squeeze(-1).detach()

        # ---- GAE -----------------------------------------------------
        advantages, returns = rollout.compute_gae(
            last_value, done, gamma, gae_lambda
        )

        # ---- Flatten for mini-batch updates --------------------------
        b_obs = rollout.observations.reshape(batch_size, *obs_shape)
        b_actions = rollout.actions.reshape(batch_size)
        b_log_probs = rollout.log_probs.reshape(batch_size)
        b_advantages = advantages.reshape(batch_size)
        b_returns = returns.reshape(batch_size)
        b_values = rollout.values.reshape(batch_size)
        b_lstm_h = rollout.lstm_h.reshape(batch_size, hidden_size)
        b_lstm_c = rollout.lstm_c.reshape(batch_size, hidden_size)

        b_advantages = (b_advantages - b_advantages.mean()) / (
            b_advantages.std() + 1e-8
        )

        # ---- PPO gradient steps --------------------------------------
        policy.train()
        clip_fracs: list[float] = []
        last_loss = last_pg = last_vf = last_ent = 0.0
        grad_norm: float = 0.0

        for _ in range(update_epochs):
            perm = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, minibatch_size):
                idx = perm[start : start + minibatch_size]

                mb_state = {
                    "lstm_h": b_lstm_h[idx].unsqueeze(0),
                    "lstm_c": b_lstm_c[idx].unsqueeze(0),
                }
                logits, new_value = policy.forward_eval(b_obs[idx], mb_state)
                masked_logits = _apply_action_mask(logits, b_obs[idx])
                dist = Categorical(logits=masked_logits)
                new_log_prob = dist.log_prob(b_actions[idx])
                entropy = dist.entropy()

                ratio = torch.exp(new_log_prob - b_log_probs[idx])
                clip_fracs.append(
                    ((ratio - 1.0).abs() > clip_coef).float().mean().item()
                )
                mb_adv = b_advantages[idx]
                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * ratio.clamp(1 - clip_coef, 1 + clip_coef),
                ).mean()

                v_pred = new_value.squeeze(-1)
                v_clipped = b_values[idx] + (v_pred - b_values[idx]).clamp(
                    -clip_coef, clip_coef
                )
                vf_loss = torch.max(
                    (v_pred - b_returns[idx]).pow(2),
                    (v_clipped - b_returns[idx]).pow(2),
                ).mean()

                ent_loss = entropy.mean()
                loss = pg_loss + vf_coef * vf_loss - ent_coef * ent_loss

                optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()

                last_loss = loss.item()
                last_pg = pg_loss.item()
                last_vf = vf_loss.item()
                last_ent = ent_loss.item()

        # ---- Periodic logging ----------------------------------------
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
            if _wandb is not None:
                # Phase index: 0=random, 1=negamax_d1, 2=negamax_d2, 3=negamax_d3,
                # 4=negamax_d5, 5=self_play — mirrors PHASE_BOUNDARIES order
                _negamax_depth_to_phase = {1: 1, 2: 2, 3: 3, 5: 4}
                if opp_type == "random":
                    _phase_idx = 0
                elif opp_type == "negamax":
                    _phase_idx = _negamax_depth_to_phase.get(opp_depth, 1)
                else:  # self_play
                    _phase_idx = 5
                _wandb.log(
                    {
                        "loss/total": last_loss,
                        "loss/policy": last_pg,
                        "loss/value": last_vf,
                        "loss/entropy": last_ent,
                        "train/clip_fraction": mean_clip,
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                        "train/sps": sps,
                        "train/grad_norm": float(grad_norm),
                        "curriculum/phase": _phase_idx,
                        "curriculum/opp_depth": opp_depth,
                    },
                    step=global_step,
                )

        # ---- Checkpoint ----------------------------------------------
        if (update + 1) % checkpoint_interval == 0:
            path = ckpt_dir / f"checkpoint_{update + 1:06d}.pt"
            torch.save(
                {
                    "global_step": global_step,
                    "policy_state_dict": policy.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                str(path),
            )
            print(f"  → saved {path}")
            if _wandb is not None:
                artifact = _wandb.Artifact(
                    name=f"{exp_name}_checkpoint",
                    type="model",
                    metadata={"global_step": global_step, "update": update + 1},
                )
                artifact.add_file(str(path))
                _wandb.log_artifact(artifact)

        # ---- Periodic eval ------------------------------------------
        if global_step > 0 and global_step % eval_interval < num_envs * bptt_horizon:
            policy.eval()
            eval_results = _evaluate(
                policy, depths=eval_depths, n_games=eval_n_games, device=device
            )
            policy.train()
            eval_str = "  ".join(
                f"d{d}={wr:.0%}" for d, wr in sorted(eval_results.items())
            )
            print(f"  [eval] {eval_str}")
            if _wandb is not None:
                _wandb.log(
                    {f"eval/win_rate_vs_negamax_d{d}": wr for d, wr in eval_results.items()},
                    step=global_step,
                )

    # ---- Final checkpoint -------------------------------------------
    final_path = ckpt_dir / "checkpoint_final.pt"
    torch.save(
        {
            "global_step": global_step,
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        str(final_path),
    )
    print(f"\n[Train] Complete. final checkpoint → {final_path}")
    if _wandb is not None:
        _wandb.finish()
    env.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train()
