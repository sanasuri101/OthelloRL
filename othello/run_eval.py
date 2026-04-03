"""Evaluation script for Othello RL agents.

Supports four modes:
  --render    : Visual playback with raylib rendering (~2 moves/sec)
  --episodes N: Headless evaluation over N games (win/loss/draw stats)
  --ladder    : Ladder test vs negamax at depth 1, 2, 3, 5
  --human     : Human plays against the agent via mouse clicks
"""

import argparse
import time

import numpy as np
import torch

# OBS_DIM and NUM_ACTIONS must match othello.h
OBS_DIM = 192
NUM_ACTIONS = 65

# ---------------------------------------------------------------------------
# Policy construction
# ---------------------------------------------------------------------------

try:
    from train import make_policy
except ImportError:

    def make_policy(env_obs_space, env_action_space):
        """Fallback policy factory when train.py is not available.

        Builds a simple LSTM-backed policy compatible with pufferlib.
        """
        import pufferlib.models

        class OthelloPolicy(pufferlib.models.Default):
            pass

        return OthelloPolicy(
            env_obs_space,
            env_action_space,
            input_size=OBS_DIM,
            hidden_size=256,
        )


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def load_checkpoint(path, policy):
    """Load a policy checkpoint, handling multiple serialisation formats."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict):
        for key in ("policy_state_dict", "model_state_dict", "state_dict"):
            if key in checkpoint:
                policy.load_state_dict(checkpoint[key])
                return
        # Assume the entire dict is a raw state_dict
        policy.load_state_dict(checkpoint)
    else:
        raise ValueError(f"Unexpected checkpoint type: {type(checkpoint)}")


# ---------------------------------------------------------------------------
# Single-env wrapper around the C VecEnv
# ---------------------------------------------------------------------------


class SingleEnv:
    """Thin wrapper that drives a 1-env VecEnv for evaluation."""

    def __init__(self):
        from binding import VecEnv

        self.vec = VecEnv()
        self.obs = np.zeros((1, OBS_DIM), dtype=np.float32)
        self.actions = np.zeros((1,), dtype=np.int32)
        self.rewards = np.zeros((1,), dtype=np.float32)
        self.dones = np.zeros((1,), dtype=np.int32)
        self.vec.init(1, self.obs, self.actions, self.rewards, self.dones)
        self.vec.reset()

    def reset(self):
        self.vec.reset()
        return self.obs[0].copy()

    def step(self, action):
        self.actions[0] = action
        self.vec.step()
        obs = self.obs[0].copy()
        reward = float(self.rewards[0])
        done = bool(self.dones[0])
        return obs, reward, done

    def render(self):
        self.vec.render(0)

    def render_should_close(self):
        return self.vec.render_should_close()

    def render_get_click(self):
        return self.vec.render_get_click()

    def negamax_move(self, depth):
        return self.vec.negamax_move(0, depth)

    def apply_opponent_move(self, move):
        self.vec.apply_opponent_move(0, move)

    def close(self):
        self.vec.close()


# ---------------------------------------------------------------------------
# Agent inference helpers
# ---------------------------------------------------------------------------


def select_action(policy, obs_tensor, lstm_state):
    """Run one forward pass and return (action, new_lstm_state)."""
    with torch.no_grad():
        obs_batch = obs_tensor.unsqueeze(0)  # (1, OBS_DIM)

        # Initialise LSTM state on first call
        if lstm_state is None:
            h = policy.lstm_cell.hidden_size
            lstm_state = {
                "lstm_h": torch.zeros(1, 1, h),
                "lstm_c": torch.zeros(1, 1, h),
            }

        logits, _ = policy.forward_eval(obs_batch, lstm_state)

        # Mask illegal actions using the legal-move plane (obs dims 128-191)
        legal = obs_batch[:, 128:192]
        has_legal = legal.sum(dim=-1, keepdim=True) > 0
        pass_legal = (~has_legal).float()
        full_mask = torch.cat([legal, pass_legal], dim=-1)  # (1, 65)
        logits = logits.masked_fill(full_mask < 0.5, float("-inf"))

        action = int(torch.argmax(logits, dim=-1).item())
    return action, lstm_state


# ---------------------------------------------------------------------------
# Mode: headless evaluation
# ---------------------------------------------------------------------------


def run_headless(checkpoint_path, num_episodes):
    """Play *num_episodes* games headless and print statistics."""
    env = SingleEnv()
    policy = _build_policy()
    load_checkpoint(checkpoint_path, policy)
    policy.eval()

    wins, losses, draws = 0, 0, 0
    lstm_state = None

    for ep in range(1, num_episodes + 1):
        obs = env.reset()
        lstm_state = None
        done = False

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32)
            action, lstm_state = select_action(policy, obs_t, lstm_state)
            obs, reward, done = env.step(action)

        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            draws += 1

        if ep % max(1, num_episodes // 10) == 0:
            print(
                f"  [{ep}/{num_episodes}] "
                f"W={wins} L={losses} D={draws} "
                f"WR={wins / ep:.1%}"
            )

    total = wins + losses + draws
    print("\n--- Headless Evaluation ---")
    print(f"Episodes : {total}")
    print(f"Wins     : {wins}  ({wins / total:.1%})")
    print(f"Losses   : {losses}  ({losses / total:.1%})")
    print(f"Draws    : {draws}  ({draws / total:.1%})")
    env.close()


# ---------------------------------------------------------------------------
# Mode: visual (render)
# ---------------------------------------------------------------------------


def run_visual(checkpoint_path, num_episodes):
    """Watch the agent play with raylib rendering at ~2 moves/sec."""
    env = SingleEnv()
    policy = _build_policy()
    load_checkpoint(checkpoint_path, policy)
    policy.eval()

    episodes_done = 0

    while episodes_done < num_episodes:
        obs = env.reset()
        lstm_state = None
        done = False

        while not done:
            if env.render_should_close():
                env.close()
                return

            env.render()
            time.sleep(0.5)  # ~2 moves per second

            obs_t = torch.tensor(obs, dtype=torch.float32)
            action, lstm_state = select_action(policy, obs_t, lstm_state)
            obs, reward, done = env.step(action)

        # Show final board briefly
        env.render()
        time.sleep(1.0)
        episodes_done += 1
        result = "WIN" if reward > 0 else ("LOSS" if reward < 0 else "DRAW")
        print(f"Episode {episodes_done}: {result} (reward={reward:.2f})")

    env.close()


# ---------------------------------------------------------------------------
# Mode: self-play visual
# ---------------------------------------------------------------------------


def run_selfplay(checkpoint_path, num_episodes, move_delay=0.5):
    """Watch checkpoint_016600 play against itself with raylib rendering.

    Black = checkpoint policy, opening Dirichlet noise then argmax.
    White = same checkpoint policy, opening Dirichlet noise then argmax.
    Both use independent LSTM states.
    """
    import othello as _othello_pkg

    env = _othello_pkg.Othello(num_envs=1)
    policy = _build_policy()
    load_checkpoint(checkpoint_path, policy)
    policy.eval()

    h = policy.lstm_cell.hidden_size

    def fresh_lstm():
        return {
            "lstm_h": torch.zeros(1, 1, h),
            "lstm_c": torch.zeros(1, 1, h),
        }

    def pick_action(logits, full_mask, move_count):
        """Dirichlet-noised sampling for opening, argmax thereafter."""
        if move_count < 8:
            probs = torch.softmax(logits, dim=-1)
            legal_indices = full_mask[0].nonzero(as_tuple=True)[0]
            n_legal = len(legal_indices)
            if n_legal > 0:
                noise = torch.tensor(
                    np.random.dirichlet([0.3] * n_legal), dtype=torch.float32
                )
                probs[0, legal_indices] = 0.75 * probs[0, legal_indices] + 0.25 * noise
            return torch.distributions.Categorical(probs=probs).sample().cpu().numpy().astype(np.int32)
        return logits.argmax(dim=-1).cpu().numpy().astype(np.int32)

    black_lstm = fresh_lstm()
    white_lstm = fresh_lstm()
    episodes_done = 0
    black_wins = white_wins = draws = 0
    move_count = 0

    env.reset()
    print(f"--- Agent (Black) vs Agent (White) — checkpoint_016600 self-play ---")

    while episodes_done < num_episodes:
        if env._c_env.render_should_close():
            break

        # ── Black's turn ─────────────────────────────────────────────────
        env._c_env.render(0)
        time.sleep(move_delay)

        obs_t = torch.tensor(env.observations, dtype=torch.float32)
        with torch.no_grad():
            logits, _ = policy.forward_eval(obs_t, black_lstm)
            legal = obs_t[:, 128:192]
            has_legal = legal.sum(dim=-1, keepdim=True) > 0
            pass_legal = (~has_legal).float()
            full_mask = torch.cat([legal, pass_legal], dim=-1)
            logits = logits.masked_fill(full_mask < 0.5, float("-inf"))
            black_action = pick_action(logits, full_mask, move_count)

        opp_obs_np = env.step_agent(black_action)
        move_count += 1

        # ── White's turn ─────────────────────────────────────────────────
        env._c_env.render(0)
        time.sleep(move_delay)

        opp_obs_t = torch.tensor(opp_obs_np, dtype=torch.float32)
        with torch.no_grad():
            logits, _ = policy.forward_eval(opp_obs_t, white_lstm)
            legal = opp_obs_t[:, 128:192]
            has_legal = legal.sum(dim=-1, keepdim=True) > 0
            pass_legal = (~has_legal).float()
            full_mask = torch.cat([legal, pass_legal], dim=-1)
            logits = logits.masked_fill(full_mask < 0.5, float("-inf"))
            white_action = pick_action(logits, full_mask, move_count)

        _, rew_np, term_np, _, _ = env.step_opponent(white_action)
        move_count += 1

        if term_np[0]:
            env._c_env.render(0)
            episodes_done += 1
            if rew_np[0] > 0:
                result = "Black wins"
                black_wins += 1
            elif rew_np[0] < 0:
                result = "White wins"
                white_wins += 1
            else:
                result = "Draw"
                draws += 1
            print(f"Game {episodes_done}: {result}  "
                  f"[Black {black_wins}W / White {white_wins}W / {draws}D]")
            time.sleep(1.5)
            env.reset()
            black_lstm = fresh_lstm()
            white_lstm = fresh_lstm()
            move_count = 0

    env.close()
    print(f"\nFinal: Black {black_wins}W / White {white_wins}W / {draws}D "
          f"out of {episodes_done} games")


# ---------------------------------------------------------------------------
# Mode: ladder (negamax depths)
# ---------------------------------------------------------------------------


def run_ladder(checkpoint_path, episodes_per_depth=50):
    """Test agent vs negamax at increasing depths."""
    depths = [1, 2, 3, 5]
    policy = _build_policy()
    load_checkpoint(checkpoint_path, policy)
    policy.eval()

    print("--- Ladder Evaluation ---")
    print(f"Episodes per depth: {episodes_per_depth}\n")

    for depth in depths:
        env = SingleEnv()
        wins, losses, draws = 0, 0, 0

        for ep in range(episodes_per_depth):
            obs = env.reset()
            lstm_state = None
            done = False

            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32)
                action, lstm_state = select_action(policy, obs_t, lstm_state)
                obs, reward, done = env.step(action)

            if reward > 0:
                wins += 1
            elif reward < 0:
                losses += 1
            else:
                draws += 1

        total = wins + losses + draws
        wr = wins / total if total > 0 else 0.0
        print(
            f"  depth={depth}: "
            f"W={wins} L={losses} D={draws} "
            f"WR={wr:.1%}"
        )
        env.close()

    print("\nNote: ladder currently uses the default random opponent in the C env.")
    print("For true negamax opposition, integrate negamax_move into the step loop.")


# ---------------------------------------------------------------------------
# Mode: human play
# ---------------------------------------------------------------------------


def run_human(checkpoint_path):
    """Human (Black) vs Agent (White).

    Flow mirrors _rollout_step_selfplay in train.py:
      step_agent(human_click)  →  human plays Black, returns White obs
      step_opponent(ai_action) →  AI plays White from White obs, returns Black obs + reward
    """
    import othello as _othello_pkg

    env = _othello_pkg.Othello(num_envs=1)
    policy = _build_policy()
    load_checkpoint(checkpoint_path, policy)
    policy.eval()

    h = policy.lstm_cell.hidden_size

    print("--- Human (Black) vs Agent (White) ---")
    print("You are Black and move first. Click a highlighted square.")
    print("Close the window to quit.")

    env.reset()
    ai_lstm = {
        "lstm_h": torch.zeros(1, 1, h),
        "lstm_c": torch.zeros(1, 1, h),
    }
    game_count = 0

    while not env._c_env.render_should_close():

        # ── Human's turn (Black) ────────────────────────────────────────
        # Drain any stale clicks buffered during AI's turn
        while env._c_env.render_get_click() >= 0:
            pass

        # Check if Black has any legal moves — if not, auto-pass
        obs_now = env.observations[0]
        legal_plane = obs_now[128:192]
        has_legal_moves = legal_plane.sum() > 0.5

        if not has_legal_moves:
            print("  (Black has no legal moves — passing)")
            opp_obs_np = env.step_agent(np.array([64], dtype=np.int32))  # 64 = pass
        else:
            # Render and wait for a LEGAL click — ignore illegal squares silently
            click = -1
            while click < 0:
                if env._c_env.render_should_close():
                    env.close()
                    return
                env._c_env.render(0)
                candidate = env._c_env.render_get_click()
                if candidate < 0:
                    time.sleep(0.016)
                    continue
                # Validate against legal move plane (obs dims 128-191)
                if candidate < 64 and legal_plane[candidate] > 0.5:
                    click = candidate  # legal — accept
                # else: illegal square, ignore and wait

            opp_obs_np = env.step_agent(np.array([click], dtype=np.int32))

        # Show board immediately after human's move
        env._c_env.render(0)

        # ── AI's turn (White = opponent) ────────────────────────────────
        time.sleep(0.5)

        if env._c_env.render_should_close():
            env.close()
            return

        opp_obs_t = torch.tensor(opp_obs_np, dtype=torch.float32)
        with torch.no_grad():
            logits, _ = policy.forward_eval(opp_obs_t, ai_lstm)
            legal = opp_obs_t[:, 128:192]
            has_legal = legal.sum(dim=-1, keepdim=True) > 0
            pass_legal = (~has_legal).float()
            full_mask = torch.cat([legal, pass_legal], dim=-1)
            logits = logits.masked_fill(full_mask < 0.5, float("-inf"))
            ai_action = logits.argmax(dim=-1).cpu().numpy().astype(np.int32)

        _, rew_np, term_np, _, _ = env.step_opponent(ai_action)

        if term_np[0]:
            game_count += 1
            # reward is from agent (Black/human) perspective
            result = "YOU WIN" if rew_np[0] > 0 else ("YOU LOSE" if rew_np[0] < 0 else "DRAW")
            print(f"Game {game_count}: {result}")
            env._c_env.render(0)
            time.sleep(2.0)
            env.reset()
            ai_lstm = {
                "lstm_h": torch.zeros(1, 1, h),
                "lstm_c": torch.zeros(1, 1, h),
            }

    env.close()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _build_policy():
    """Construct a policy network with the correct observation/action spaces."""
    import gymnasium as gym

    obs_space = gym.spaces.Box(low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
    act_space = gym.spaces.Discrete(NUM_ACTIONS)

    # make_policy expects an env-like object with single_observation_space / single_action_space
    class _FakeEnv:
        single_observation_space = obs_space
        single_action_space = act_space

    return make_policy(_FakeEnv(), hidden_size=256)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate an Othello RL agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the saved model checkpoint",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--render",
        action="store_true",
        help="Visual mode: watch the agent play with rendering",
    )
    mode.add_argument(
        "--episodes",
        type=int,
        metavar="N",
        help="Headless mode: run N evaluation games",
    )
    mode.add_argument(
        "--ladder",
        action="store_true",
        help="Ladder mode: test vs negamax at depth 1, 2, 3, 5",
    )
    mode.add_argument(
        "--human",
        action="store_true",
        help="Human mode: play against the agent via mouse clicks",
    )
    mode.add_argument(
        "--selfplay",
        action="store_true",
        help="Self-play mode: watch the agent play against itself",
    )

    parser.add_argument(
        "--render-episodes",
        type=int,
        default=10,
        help="Number of episodes for visual mode (default: 10)",
    )
    parser.add_argument(
        "--ladder-episodes",
        type=int,
        default=50,
        help="Episodes per depth level in ladder mode (default: 50)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.render:
        run_visual(args.checkpoint, args.render_episodes)
    elif args.episodes is not None:
        run_headless(args.checkpoint, args.episodes)
    elif args.ladder:
        run_ladder(args.checkpoint, args.ladder_episodes)
    elif args.human:
        run_human(args.checkpoint)
    elif args.selfplay:
        run_selfplay(args.checkpoint, args.render_episodes)


if __name__ == "__main__":
    main()
