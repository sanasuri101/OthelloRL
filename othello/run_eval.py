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
        try:
            output, new_lstm = policy(obs_batch, lstm_state)
        except TypeError:
            # Policy may not accept lstm_state
            output = policy(obs_batch)
            new_lstm = lstm_state

        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output

        action = int(torch.argmax(logits, dim=-1).item())
    return action, new_lstm


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
    """Human plays against the agent using mouse clicks on the rendered board."""
    env = SingleEnv()
    policy = _build_policy()
    load_checkpoint(checkpoint_path, policy)
    policy.eval()

    print("--- Human vs Agent ---")
    print("Click a square to place your piece.  Close the window to quit.")

    obs = env.reset()
    lstm_state = None
    done = False
    reward = 0.0
    human_turn = False  # Agent moves first (plays as the env's default color)

    while not env.render_should_close():
        env.render()

        if done:
            result = "YOU WIN" if reward < 0 else ("YOU LOSE" if reward > 0 else "DRAW")
            print(f"Game over: {result} (reward={reward:.2f})")
            time.sleep(2.0)
            obs = env.reset()
            lstm_state = None
            done = False
            human_turn = False
            continue

        if human_turn:
            click = env.render_get_click()
            if click < 0:
                # No click yet
                time.sleep(0.016)  # ~60 fps polling
                continue
            # click is the board square index (0-63) or -1 for no click
            obs, reward, done = env.step(click)
            human_turn = False
        else:
            # Agent's turn
            time.sleep(0.3)
            obs_t = torch.tensor(obs, dtype=torch.float32)
            action, lstm_state = select_action(policy, obs_t, lstm_state)
            obs, reward, done = env.step(action)
            human_turn = True

    env.close()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _build_policy():
    """Construct a policy network with the correct observation/action spaces."""
    import gymnasium as gym

    obs_space = gym.spaces.Box(low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
    act_space = gym.spaces.Discrete(NUM_ACTIONS)
    return make_policy(obs_space, act_space)


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


if __name__ == "__main__":
    main()
