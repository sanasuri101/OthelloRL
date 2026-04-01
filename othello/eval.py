"""Fixed-opponent evaluation for Othello RL policies.

Usage (standalone):
    python othello/eval.py --checkpoint experiments/othello_ppo/checkpoint_000200.pt
    python othello/eval.py --checkpoint model.pt --depths 1,2,3,5 --n_games 200 --wandb
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def evaluate(
    policy: nn.Module,
    depths: list[int] | None = None,
    n_games: int = 100,
    device: str = "cpu",
) -> dict[int, float]:
    """Evaluate *policy* against negamax at each depth in *depths*.

    Parameters
    ----------
    policy:
        An OthelloPolicy (or compatible) instance with ``forward_eval``
        and an ``lstm_cell`` attribute (used to infer hidden size).
    depths:
        Negamax search depths. Defaults to ``[1, 2, 3, 5]``.
    n_games:
        Number of games to play at each depth.
    device:
        Torch device string.

    Returns
    -------
    dict[int, float]
        Win rate (0.0-1.0) keyed by depth.
    """
    from othello.othello import Othello  # local import avoids circular at module load

    if depths is None:
        depths = [1, 2, 3, 5]

    hidden_size: int = policy.lstm_cell.hidden_size
    policy = policy.to(device)
    policy.eval()

    results: dict[int, float] = {}

    for depth in depths:
        env = Othello(num_envs=1)
        obs_np, _ = env.reset()

        lstm_state: dict[str, torch.Tensor] = {
            "lstm_h": torch.zeros(1, 1, hidden_size, device=device),
            "lstm_c": torch.zeros(1, 1, hidden_size, device=device),
        }

        games_done = 0
        wins = 0

        while games_done < n_games:
            obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device)
            with torch.no_grad():
                logits, _ = policy.forward_eval(obs_t, lstm_state)
            action = logits.argmax(dim=-1).cpu().numpy().astype(np.int32)

            obs_np, rew_np, term_np, _trunc_np, _infos = env.step_negamax(
                action, depth
            )

            if term_np[0]:
                games_done += 1
                if rew_np[0] > 0:
                    wins += 1
                lstm_state["lstm_h"].zero_()
                lstm_state["lstm_c"].zero_()

        env.close()
        results[depth] = wins / n_games

    return results


def _print_results(results: dict[int, float]) -> None:
    print("\nEvaluation Results")
    print("-" * 32)
    for depth, wr in sorted(results.items()):
        bar = "\u2588" * int(wr * 20)
        print(f"  vs negamax d{depth}: {wr:.1%}  {bar}")
    print()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate an Othello RL checkpoint")
    parser.add_argument(
        "--checkpoint", required=True, help="Path to .pt checkpoint file"
    )
    parser.add_argument(
        "--depths", default="1,2,3,5", help="Comma-separated negamax depths"
    )
    parser.add_argument("--n_games", type=int, default=100, help="Games per depth")
    parser.add_argument("--device", default="cpu", help="Torch device")
    parser.add_argument("--wandb", action="store_true", help="Log results to wandb")
    parser.add_argument("--wandb_project", default="othello-rl")
    parser.add_argument("--wandb_run_name", default=None)
    args = parser.parse_args(argv)

    depths = [int(d.strip()) for d in args.depths.split(",")]
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Error: checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from othello.train import make_policy  # noqa: PLC0415
    from othello.othello import Othello  # noqa: PLC0415

    _env = Othello(num_envs=1)
    policy = make_policy(_env, hidden_size=256).to(args.device)
    _env.close()

    ckpt = torch.load(str(ckpt_path), map_location=args.device, weights_only=True)
    policy.load_state_dict(ckpt["policy_state_dict"])
    global_step = ckpt.get("global_step", 0)
    print(f"Loaded checkpoint: {ckpt_path}  (global_step={global_step})")

    results = evaluate(
        policy, depths=depths, n_games=args.n_games, device=args.device
    )
    _print_results(results)

    if args.wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"eval_{ckpt_path.stem}",
            config={"checkpoint": str(ckpt_path), "n_games": args.n_games},
        )
        wandb.log(
            {f"eval/win_rate_vs_negamax_d{d}": wr for d, wr in results.items()},
            step=global_step,
        )
        run.finish()
        print("Results logged to wandb.")


if __name__ == "__main__":
    main()
