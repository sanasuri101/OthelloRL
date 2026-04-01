"""Opponent curriculum scheduler for Othello RL training."""
import copy
import multiprocessing
import random


# Default phase split-points as fractions of total_timesteps.
# Each value is where the *next* phase begins.
# Override by passing phase_fractions to CurriculumScheduler.
_DEFAULT_PHASE_FRACTIONS = [0.05, 0.15, 0.30, 0.50, 0.65]

# Phase type/depth sequence (fixed; only the boundary fractions are configurable).
_PHASE_DEFS = [
    ("random",   0),
    ("negamax",  1),
    ("negamax",  2),
    ("negamax",  3),
    ("negamax",  5),
    ("self_play", 0),
]


def _build_phase_boundaries(phase_fractions):
    """Build the list of (start, end, type, depth) tuples from split-point fractions."""
    fracs = list(phase_fractions)
    starts = [0.0] + fracs
    ends   = fracs + [1.0]
    return [
        (start, end, ptype, depth)
        for (ptype, depth), start, end in zip(_PHASE_DEFS, starts, ends)
    ]


class SelfPlayPool:
    """Pool of frozen policy snapshots for self-play opponent diversity."""

    def __init__(self, max_size=5, refresh_interval=5_000_000):
        self.max_size = max_size
        self.refresh_interval = refresh_interval
        self.snapshots = []
        self._last_refresh_step = -refresh_interval

    def maybe_refresh(self, global_step, policy):
        """Snapshot the policy if enough steps have passed."""
        if global_step - self._last_refresh_step >= self.refresh_interval:
            snapshot = copy.deepcopy(policy.state_dict())
            self.snapshots.append(snapshot)
            if len(self.snapshots) > self.max_size:
                self.snapshots.pop(0)
            self._last_refresh_step = global_step

    def sample(self):
        """Return a random snapshot state_dict, or None if pool is empty."""
        if not self.snapshots:
            return None
        return random.choice(self.snapshots)

    @property
    def size(self):
        return len(self.snapshots)


class CurriculumScheduler:
    def __init__(
        self,
        total_timesteps=200_000_000,
        self_play_pool_capacity=5,
        self_play_refresh_interval=5_000_000,
        phase_fractions=None,
    ):
        self.total_timesteps = total_timesteps
        self.difficulty_value = multiprocessing.Value("f", 0.0)
        self.self_play_pool = SelfPlayPool(
            max_size=self_play_pool_capacity,
            refresh_interval=self_play_refresh_interval,
        )
        fracs = phase_fractions if phase_fractions is not None else _DEFAULT_PHASE_FRACTIONS
        self.phase_boundaries = _build_phase_boundaries(fracs)

    def get_difficulty(self, global_step):
        return min(1.0, max(0.0, global_step / self.total_timesteps))

    def get_phase(self, global_step):
        frac = self.get_difficulty(global_step)
        for i, (start, end, ptype, depth) in enumerate(self.phase_boundaries):
            is_last = i == len(self.phase_boundaries) - 1
            # Use >= start (no upper bound) for the last phase so frac=1.0 is included
            if start <= frac < end or (is_last and frac >= start):
                return {"type": ptype, "depth": depth, "difficulty": frac}
        return {"type": "self_play", "depth": 0, "difficulty": 1.0}

    def update(self, global_step, policy=None):
        """Update shared difficulty value and manage self-play pool."""
        d = self.get_difficulty(global_step)
        self.difficulty_value.value = d
        phase = self.get_phase(global_step)
        if phase["type"] == "self_play" and policy is not None:
            self.self_play_pool.maybe_refresh(global_step, policy)
        return phase
