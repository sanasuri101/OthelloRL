"""Tests for the PufferEnv Othello wrapper."""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from othello import Othello


class TestObservationSpace:
    def test_observation_space_shape(self):
        """single_observation_space should have shape (192,)."""
        env = Othello(num_envs=1)
        assert env.single_observation_space.shape == (192,)
        env.close()

    def test_observation_space_dtype(self):
        """single_observation_space dtype should be float32."""
        env = Othello(num_envs=1)
        assert env.single_observation_space.dtype == np.float32
        env.close()


class TestActionSpace:
    def test_action_space_n(self):
        """single_action_space should have n=65 (64 squares + pass)."""
        env = Othello(num_envs=1)
        assert env.single_action_space.n == 65
        env.close()


class TestReset:
    def test_reset_returns_valid_obs(self):
        """Reset should return observations with correct shape, dtype, and starting pieces."""
        env = Othello(num_envs=1)
        obs, infos = env.reset()

        assert obs.shape == (1, 192)
        assert obs.dtype == np.float32

        # Starting position: 2 own pieces + 2 opponent pieces + 4 legal moves
        my_pieces = obs[0, :64].sum()
        opp_pieces = obs[0, 64:128].sum()
        legal_moves = obs[0, 128:].sum()
        assert my_pieces + opp_pieces + legal_moves == 2 + 2 + 4, (
            f"Expected 8 total markers, got {my_pieces + opp_pieces + legal_moves}"
        )
        env.close()


class TestStep:
    def test_step_with_valid_action(self):
        """A valid non-terminal move should produce reward 0 and not be terminal."""
        env = Othello(num_envs=1)
        obs, _ = env.reset()

        legal_plane = obs[0, 128:]
        legal_moves = np.where(legal_plane > 0.5)[0]
        assert len(legal_moves) > 0

        actions = np.array([legal_moves[0]], dtype=np.int32)
        obs, rewards, terminals, truncations, infos = env.step(actions)

        if not terminals[0]:
            assert rewards[0] == 0.0
        env.close()

    def test_step_with_invalid_action(self):
        """An invalid move should produce reward -1 and be terminal."""
        env = Othello(num_envs=1)
        obs, _ = env.reset()

        # Find an illegal empty square
        legal_plane = obs[0, 128:]
        my_plane = obs[0, :64]
        opp_plane = obs[0, 64:128]

        illegal_moves = [
            m
            for m in range(64)
            if legal_plane[m] < 0.5 and my_plane[m] < 0.5 and opp_plane[m] < 0.5
        ]
        assert len(illegal_moves) > 0

        actions = np.array([illegal_moves[0]], dtype=np.int32)
        obs, rewards, terminals, truncations, infos = env.step(actions)

        assert terminals[0]
        assert rewards[0] == -1.0
        env.close()


class TestMultipleEnvs:
    def test_multiple_envs(self):
        """Multiple environments should produce correct batch observation shape."""
        env = Othello(num_envs=4)
        obs, _ = env.reset()
        assert obs.shape == (4, 192)
        env.close()
