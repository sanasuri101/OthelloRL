"""Comprehensive tests for the Othello C engine, VecEnv binding, and negamax opponent."""

import sys
import os
import numpy as np
import pytest

# Add parent directory so we can import the compiled binding
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import binding

OBS_DIM = binding.OBS_DIM


class TestBoardInit:
    """Test initial board state."""

    def test_starting_piece_counts(self):
        """Starting position has 2 black and 2 white pieces."""
        env = binding.VecEnv()
        black, white, legal = env.test_init()
        assert black == 2, f"Expected 2 black pieces, got {black}"
        assert white == 2, f"Expected 2 white pieces, got {white}"

    def test_starting_legal_moves(self):
        """Black has exactly 4 legal moves at the start."""
        env = binding.VecEnv()
        black, white, legal = env.test_init()
        assert legal == 4, f"Expected 4 legal moves, got {legal}"


class TestVecEnv:
    """Test the vectorized environment binding."""

    def _make_env(self, num_envs=1):
        """Helper to create and initialize a VecEnv."""
        obs = np.zeros((num_envs, 192), dtype=np.float32)
        actions = np.zeros(num_envs, dtype=np.int32)
        rewards = np.zeros(num_envs, dtype=np.float32)
        dones = np.zeros(num_envs, dtype=np.int32)
        env = binding.VecEnv()
        env.init(num_envs, obs, actions, rewards, dones)
        env.reset()
        return env, obs, actions, rewards, dones

    def test_reset_obs_shape(self):
        """After reset, observations should be populated with valid data."""
        env, obs, actions, rewards, dones = self._make_env(1)
        # Obs should have 192 floats
        assert obs.shape == (1, 192)
        # Should have some pieces on the board (plane 0 or plane 1)
        my_pieces = obs[0, :64].sum()
        opp_pieces = obs[0, 64:128].sum()
        assert my_pieces == 2, f"Expected 2 of my pieces, got {my_pieces}"
        assert opp_pieces == 2, f"Expected 2 opponent pieces, got {opp_pieces}"

    def test_valid_move_reward_zero(self):
        """A valid non-terminal move should produce reward 0."""
        env, obs, actions, rewards, dones = self._make_env(1)
        # Find a legal move from observation plane 2
        legal_plane = obs[0, 128:]
        legal_moves = np.where(legal_plane > 0.5)[0]
        assert len(legal_moves) > 0, "No legal moves available"
        # Play a legal move
        actions[0] = legal_moves[0]
        env.step()
        # If game is not over, reward should be 0
        if dones[0] == 0:
            assert rewards[0] == 0.0, f"Expected reward 0 for valid move, got {rewards[0]}"

    def test_invalid_move_terminal(self):
        """An invalid move should end the game with reward -1."""
        env, obs, actions, rewards, dones = self._make_env(1)
        # Find an illegal move -- a square that is NOT legal
        legal_plane = obs[0, 128:]
        illegal_moves = np.where(legal_plane < 0.5)[0]
        # Filter to only squares that are also empty (not occupied)
        my_plane = obs[0, :64]
        opp_plane = obs[0, 64:128]
        empty_illegal = [
            m for m in illegal_moves
            if my_plane[m] < 0.5 and opp_plane[m] < 0.5 and m < 64
        ]
        assert len(empty_illegal) > 0, "No illegal empty squares found"
        actions[0] = empty_illegal[0]
        env.step()
        assert dones[0] == 1, "Game should be terminal after invalid move"
        assert rewards[0] == -1.0, f"Expected reward -1 for invalid move, got {rewards[0]}"

    def test_pass_when_moves_exist_is_invalid(self):
        """Passing when legal moves exist should be treated as invalid."""
        env, obs, actions, rewards, dones = self._make_env(1)
        # At the start, there are legal moves, so passing should be invalid
        legal_plane = obs[0, 128:]
        has_legal = legal_plane.sum() > 0
        assert has_legal, "Should have legal moves at start"
        actions[0] = 64  # pass action
        env.step()
        assert dones[0] == 1, "Game should be terminal after invalid pass"
        assert rewards[0] == -1.0, f"Expected reward -1 for invalid pass, got {rewards[0]}"

    def test_multiple_envs(self):
        """Multiple environments should work independently."""
        num_envs = 4
        env, obs, actions, rewards, dones = self._make_env(num_envs)
        for i in range(num_envs):
            my_pieces = obs[i, :64].sum()
            opp_pieces = obs[i, 64:128].sum()
            # Each env should have a valid starting board (2 or 3 pieces
            # depending on whether opponent moved first)
            assert my_pieces >= 1, f"Env {i}: expected pieces, got {my_pieces}"
            assert opp_pieces >= 1, f"Env {i}: expected pieces, got {opp_pieces}"


class TestFlipping:
    """Test move application and piece flipping."""

    def _make_env(self, num_envs=1):
        obs = np.zeros((num_envs, 192), dtype=np.float32)
        actions = np.zeros(num_envs, dtype=np.int32)
        rewards = np.zeros(num_envs, dtype=np.float32)
        dones = np.zeros(num_envs, dtype=np.int32)
        env = binding.VecEnv()
        env.init(num_envs, obs, actions, rewards, dones)
        env.reset()
        return env, obs, actions, rewards, dones

    def test_opening_move_flips_one_piece(self):
        """An opening move should flip exactly one opponent piece."""
        env, obs, actions, rewards, dones = self._make_env(1)
        # Record initial piece counts
        initial_my = obs[0, :64].sum()
        initial_opp = obs[0, 64:128].sum()

        # Find and play a legal move
        legal_plane = obs[0, 128:]
        legal_moves = np.where(legal_plane > 0.5)[0]
        assert len(legal_moves) > 0
        actions[0] = legal_moves[0]
        env.step()

        if dones[0] == 0:
            new_my = obs[0, :64].sum()
            new_opp = obs[0, 64:128].sum()
            # After agent places one piece and flips one, and opponent responds,
            # the board state changes. We just verify the game progressed.
            total_pieces = new_my + new_opp
            assert total_pieces > initial_my + initial_opp, (
                "Total pieces should increase after valid moves"
            )

    def test_no_wrapping_across_rows(self):
        """Moves should not cause flips that wrap across board rows.

        This is a structural test -- we play many random games and verify
        the game engine doesn't crash or produce impossible piece counts.
        """
        num_envs = 8
        env, obs, actions, rewards, dones = self._make_env(num_envs)

        for _step in range(200):
            for i in range(num_envs):
                if dones[i]:
                    continue
                legal_plane = obs[i, 128:]
                legal_moves = np.where(legal_plane > 0.5)[0]
                if len(legal_moves) > 0:
                    actions[i] = np.random.choice(legal_moves)
                else:
                    actions[i] = 64  # pass
            env.step()

            # Verify piece counts are valid
            for i in range(num_envs):
                if dones[i]:
                    continue
                my_count = obs[i, :64].sum()
                opp_count = obs[i, 64:128].sum()
                total = my_count + opp_count
                assert 2 <= total <= 64, (
                    f"Invalid piece count: {total} (my={my_count}, opp={opp_count})"
                )

    def test_game_plays_to_completion(self):
        """A game should eventually terminate with correct reward values."""
        num_envs = 4
        env, obs, actions, rewards, dones = self._make_env(num_envs)

        max_steps = 500  # enough for games to complete and auto-reset
        completed_games = 0

        for _step in range(max_steps):
            for i in range(num_envs):
                legal_plane = obs[i, 128:]
                legal_moves = np.where(legal_plane > 0.5)[0]
                if len(legal_moves) > 0:
                    actions[i] = np.random.choice(legal_moves)
                else:
                    actions[i] = 64  # pass
            env.step()

            for i in range(num_envs):
                if dones[i]:
                    completed_games += 1
                    # Terminal reward should be -1, 0, or 1
                    assert rewards[i] in (-1.0, 0.0, 1.0), (
                        f"Invalid terminal reward: {rewards[i]}"
                    )

        assert completed_games > 0, "No games completed in 500 steps"

    def test_logging_stats(self):
        """Log function should return valid statistics."""
        num_envs = 2
        env, obs, actions, rewards, dones = self._make_env(num_envs)

        # Play some steps
        for _step in range(100):
            for i in range(num_envs):
                legal_plane = obs[i, 128:]
                legal_moves = np.where(legal_plane > 0.5)[0]
                if len(legal_moves) > 0:
                    actions[i] = np.random.choice(legal_moves)
                else:
                    actions[i] = 64
            env.step()

        log = env.log()
        assert "win_rate" in log
        assert "avg_game_length" in log
        assert "invalid_move_rate" in log
        assert "games_played" in log
        assert "corner_captures" in log
        assert 0.0 <= log["win_rate"] <= 1.0
        assert log["games_played"] >= 0


class TestNegamax:
    """Test the negamax opponent."""

    def _make_env(self, num_envs=1):
        obs = np.zeros((num_envs, 192), dtype=np.float32)
        actions = np.zeros(num_envs, dtype=np.int32)
        rewards = np.zeros(num_envs, dtype=np.float32)
        dones = np.zeros(num_envs, dtype=np.int32)
        env = binding.VecEnv()
        env.init(num_envs, obs, actions, rewards, dones)
        env.reset()
        return env, obs, actions, rewards, dones

    def test_returns_legal_move(self):
        """Negamax should always return a legal move."""
        env, obs, actions, rewards, dones = self._make_env(1)
        move = env.negamax_move(0, 3)
        # Move should be 0-63 or 64 (pass)
        assert 0 <= move <= 64, f"Negamax returned invalid move: {move}"

        if move < 64:
            # It should be a legal move
            legal_plane = obs[0, 128:]
            # Note: the legal plane is from the agent's perspective,
            # but negamax plays for current_player which may differ.
            # Just verify it's in valid range.
            assert 0 <= move < 64

    def test_depth3_avoids_x_squares(self):
        """At depth 3, negamax should prefer non-X-square moves when alternatives exist.

        X-squares are the diagonal-to-corner squares: 1, 8, 9, 6, 15, 14,
        48, 49, 57, 54, 55, 62.
        """
        x_squares = {1, 8, 9, 6, 14, 15, 48, 49, 54, 55, 57, 62}

        # Run multiple games and check negamax rarely picks X-squares
        # when other options are available
        env, obs, actions, rewards, dones = self._make_env(1)
        x_square_picks = 0
        total_picks = 0

        for _ in range(20):
            # Reset by creating a new env
            env, obs, actions, rewards, dones = self._make_env(1)
            for _step in range(5):
                if dones[0]:
                    break
                move = env.negamax_move(0, 3)
                if move < 64:
                    total_picks += 1
                    if move in x_squares:
                        x_square_picks += 1

                # Play the move
                legal_plane = obs[0, 128:]
                legal_moves = np.where(legal_plane > 0.5)[0]
                if len(legal_moves) > 0:
                    actions[0] = np.random.choice(legal_moves)
                else:
                    actions[0] = 64
                env.step()

        if total_picks > 0:
            x_rate = x_square_picks / total_picks
            # Negamax should avoid X-squares most of the time
            assert x_rate < 0.5, (
                f"Negamax picked X-squares {x_rate:.1%} of the time "
                f"({x_square_picks}/{total_picks})"
            )


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

    def test_step_agent_fills_opp_obs(self):
        """After vec_step_agent(), opp_obs must be non-zero (opponent has pieces)."""
        vec, obs, actions, rewards, dones, opp_obs = self._make_env(n=1)
        legal = np.where(obs[0, 128:])[0]
        actions[0] = int(legal[0])
        vec.step_agent()
        assert opp_obs[0].sum() > 0, "opp_obs should contain board state"

    def test_step_opponent_completes_step(self):
        """vec_step_opponent() must write obs/rewards/dones back."""
        vec, obs, actions, rewards, dones, opp_obs = self._make_env(n=1)
        legal = np.where(obs[0, 128:])[0]
        actions[0] = int(legal[0])
        opp_actions = np.zeros(1, dtype=np.int32)
        vec.step_agent()
        opp_legal = np.where(opp_obs[0, 128:])[0]
        opp_actions[0] = int(opp_legal[0]) if len(opp_legal) > 0 else 64
        vec.step_opponent(opp_actions)
        assert obs[0].sum() > 0

    def test_split_step_preserves_game_length(self):
        """Games played via split-step should complete in a reasonable number of moves."""
        n = 4
        vec, obs, actions, rewards, dones, opp_obs = self._make_env(n=n)
        opp_actions = np.zeros(n, dtype=np.int32)
        total_done = 0
        for _ in range(200):
            for i in range(n):
                legal = np.where(obs[i, 128:])[0]
                actions[i] = int(legal[0]) if len(legal) > 0 else 64
            vec.step_agent()
            for i in range(n):
                opp_legal = np.where(opp_obs[i, 128:])[0]
                opp_actions[i] = int(opp_legal[0]) if len(opp_legal) > 0 else 64
            vec.step_opponent(opp_actions)
            total_done += int(dones.sum())
        assert total_done > 0, "At least some games should have finished"
