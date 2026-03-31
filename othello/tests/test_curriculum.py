"""Tests for the opponent curriculum scheduler."""
import sys
import os
import numpy as np
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from curriculum import CurriculumScheduler, SelfPlayPool, PHASE_BOUNDARIES


class TestCurriculumPhases:
    def test_phase_at_step_zero_is_random(self):
        sched = CurriculumScheduler(total_timesteps=200_000_000)
        phase = sched.get_phase(0)
        assert phase["type"] == "random"

    def test_phase_at_5_percent_is_negamax_d1(self):
        sched = CurriculumScheduler(total_timesteps=200_000_000)
        phase = sched.get_phase(10_000_001)
        assert phase["type"] == "negamax"
        assert phase["depth"] == 1

    def test_phase_at_15_percent_is_negamax_d2(self):
        sched = CurriculumScheduler(total_timesteps=200_000_000)
        phase = sched.get_phase(30_000_001)
        assert phase["type"] == "negamax"
        assert phase["depth"] == 2

    def test_phase_at_30_percent_is_negamax_d3(self):
        sched = CurriculumScheduler(total_timesteps=200_000_000)
        phase = sched.get_phase(60_000_001)
        assert phase["type"] == "negamax"
        assert phase["depth"] == 3

    def test_phase_at_50_percent_is_negamax_d5(self):
        sched = CurriculumScheduler(total_timesteps=200_000_000)
        phase = sched.get_phase(100_000_001)
        assert phase["type"] == "negamax"
        assert phase["depth"] == 5

    def test_phase_at_65_percent_is_self_play(self):
        sched = CurriculumScheduler(total_timesteps=200_000_000)
        phase = sched.get_phase(130_000_001)
        assert phase["type"] == "self_play"

    def test_difficulty_value_monotonically_increases(self):
        sched = CurriculumScheduler(total_timesteps=200_000_000)
        prev = -1.0
        for step in range(0, 200_000_000, 10_000_000):
            d = sched.get_difficulty(step)
            assert d >= prev
            prev = d

    def test_difficulty_at_end_is_one(self):
        sched = CurriculumScheduler(total_timesteps=200_000_000)
        assert sched.get_difficulty(200_000_000) == 1.0

    def test_update_sets_shared_value(self):
        sched = CurriculumScheduler(total_timesteps=200_000_000)
        sched.update(100_000_000)
        assert sched.difficulty_value.value == pytest.approx(0.5)


class TestSelfPlayPool:
    def test_pool_starts_empty(self):
        pool = SelfPlayPool()
        assert pool.size == 0
        assert pool.sample() is None

    def test_pool_accumulates_snapshots(self):
        import torch
        pool = SelfPlayPool(max_size=3, refresh_interval=100)
        policy = torch.nn.Linear(10, 5)
        pool.maybe_refresh(0, policy)
        assert pool.size == 1
        pool.maybe_refresh(100, policy)
        assert pool.size == 2
        pool.maybe_refresh(200, policy)
        assert pool.size == 3

    def test_pool_evicts_oldest_when_full(self):
        import torch
        pool = SelfPlayPool(max_size=2, refresh_interval=100)
        policy = torch.nn.Linear(10, 5)
        pool.maybe_refresh(0, policy)
        pool.maybe_refresh(100, policy)
        pool.maybe_refresh(200, policy)
        assert pool.size == 2

    def test_pool_does_not_refresh_too_soon(self):
        import torch
        pool = SelfPlayPool(max_size=5, refresh_interval=100)
        policy = torch.nn.Linear(10, 5)
        pool.maybe_refresh(0, policy)
        pool.maybe_refresh(50, policy)  # Too soon
        assert pool.size == 1

    def test_pool_sample_returns_state_dict(self):
        import torch
        pool = SelfPlayPool(max_size=3, refresh_interval=100)
        policy = torch.nn.Linear(10, 5)
        pool.maybe_refresh(0, policy)
        snapshot = pool.sample()
        assert snapshot is not None
        assert "weight" in snapshot

    def test_pool_snapshots_are_independent_copies(self):
        import torch
        pool = SelfPlayPool(max_size=3, refresh_interval=100)
        policy = torch.nn.Linear(10, 5)
        pool.maybe_refresh(0, policy)
        # Modify original policy
        with torch.no_grad():
            policy.weight.fill_(999.0)
        snapshot = pool.sample()
        assert not torch.allclose(snapshot["weight"], torch.tensor(999.0))
