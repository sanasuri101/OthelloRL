"""Tests for othello/eval.py -- fixed-opponent evaluation."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from othello.eval import evaluate


def _make_dummy_policy(obs_size: int = 192, act_size: int = 65, hidden_size: int = 64):
    """Tiny policy that outputs uniform logits -- for structural testing only."""

    class DummyPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm_cell = nn.LSTMCell(hidden_size, hidden_size)
            self.encoder = nn.Linear(obs_size, hidden_size)
            self.actor = nn.Linear(hidden_size, act_size)
            self.critic = nn.Linear(hidden_size, 1)

        def forward_eval(self, obs, state):
            h = state["lstm_h"].squeeze(0)
            c = state["lstm_c"].squeeze(0)
            enc = torch.relu(self.encoder(obs.float()))
            h, c = self.lstm_cell(enc, (h, c))
            state["lstm_h"] = h.unsqueeze(0)
            state["lstm_c"] = c.unsqueeze(0)
            return self.actor(h), self.critic(h)

    return DummyPolicy()


class TestEvaluate:
    def test_returns_dict_keyed_by_depth(self):
        policy = _make_dummy_policy()
        results = evaluate(policy, depths=[1], n_games=5, device="cpu")
        assert isinstance(results, dict)
        assert 1 in results

    def test_win_rate_is_between_zero_and_one(self):
        policy = _make_dummy_policy()
        results = evaluate(policy, depths=[1], n_games=10, device="cpu")
        assert 0.0 <= results[1] <= 1.0

    def test_multiple_depths_all_present(self):
        policy = _make_dummy_policy()
        results = evaluate(policy, depths=[1, 2], n_games=5, device="cpu")
        assert set(results.keys()) == {1, 2}

    def test_n_games_respected(self):
        policy = _make_dummy_policy()
        results = evaluate(policy, depths=[1], n_games=3, device="cpu")
        wr = results[1]
        # win_rate must be a multiple of 1/3: 0, 1/3, 2/3, or 1
        assert any(abs(wr - k / 3) < 1e-9 for k in range(4))
