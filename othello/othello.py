"""PufferEnv wrapper for the Othello C engine."""

import gymnasium
import numpy as np
import pufferlib

from . import binding


class Othello(pufferlib.PufferEnv):
    """Vectorized Othello environment compatible with PufferLib training."""

    def __init__(
        self,
        num_envs=1,
        render_mode=None,
        report_interval=128,
        buf=None,
        seed=0,
        opponent_type="random",
        opponent_depth=0,
        curriculum_difficulty_value=None,
        self_play_pool=None,
    ):
        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=1, shape=(binding.OBS_DIM,), dtype=np.float32
        )
        self.single_action_space = gymnasium.spaces.Discrete(binding.NUM_ACTIONS)
        self.num_agents = num_envs
        self.render_mode = render_mode
        self.report_interval = report_interval
        self._seed = seed
        self.opponent_type = opponent_type
        self.opponent_depth = opponent_depth
        self.curriculum_difficulty_value = curriculum_difficulty_value
        self.self_play_pool = self_play_pool

        self._step_count = 0
        self._closed = False

        super().__init__(buf=buf)

        self._c_env = binding.VecEnv()

        # Internal buffers for the C engine (int32 dones, not bool)
        self._c_actions = np.zeros(num_envs, dtype=np.int32)
        self._c_rewards = np.zeros(num_envs, dtype=np.float32)
        self._c_dones = np.zeros(num_envs, dtype=np.int32)

        self._c_env.init(
            num_envs, self.observations, self._c_actions, self._c_rewards, self._c_dones
        )

        self._opp_obs = np.zeros((num_envs, binding.OBS_DIM), dtype=np.float32)
        self._c_env.set_opp_obs(self._opp_obs)
        self._c_opp_actions = np.zeros(num_envs, dtype=np.int32)

    def reset(self, seed=None):
        self._c_env.reset()
        return self.observations, []

    def step(self, actions):
        np.copyto(self._c_actions, actions.ravel()[:self.num_agents].astype(np.int32))
        self._c_env.step()

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

    def step_negamax(self, agent_actions: np.ndarray, depth: int):
        """Apply agent moves then negamax opponent moves in one call."""
        self.step_agent(agent_actions)
        self._c_env.negamax_moves_batch(self._c_opp_actions, depth)
        return self.step_opponent(self._c_opp_actions)

    def render(self):
        if self.render_mode == "human":
            self._c_env.render()

    def close(self):
        if not self._closed:
            self._c_env.close()
            self._closed = True

    def __del__(self):
        self.close()
