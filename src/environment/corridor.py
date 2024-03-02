import gymnasium as gym
from gymnasium.spaces import Discrete, Box
from ray.rllib.env.env_context import EnvContext
import numpy as np
import os
import random


class SimpleCorridor(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.

    You can configure the length of the corridor via the env config."""

    def __init__(self, config: EnvContext):
        self.end_pos = config["corridor_length"]
        self.cur_pos = 0
        self.action_space = Discrete(2)
        self.observation_space = Box(0, self.end_pos, shape=(1,), dtype=np.float32)  
        # Set the seed. This is only used for the final (reach goal) reward.
        self.reset(seed=42)

    def reset(self, *, seed=None, options=None):
        random.seed(seed)
        self.cur_pos = 0
        #return np.array([self.cur_pos]).astype(np.float32), {}  # convert return type to  np.float32
        return self.observation_space.sample(), {}

    def step(self, action):
        assert action in [0, 1], action
        if action == 0 and self.cur_pos > 0:
            self.cur_pos -= 1
        elif action == 1:
            self.cur_pos += 1
        done = truncated = self.cur_pos >= self.end_pos
        # Produce a random reward when we reach the goal.
        return (
            np.array([self.cur_pos]).astype(np.float32),  # convert return type to  np.float32
            random.random() * 2 if done else -0.1,
            done,
            truncated,
            {},
        )


def simple_corridor_creator(env_config):
    return SimpleCorridor(env_config)
