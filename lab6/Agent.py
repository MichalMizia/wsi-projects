import gymnasium as gym
import numpy as np
from typing import Tuple
from collections import defaultdict


class Agent:
    def __init__(
        self,
        env: gym.Env,
        epsilon: float,
        min_epsilon=0.01,
        epsilon_decay=0.001,
        learning_rate=0.1,
        discount_factor=0.9,
    ) -> None:
        self.env = env
        self.q_values = []

        self.epsilon = epsilon
        self.starting_epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.reset_q_values()
        self.training_errors = []

    def reset_q_values(self):
        self.q_values = np.zeros(
            (self.env.observation_space.n, self.env.action_space.n)  # type: ignore
        )
        self.epsilon = self.starting_epsilon

    def get_action(self, obs: int):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_values[obs])

    def update(self, state, new_state, action, reward) -> None:
        target = reward + self.discount_factor * np.max(self.q_values[new_state])
        error = target - self.q_values[state][action]
        self.training_errors.append(error)
        self.q_values[state][action] += self.lr * error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon - self.epsilon_decay, 0.01)
