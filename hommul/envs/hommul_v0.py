import numpy as np

from pettingzoo.utils.env import ParallelEnv
from gymnasium.spaces import Discrete, Box

class HomMul(ParallelEnv):
    metadata = {"render_mode": ["human"]}
    def __init__(self, render_mode=None):
        self.render_mode = render_mode

        self.num_days = 2
        self.num_life_points = 3
        
        self.action_dict = {0: "wait", 1: "play", 2: "none"}
        self.num_actions = len(self.action_dict)

        self.possible_agents = ["player1", "player2"]
        self.agents = self.possible_agents[:]

    def reset(self, seed=None, options=None):
        pass

    def step(self, actions):
        pass

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def render(self):
        pass