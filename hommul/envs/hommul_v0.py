import numpy as np

from pettingzoo.utils.env import ParallelEnv
from gymnasium.spaces import Discrete, Box


class HomMul(ParallelEnv):
    metadata = {"render_mode": ["human"]}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode

        self.num_days = 3
        self.num_life_points = 4

        self.cost_wait = -1
        self.gain_play = 1
        self.cost_play = -2
        self.gain_dead = 0

        self.action_dict = {0: "wait", 1: "play", 2: "none"}
        self.num_actions = len(self.action_dict)

        self.possible_agents = ["player1", "player2"]
        self.agents = self.possible_agents[:]

        self.observation_spaces = {
            agent: Box(low=0, high=3, shape=(1, 7)) for agent in self.agents
        }

        self.action_spaces = {
            agent: Discrete(self.num_actions) for agent in self.agents
        }

    def _get_obs(self):
        return {
            "player1": {
                "observation": np.array(
                    [
                        [
                            self.days_left,
                            self.player1_life_points,
                            self.player2_life_points,
                            self.player1_prob_payoff,
                            self.player2_prob_payoff,
                            self.player1_action,
                            self.player2_action,
                        ]
                    ]
                ),
                "action_mask": [1, 1, 0],
            },
            "player2": {
                "observation": np.array(
                    [
                        [
                            self.days_left,
                            self.player1_life_points,
                            self.player2_life_points,
                            self.player1_prob_payoff,
                            self.player2_prob_payoff,
                            self.player1_action,
                            self.player2_action,
                        ]
                    ]
                ),
                "action_mask": [1, 1, 0],
            },
        }

    def _get_prob_payoffs(self):
        self.weather_type = np.random.choice([0, 1])

        if self.weather_type:
            self.player1_prob_payoff = 0.2
            self.player2_prob_payoff = 0.2
        else:
            self.player1_prob_payoff = 0.4
            self.player2_prob_payoff = 0.4

    def _get_payoffs(self):
        player1_possible_outcome = np.random.uniform(0, 1) <= self.player1_prob_payoff
        player2_possible_outcome = np.random.uniform(0, 1) <= self.player2_prob_payoff
        combined_possible_outcome = np.random.uniform(0, 1) <= (
            self.player1_prob_payoff + self.player2_prob_payoff
        )

        if self.player1_life_points != 0 and self.player2_life_points != 0:
            if self.player1_action and self.player2_action:
                if combined_possible_outcome:
                    player1_payoff, player2_payoff = self.gain_play, self.gain_play
                else:
                    player1_payoff, player2_payoff = self.cost_play, self.cost_play
            elif self.player1_action and not self.player2_action:
                if player1_possible_outcome:
                    player1_payoff, player2_payoff = self.gain_play, self.cost_wait
                else:
                    player1_payoff, player2_payoff = self.cost_play, self.cost_wait
            elif not self.player1_action and self.player2_action:
                if player2_possible_outcome:
                    player1_payoff, player2_payoff = self.cost_wait, self.gain_play
                else:
                    player1_payoff, player2_payoff = self.cost_wait, self.cost_play
            else:
                player1_payoff, player2_payoff = self.cost_wait, self.cost_wait
        elif self.player1_life_points != 0 and self.player2_life_points == 0:
            if self.player1_action:
                if player1_possible_outcome:
                    player1_payoff, player2_payoff = self.gain_play, self.gain_dead
                else:
                    player1_payoff, player2_payoff = self.cost_play, self.gain_dead
            else:
                player1_payoff, player2_payoff = self.cost_wait, self.gain_dead
        elif self.player1_life_points == 0 and self.player2_life_points != 0:
            if self.player2_action:
                if player2_possible_outcome:
                    player1_payoff, player2_payoff = self.gain_dead, self.gain_play
                else:
                    player1_payoff, player2_payoff = self.gain_dead, self.cost_play
            else:
                player1_payoff, player2_payoff = self.gain_dead, self.cost_wait
        else:
            player1_payoff, player2_payoff = self.gain_dead, self.gain_dead

        return player1_payoff, player2_payoff

    def _get_rewards(self):
        return {
            "player1": -1 if self.player1_life_points == 0 else 0,
            "player2": -1 if self.player2_life_points == 0 else 0,
        }

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.days_left = np.random.randint(1, self.num_days)

        life_point_perms = [[1, 1], [1, 2], [2, 2], [1, 3], [3, 3]]

        if self.days_left != 1:
            self.player1_life_points = np.random.randint(1, self.num_life_points)
            self.player2_life_points = np.random.randint(1, self.num_life_points)
        else:
            rng = np.random.default_rng()
            self.player1_life_points, self.player2_life_points = rng.choice(life_point_perms, 1, axis=0)[0]

        self.player1_action = 2
        self.player2_action = 2

        self._get_prob_payoffs()

        if self.render_mode == "human":
            self.render_text(is_start=True)

        return self._get_obs()

    def step(self, actions):
        self.player1_action = actions["player1"]
        self.player2_action = actions["player2"]

        player1_payoff, player2_payoff = self._get_payoffs()

        self.player1_life_points += player1_payoff
        self.player1_life_points = np.clip(
            self.player1_life_points, 0, self.num_life_points - 1
        )
        self.player2_life_points += player2_payoff
        self.player2_life_points = np.clip(
            self.player2_life_points, 0, self.num_life_points - 1
        )

        rewards = {a: 0 for a in self.agents}
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        infos = {a: {} for a in self.agents}

        self.days_left -= 1
        self._get_prob_payoffs()

        if self.render_mode == "human":
            self.render_text()

        if self.days_left == 0 or (self.player1_life_points == 0 and self.player2_life_points == 0):
            truncations = {a: True for a in self.agents}
            terminations = {a: False for a in self.agents}
            infos = {a: {} for a in self.agents}

            self.agents = []
            return (
                self._get_obs(),
                self._get_rewards(),
                terminations,
                truncations,
                infos,
            )

        return self._get_obs(), rewards, terminations, truncations, infos

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def render(self):
        if self.render_mode == "human":
            self.render_text()

    def render_text(self, is_start=False):
        print(f"--Days left: {self.days_left}")
        print(f"--Current life of agent 1: {self.player1_life_points}")
        print(f"--Current life of agent 2: {self.player2_life_points}")
        print(f"--Probability of payoff for agent 1: {self.player1_prob_payoff}")
        print(f"--Probability of payoff for agent 2: {self.player2_prob_payoff}")

        if not is_start:
            print(
                f"--Previous action of agent 1: {self.action_dict[self.player1_action]}"
            )
            print(
                f"--Previous action of agent 2: {self.action_dict[self.player2_action]}"
            )
