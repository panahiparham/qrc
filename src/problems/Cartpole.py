import numpy as np
from environments.Gym import Gym
from PyExpUtils.collection.Collector import Collector
from experiment.ExperimentModel import ExperimentModel
from problems.BaseProblem import BaseProblem


class Cartpole(BaseProblem):
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
        super().__init__(exp, idx, collector)
        self.env = Gym(name='CartPole-v1', seed=self.seed)
        self.actions = 2
        self.observations = (4,)
        self.gamma = 0.99

        x_thresh = 4.8
        theta_thresh = 12 * 2 * np.pi / 360
        self.rep_params['input_ranges'] = [
            [-x_thresh, x_thresh],
            [-6, 6],
            [-theta_thresh, theta_thresh],
            [-2.0, 2.0],
        ]
