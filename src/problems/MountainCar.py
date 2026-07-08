from environments.Gym import Gym
from PyExpUtils.collection.Collector import Collector
from experiment.ExperimentModel import ExperimentModel
from problems.BaseProblem import BaseProblem


class MountainCar(BaseProblem):
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
        super().__init__(exp, idx, collector)
        self.env = Gym(name='MountainCar-v0', seed=self.seed, max_steps=500)
        self.actions = 3
        self.observations = (2,)
        self.gamma = 0.99

        self.rep_params['input_ranges'] = [
            [-1.2, 0.5],
            [-0.07, 0.07],
        ]
