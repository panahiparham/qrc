import numpy as np
import rlglue.agent

from typing import Dict, Tuple
from PyExpUtils.collection.Collector import Collector


class BaseAgent(rlglue.agent.BaseAgent):
    def __init__(self, observations: Tuple[int, ...], actions: int, params: Dict, collector: Collector, seed: int):
        self.observations = observations
        self.actions = actions
        self.params = params
        self.collector = collector

        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.policy_rng = np.random.default_rng(seed)

        self.gamma = params.get('gamma', 1)
        self.n_step = params.get('n_step', 1)

    def cleanup(self):
        ...
