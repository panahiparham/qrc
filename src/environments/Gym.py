from typing import Optional
import gymnasium
from rlglue.environment import BaseEnvironment


class Gym(BaseEnvironment):
    def __init__(self, name: str, seed: int, max_steps: Optional[int] = None):
        self.env = gymnasium.make(name, max_episode_steps=max_steps if max_steps and max_steps > 0 else None)
        self.seed = seed

    def start(self):
        self.seed += 1
        s, info = self.env.reset(seed=self.seed)
        return s

    def step(self, a):
        sp, r, term, trunc, info = self.env.step(a)
        return (sp, r, term, trunc, {})
