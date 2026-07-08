import numpy as np
import pytest
from environments.Gym import Gym


def test_start_returns_obs():
    env = Gym(name='Acrobot-v1', seed=42)
    obs = env.start()
    assert obs is not None
    assert hasattr(obs, 'shape')


def test_start_obs_shape():
    env = Gym(name='Acrobot-v1', seed=42)
    obs = env.start()
    assert obs.shape == (6,)


def test_cartpole_obs_shape():
    env = Gym(name='CartPole-v1', seed=0)
    obs = env.start()
    assert obs.shape == (4,)


def test_mountaincar_obs_shape():
    env = Gym(name='MountainCar-v0', seed=0)
    obs = env.start()
    assert obs.shape == (2,)


def test_step_returns_5_tuple():
    env = Gym(name='Acrobot-v1', seed=42)
    env.start()
    result = env.step(0)
    assert len(result) == 5, f'Expected 5-tuple, got {len(result)}-tuple'


def test_step_obs_shape():
    env = Gym(name='Acrobot-v1', seed=42)
    env.start()
    obs, reward, term, trunc, extra = env.step(0)
    assert obs.shape == (6,)


def test_step_reward_is_float():
    env = Gym(name='Acrobot-v1', seed=42)
    env.start()
    obs, reward, term, trunc, extra = env.step(0)
    assert isinstance(reward, float)


def test_step_term_trunc_are_bool():
    env = Gym(name='Acrobot-v1', seed=42)
    env.start()
    obs, reward, term, trunc, extra = env.step(0)
    assert isinstance(term, bool)
    assert isinstance(trunc, bool)


def test_step_extra_is_dict():
    env = Gym(name='Acrobot-v1', seed=42)
    env.start()
    obs, reward, term, trunc, extra = env.step(0)
    assert isinstance(extra, dict)


def test_seed_increments_on_start():
    env = Gym(name='CartPole-v1', seed=10)
    initial_seed = env.seed
    env.start()
    assert env.seed == initial_seed + 1
