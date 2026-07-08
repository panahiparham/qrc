import numpy as np
import pytest
from ReplayTables.ReplayBuffer import Timestep
from algorithms.buffers.iid import IIDBuffer, IIDConfig


def make_buffer(max_size=100):
    rng = np.random.default_rng(0)
    return IIDBuffer(max_size=max_size, lag=1, rng=rng, config=IIDConfig(), collector=None)


def add_transitions(buf, n=10, obs_dim=4):
    for i in range(n):
        buf.add_step(Timestep(
            x=np.random.randn(obs_dim).astype(np.float32),
            a=i % 2,
            r=float(i),
            gamma=0.99,
            terminal=False,
        ))


def test_buffer_starts_empty():
    buf = make_buffer()
    assert buf.size() == 0


def test_buffer_size_grows():
    buf = make_buffer()
    add_transitions(buf, n=5)
    assert buf.size() > 0


def test_buffer_respects_max_size():
    buf = make_buffer(max_size=10)
    add_transitions(buf, n=20)
    assert buf.size() <= 10


def test_sample_batch_size():
    buf = make_buffer(max_size=200)
    add_transitions(buf, n=100)
    batch = buf.sample(32)
    assert batch.x.shape[0] == 32


def test_sample_fields_present():
    buf = make_buffer(max_size=200)
    add_transitions(buf, n=100)
    batch = buf.sample(16)
    assert hasattr(batch, 'x')
    assert hasattr(batch, 'a')
    assert hasattr(batch, 'r')
    assert hasattr(batch, 'gamma')
    assert hasattr(batch, 'xp')


def test_sample_obs_shape():
    obs_dim = 6
    buf = IIDBuffer(max_size=200, lag=1, rng=np.random.default_rng(0), config=IIDConfig(), collector=None)
    for _ in range(100):
        buf.add_step(Timestep(
            x=np.zeros(obs_dim, dtype=np.float32),
            a=0,
            r=0.0,
            gamma=0.99,
            terminal=False,
        ))
    batch = buf.sample(8)
    assert batch.x.shape == (8, obs_dim)
    assert batch.xp.shape == (8, obs_dim)


def test_update_priorities_noop():
    buf = make_buffer()
    add_transitions(buf, n=10)
    batch = buf.sample(5)
    # should not raise
    buf.update_priorities(batch, np.ones(5))
