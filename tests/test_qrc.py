import numpy as np
import pytest
from ReplayTables.ReplayBuffer import Timestep
from PyExpUtils.collection.Collector import Collector
from PyExpUtils.collection.Sampler import Ignore, Identity, MovingAverage

from algorithms.nn.QRC import QRC
from algorithms.nn.NNAgent import AgentState


def make_collector():
    return Collector(
        config={'v_loss': Identity(), 'h_loss': Identity()},
        default=Ignore(),
    )


def make_qrc_params(lr=1e-3, hidden=8, batch=32):
    return {
        'epsilon': 0.1,
        'buffer_type': 'iid',
        'buffer_size': 1000,
        'batch': batch,
        'beta': 1.0,
        'eta': 1.0,
        'optimizer': {
            'name': 'ADAM',
            'learning_rate': lr,
            'b1': 0.9,
            'b2': 0.999,
            'eps': 1e-8,
        },
        'representation': {'type': 'TwoLayerRelu', 'hidden': hidden},
        'gamma': 0.99,
        'n_step': 1,
    }


def fill_buffer(agent, n=100, obs_dim=4, n_actions=2):
    for i in range(n):
        obs = np.random.randn(obs_dim).astype(np.float32)
        agent.buffer.add_step(Timestep(
            x=obs,
            a=i % n_actions,
            r=float(i % 10),
            gamma=0.99,
            terminal=False,
        ))


def test_qrc_construction():
    agent = QRC(
        observations=(4,),
        actions=2,
        params=make_qrc_params(),
        collector=make_collector(),
        seed=0,
    )
    assert agent is not None
    assert agent.state.params is not None


def test_qrc_no_target_params():
    agent = QRC(
        observations=(4,),
        actions=2,
        params=make_qrc_params(),
        collector=make_collector(),
        seed=0,
    )
    assert not hasattr(agent.state, 'target_params'), 'QRC should not have target_params'


def test_qrc_has_h_head():
    agent = QRC(
        observations=(4,),
        actions=2,
        params=make_qrc_params(),
        collector=make_collector(),
        seed=0,
    )
    assert hasattr(agent, 'h'), 'QRC must have an h-head (core to qc_loss)'


def test_qrc_values_shape():
    agent = QRC(
        observations=(4,),
        actions=2,
        params=make_qrc_params(),
        collector=make_collector(),
        seed=0,
    )
    obs = np.zeros(4, dtype=np.float32)
    q = agent.values(obs)
    assert q.shape == (2,)


def test_qrc_triggers_update():
    agent = QRC(
        observations=(4,),
        actions=2,
        params=make_qrc_params(batch=16),
        collector=make_collector(),
        seed=0,
    )
    fill_buffer(agent, n=50, obs_dim=4)
    for _ in range(20):
        agent.update()
    assert agent.updates > 0


def test_qrc_h_loss_emitted():
    collected = {}

    class CapturingCollector:
        def collect(self, key, value):
            collected[key] = value

        def setIdx(self, idx):
            pass

    agent = QRC(
        observations=(4,),
        actions=2,
        params=make_qrc_params(batch=16),
        collector=CapturingCollector(),
        seed=0,
    )
    fill_buffer(agent, n=50, obs_dim=4)
    for _ in range(5):
        agent.update()

    if agent.updates > 0:
        assert 'h_loss' in collected, 'QRC should emit h_loss each update step'
        assert 'v_loss' in collected, 'QRC should emit v_loss each update step'
