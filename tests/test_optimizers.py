import numpy as np
import pytest
from PyExpUtils.collection.Collector import Collector
from PyExpUtils.collection.Sampler import Ignore

from algorithms.nn.DQN import DQN
from algorithms.optimizers.registry import buildOptimizer


def make_collector():
    return Collector(config={}, default=Ignore())


def make_dqn_params(optimizer):
    return {
        'epsilon': 0.1,
        'buffer_type': 'iid',
        'buffer_size': 1000,
        'batch': 32,
        'target_refresh': 4,
        'optimizer': optimizer,
        'representation': {'type': 'TwoLayerRelu', 'hidden': 8},
        'gamma': 0.99,
        'n_step': 1,
    }


def test_defaults_to_adam():
    opt = buildOptimizer({'learning_rate': 1e-3})
    expected = buildOptimizer({'name': 'ADAM', 'learning_rate': 1e-3, 'b1': 0.9, 'b2': 0.999, 'eps': 1e-8})

    params = {'w': np.zeros(3, dtype=np.float32)}
    assert repr(opt.init(params)) == repr(expected.init(params))


def test_sgd():
    opt = buildOptimizer({'name': 'SGD', 'learning_rate': 0.5})

    params = {'w': np.ones(3, dtype=np.float32)}
    grads = {'w': np.ones(3, dtype=np.float32)}
    state = opt.init(params)
    updates, _ = opt.update(grads, state, params)
    assert np.allclose(updates['w'], -0.5)


def test_unknown_optimizer_raises():
    with pytest.raises(NotImplementedError):
        buildOptimizer({'name': 'adagrad', 'learning_rate': 1e-3})


@pytest.mark.parametrize('optimizer', [
    {'name': 'ADAM', 'learning_rate': 1e-3, 'b1': 0.9, 'b2': 0.999, 'eps': 1e-8},
    {'name': 'SGD', 'learning_rate': 1e-3},
    {'learning_rate': 1e-3},
])
def test_agent_runs_with_optimizer(optimizer):
    agent = DQN(
        observations=(4,),
        actions=2,
        params=make_dqn_params(optimizer),
        collector=make_collector(),
        seed=0,
    )

    obs = np.zeros(4, dtype=np.float32)
    a = agent.start(obs)
    assert a in (0, 1)

    for i in range(40):
        obs = np.random.randn(4).astype(np.float32)
        a = agent.step(float(i % 3), obs, {})
        assert a in (0, 1)
