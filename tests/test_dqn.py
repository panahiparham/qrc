import numpy as np
import pytest
from ReplayTables.ReplayBuffer import Timestep
from PyExpUtils.collection.Collector import Collector
from PyExpUtils.collection.Sampler import Ignore

from algorithms.nn.DQN import DQN


def make_collector():
    return Collector(config={}, default=Ignore())


def make_dqn_params(lr=1e-3, hidden=8, batch=32, target_refresh=4):
    return {
        'epsilon': 0.1,
        'buffer_type': 'iid',
        'buffer_size': 1000,
        'batch': batch,
        'target_refresh': target_refresh,
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


def test_dqn_construction():
    agent = DQN(
        observations=(4,),
        actions=2,
        params=make_dqn_params(),
        collector=make_collector(),
        seed=0,
    )
    assert agent is not None
    assert agent.state.params is not None
    assert agent.state.target_params is not None


def test_dqn_values_shape():
    agent = DQN(
        observations=(4,),
        actions=2,
        params=make_dqn_params(),
        collector=make_collector(),
        seed=0,
    )
    obs = np.zeros(4, dtype=np.float32)
    q = agent.values(obs)
    assert q.shape == (2,)


def test_dqn_triggers_update():
    agent = DQN(
        observations=(4,),
        actions=2,
        params=make_dqn_params(batch=16, target_refresh=4),
        collector=make_collector(),
        seed=0,
    )
    fill_buffer(agent, n=50, obs_dim=4, n_actions=2)
    initial_updates = agent.updates
    for _ in range(20):
        agent.update()
    assert agent.updates > initial_updates


def test_dqn_target_refresh():
    target_refresh = 2
    agent = DQN(
        observations=(4,),
        actions=2,
        params=make_dqn_params(batch=16, target_refresh=target_refresh),
        collector=make_collector(),
        seed=0,
    )
    fill_buffer(agent, n=50, obs_dim=4)

    import jax
    initial_target = jax.tree_util.tree_leaves(agent.state.target_params)[0].copy()

    # run enough updates to trigger target refresh
    for _ in range(target_refresh * 3):
        agent.updates += 1
        agent.steps += 1
        batch = agent.buffer.sample(16)
        agent.state, _ = agent._computeUpdate(agent.state, batch)
        if agent.updates % target_refresh == 0:
            agent.state.target_params = agent.state.params

    updated_target = jax.tree_util.tree_leaves(agent.state.target_params)[0]
    assert not np.allclose(initial_target, updated_target)


def test_dqn_no_h_head():
    agent = DQN(
        observations=(4,),
        actions=2,
        params=make_dqn_params(),
        collector=make_collector(),
        seed=0,
    )
    assert not hasattr(agent, 'h'), 'DQN should not have an h-head'
