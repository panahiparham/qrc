import numpy as np
import jax
import jax.numpy as jnp
import pytest
from representations.networks import NetworkBuilder


def make_builder(obs_shape=(4,), net_type='TwoLayerRelu', hidden=8, seed=0):
    params = {'type': net_type, 'hidden': hidden}
    return NetworkBuilder(input_shape=obs_shape, params=params, seed=seed)


def test_twolayer_relu_builds():
    builder = make_builder(net_type='TwoLayerRelu')
    params = builder.getParams()
    assert 'phi' in params
    assert len(params) > 0


def test_twolayer_relu_output_shape():
    obs_shape = (4,)
    hidden = 8
    builder = make_builder(obs_shape=obs_shape, net_type='TwoLayerRelu', hidden=hidden)
    phi = builder.getFeatureFunction()
    params = builder.getParams()

    x = jnp.zeros((1,) + obs_shape)
    out = phi(params, x)
    assert out.out.shape == (1, hidden)


def test_onelayer_relu_builds():
    builder = make_builder(net_type='OneLayerRelu', hidden=16)
    params = builder.getParams()
    assert 'phi' in params


def test_onelayer_relu_output_shape():
    obs_shape = (6,)
    hidden = 16
    builder = make_builder(obs_shape=obs_shape, net_type='OneLayerRelu', hidden=hidden)
    phi = builder.getFeatureFunction()
    params = builder.getParams()

    x = jnp.zeros((1,) + obs_shape)
    out = phi(params, x)
    assert out.out.shape == (1, hidden)


def test_empty_net_builds():
    builder = make_builder(obs_shape=(4,), net_type='EmptyNet', hidden=0)
    params = builder.getParams()
    assert params is not None


def test_unknown_network_raises():
    with pytest.raises(NotImplementedError):
        builder = make_builder(net_type='AtariNet', hidden=64)
        # The error is raised during NetworkBuilder.__init__ when building the feature net


def test_add_linear_head():
    import haiku as hk
    obs_shape = (4,)
    hidden = 8
    n_actions = 3

    builder = make_builder(obs_shape=obs_shape, net_type='TwoLayerRelu', hidden=hidden)
    q_fn = builder.addHead(lambda: hk.Linear(n_actions, name='q'))
    params = builder.getParams()

    phi_fn = builder.getFeatureFunction()
    x = jnp.zeros((1,) + obs_shape)
    phi_out = phi_fn(params, x).out
    q_out = q_fn(params, phi_out)
    assert q_out.shape == (1, n_actions)
