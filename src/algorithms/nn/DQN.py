from functools import partial
from typing import Any, Dict, Tuple
from PyExpUtils.collection.Collector import Collector
from ReplayTables.ReplayBuffer import Batch

from algorithms.nn.NNAgent import NNAgent
from representations.networks import NetworkBuilder

import jax
import chex
import optax
import numpy as np
import haiku as hk
import jax.numpy as jnp
import utils.chex as cxu


@cxu.dataclass
class AgentState:
    params: Any
    target_params: Any
    optim: optax.OptState


@partial(jax.vmap, in_axes=0)
def dqn_loss(q, a, r, gamma, qp):
    vp = qp.max()
    target = r + gamma * vp
    target = jax.lax.stop_gradient(target)
    delta = target - q[a]
    q_loss = 0.5 * delta ** 2
    return q_loss, {'delta': delta}


class DQN(NNAgent):
    def __init__(self, observations: Tuple, actions: int, params: Dict, collector: Collector, seed: int):
        super().__init__(observations, actions, params, collector, seed)

        self.epsilon_decay_steps = params.get('epsilon_decay_steps', 0)
        self.min_epsilon = params.get('min_epsilon', 1.0)

        self.epsilon_decay = False
        if self.epsilon_decay_steps > 0:
            assert self.epsilon > self.min_epsilon, 'Epsilon must be greater than min_epsilon'
            self.epsilon_delta = (self.epsilon - self.min_epsilon) / self.epsilon_decay_steps
            self.epsilon_decay = True

        self.target_refresh = params['target_refresh']

        self.state = AgentState(
            params=self.state.params,
            target_params=self.state.params,
            optim=self.state.optim,
        )

    # ------------------------
    # -- NN agent interface --
    # ------------------------
    def _build_heads(self, builder: NetworkBuilder) -> None:
        if self.zero_init:
            _init = hk.initializers.Constant(0)
        else:
            _init = None

        self.q = builder.addHead(lambda: hk.Linear(self.actions, name='q', w_init=_init, b_init=_init, with_bias=self.bias_unit))

    @partial(jax.jit, static_argnums=0)
    def _values(self, state: AgentState, x: jax.Array):
        phi = self.phi(state.params, x).out
        return self.q(state.params, phi)

    def update(self):
        self.steps += 1

        if self.epsilon_decay:
            self.epsilon -= self.epsilon_delta
            self.epsilon = max(self.epsilon, self.min_epsilon)

        if self.steps % self.update_freq != 0:
            return

        if self.buffer.size() <= self.batch_size:
            return

        self.updates += 1

        batch = self.buffer.sample(self.batch_size)
        self.state, metrics = self._computeUpdate(self.state, batch)

        metrics = jax.device_get(metrics)
        for k, v in metrics.items():
            self.collector.collect(k, np.mean(v).item())

        if self.updates % self.target_refresh == 0:
            self.state.target_params = self.state.params

    # -------------
    # -- Updates --
    # -------------
    @partial(jax.jit, static_argnums=0)
    def _computeUpdate(self, state: AgentState, batch: Batch):
        grad_fn = jax.grad(self._loss, has_aux=True)
        grad, metrics = grad_fn(state.params, state.target_params, batch)

        updates, optim = self.optimizer.update(grad, state.optim, state.params)
        params = optax.apply_updates(state.params, updates)

        new_state = AgentState(
            params=params,
            target_params=state.target_params,
            optim=optim,
        )

        return new_state, metrics

    def _loss(self, params: hk.Params, target: hk.Params, batch: Batch):
        phi = self.phi(params, batch.x).out
        phi_p = self.phi(target, batch.xp).out

        qs = self.q(params, phi)
        qsp = self.q(target, phi_p)

        q_loss, metrics = dqn_loss(qs, batch.a, batch.r, batch.gamma, qsp)

        v_loss = jnp.mean(q_loss)
        metrics['v_loss'] = v_loss

        return v_loss, metrics
