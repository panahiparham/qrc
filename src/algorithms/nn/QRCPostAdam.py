from functools import partial
from typing import Dict, Tuple
from PyExpUtils.collection.Collector import Collector
from ReplayTables.interface import Batch

from algorithms.nn.NNAgent import NNAgent, AgentState
from representations.networks import NetworkBuilder

import jax
import optax
import numpy as np
import haiku as hk
import jax.numpy as jnp

tree_map = jax.tree_util.tree_map


class QRCPostAdam(NNAgent):
    def __init__(self, observations: Tuple, actions: int, params: Dict, collector: Collector, seed: int):
        super().__init__(observations, actions, params, collector, seed)

        self.beta = params.get('beta', 1.)
        self.eta = params.get('eta', 1.0)
        self.stepsize = self.optimizer_params['learning_rate']

        self.epsilon_decay_steps = params.get('epsilon_decay_steps', 0)
        self.min_epsilon = params.get('min_epsilon', 1.0)

        self.epsilon_decay = False
        if self.epsilon_decay_steps > 0:
            assert self.epsilon > self.min_epsilon, 'Epsilon must be greater than min_epsilon'
            self.epsilon_delta = (self.epsilon - self.min_epsilon) / self.epsilon_decay_steps
            self.epsilon_decay = True

    # ------------------------
    # -- NN agent interface --
    # ------------------------
    def _build_heads(self, builder: NetworkBuilder) -> None:
        if self.zero_init:
            _init = hk.initializers.Constant(0)
        else:
            _init = None

        self.q = builder.addHead(lambda: hk.Linear(self.actions, name='q', w_init=_init, b_init=_init, with_bias=self.bias_unit))
        self.h = builder.addHead(lambda: hk.Linear(self.actions, name='h', w_init=_init, b_init=_init), grad=False)

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

    # -------------
    # -- Updates --
    # -------------
    @partial(jax.jit, static_argnums=0)
    def _computeUpdate(self, state: AgentState, batch: Batch):
        grad_fn = jax.grad(self._loss, has_aux=True)
        grad, metrics = grad_fn(state.params, batch)

        updates, optim = self.optimizer.update(grad, state.optim, state.params)

        # Post-Adam decoupled weight decay on h-params only.
        # Applied after Adam transforms the gradient so beta is not absorbed
        # into Adam's moment estimates (unlike the L2 term in QRC._loss).
        decay = tree_map(
            lambda h, dh: dh - self.stepsize * self.beta * h,
            state.params['h'],
            updates['h'],
        )
        updates = dict(updates)
        updates['h'] = decay

        params = optax.apply_updates(state.params, updates)

        new_state = AgentState(
            params=params,
            optim=optim,
        )

        return new_state, metrics

    def _loss(self, params, batch: Batch):
        phi = self.phi(params, batch.x).out
        q = self.q(params, phi)
        h = self.h(params, phi)

        phi_p = self.phi(params, batch.xp).out
        qp = self.q(params, phi_p)

        q_loss, h_loss, metrics = qc_loss(q, batch.a, batch.r, batch.gamma, qp, h)

        v_loss = jnp.mean(q_loss)
        h_loss_mean = jnp.mean(h_loss)

        metrics |= {
            'v_loss': v_loss,
            'h_loss': h_loss_mean,
        }

        # No L2 term — regularization is handled post-Adam in _computeUpdate
        return v_loss + self.eta * h_loss_mean, metrics


# ---------------
# -- Utilities --
# ---------------
@partial(jax.vmap, in_axes=0)
def qc_loss(q, a, r, gamma, qp, h):
    vtp1 = qp.max()
    target = r + gamma * vtp1
    target = jax.lax.stop_gradient(target)

    delta = target - q[a]
    delta_hat = h[a]

    v_loss = 0.5 * delta ** 2 + gamma * jax.lax.stop_gradient(delta_hat) * vtp1
    h_loss = 0.5 * (jax.lax.stop_gradient(delta) - delta_hat) ** 2

    return v_loss, h_loss, {
        'delta': delta,
        'h': delta_hat,
    }
