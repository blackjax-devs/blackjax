"""Public API for the Involutive MCMC Kernel"""
from typing import Callable, Dict, NamedTuple, Sequence, Tuple

import jax

import blackjax.mcmc.proposal as proposal
from blackjax.mcmc.integrators import IntegratorState
from blackjax.types import PRNGKey, PyTree

__all__ = ["IMCMCState", "IMCMCInfo", "init", "kernel"]


class IMCMCState(NamedTuple):

    position: PyTree
    potential_energy: float
    potential_energy_grad: PyTree

    momentum: PyTree = None
    d: PyTree = None
    a: PyTree = None
    theta: PyTree = None

    def get_energy(self, aux_potential_fn):
        return self.potential_energy + aux_potential_fn(self)

    def to_integrator(self):
        return IntegratorState(
            self.position,
            self.momentum,
            self.potential_energy,
            self.potential_energy_grad,
        )

    def from_integrator(self, int_state):
        return self._replace(**int_state._asdict())

    def flip_momentum(self):
        if self.d is None:
            return self._replace(
                momentum=jax.tree_util.tree_map(lambda x: -1.0 * x, self.momentum)
            )
        return self._replace(
            momentum=jax.lax.cond(
                self.d > 0,
                lambda _: self.momentum,
                lambda _: jax.tree_util.tree_map(lambda x: -1.0 * x, self.momentum),
                operand=None,
            )
        )


class IMCMCInfo(NamedTuple):
    do_accept: bool
    p_accept: float


class Proposal(NamedTuple):
    state: IMCMCState
    weight: float


def init(
    position: PyTree,
    potential_fn: Callable,
    auxiliary: Dict,
):

    potential_energy, potential_energy_grad = jax.value_and_grad(potential_fn)(position)
    return IMCMCState(position, potential_energy, potential_energy_grad, **auxiliary)


def kernel(
    aux_generator: Callable,
    aux_potential_fn: Callable,
    involutions: Sequence[Callable],
):
    def one_step(
        rng_key: PRNGKey,
        state: IMCMCState,
    ):

        proposal_generator = imcmc_proposal(involutions, aux_potential_fn)

        key_aux, key_proposal = jax.random.split(rng_key, 2)

        auxiliary = aux_generator(key_aux, state)
        proposal, infos = proposal_generator(key_proposal, state._replace(**auxiliary))

        return proposal, infos

    return one_step


def imcmc_proposal(
    involutions: Sequence[Callable],
    aux_potential_fn: Callable,
):

    sample_proposal = proposal.static_binomial_sampling

    def generate(rng_key, state: IMCMCState) -> Tuple[IMCMCState, IMCMCInfo]:

        # n_inv = len(involutions)
        # rng_keys = jax.random.split(rng_key, n_inv)

        # def propose_sample(state, i_rng):
        #     i, rng_key = i_rng
        infos = []
        for involution in involutions:
            energy = state.get_energy(aux_potential_fn)
            new_state, log_det_jac = involution(state)
            # new_state, log_det_jac = jax.lax.switch(i, involutions, state)
            new_energy = new_state.get_energy(aux_potential_fn)
            proposal = Proposal(state, 0.0)
            new_proposal = Proposal(new_state, energy - new_energy - log_det_jac)

            sampled_proposal, *info = sample_proposal(rng_key, proposal, new_proposal)
            do_accept, p_accept = info

            info = IMCMCInfo(do_accept, p_accept)

            infos.append(info)
            state = sampled_proposal.state

        return state, infos

        #     return sampled_proposal.state, info

        # state, infos = jax.lax.scan(propose_sample, state, (jax.numpy.arange(n_inv), rng_keys))
        # return state, infos

    return generate
