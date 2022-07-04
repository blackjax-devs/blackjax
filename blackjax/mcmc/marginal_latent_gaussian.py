"""Public API for marginal latent Gaussian sampling."""
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve

from blackjax.types import Array, PRNGKey

__all__ = ["MarginalState", "MarginalInfo", "init_and_kernel"]


class MarginalState(NamedTuple):
    """State of the RMH chain.

    x
        Current position of the chain.
    log_p_x
        Current value of the log-likelihood of the model
    grad_x
        Current value of the gradient of the log-likelihood of the model

    Other Attributes:
    -----------------
    U_x, U_grad_x: Array
        Auxiliary attributes

    """

    x: Array
    log_p_x: float
    grad_x: Array

    U_x: Array
    U_grad_x: Array


class MarginalInfo(NamedTuple):
    """Additional information on the RMH chain.

    This additional information can be used for debugging or computing
    diagnostics.

    acceptance_probability
        The acceptance probability of the transition, linked to the energy
        difference between the original and the proposed states.
    is_accepted
        Whether the proposed position was accepted or the original position
        was returned.
    proposal
        The state proposed by the proposal.
    """

    acceptance_probability: float
    is_accepted: bool
    proposal: MarginalState


def init_and_kernel(log_pdf, C_or_C_inv, use_inverse=False, mean=None):
    """Build the marginal version of the auxiliary gradient-based sampler

    Returns
    -------
    A kernel that takes a rng_key and a Pytree that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.
    An init function.

    """

    U, Gamma, U_t = jnp.linalg.svd(C_or_C_inv, hermitian=True)
    if use_inverse:
        Gamma = 1.0 / Gamma

    if mean is not None:
        shift = (
            C_or_C_inv @ mean if use_inverse else solve(C_or_C_inv, mean, sym_pos=True)
        )
        val_and_grad = jax.value_and_grad(lambda x: log_pdf(x) + jnp.dot(x, shift))
    else:
        val_and_grad = jax.value_and_grad(log_pdf)

    def step(key: PRNGKey, state: MarginalState, delta):
        # Some bookkeeping
        y_key, u_key = jax.random.split(key, 2)
        h_delta = 0.5 * delta  # half delta

        x, log_p_x, grad_x, U_x, U_grad_x = state

        # Update Gamma(delta)
        # TODO: Ideally, we could have a dichotomy, where we only update Gamma(delta) if delta changes,
        #       but this is hardly the most expensive part of the algorithm (the multiplication by U below is).
        Gamma_1 = Gamma * delta / (delta + 2 * Gamma)
        Gamma_3 = (delta + 2 * Gamma) / (delta + 4 * Gamma)
        Gamma_2 = Gamma_1 / Gamma_3

        # Propose a new y
        temp = Gamma_1 * (U_x / h_delta + U_grad_x)
        temp = temp + jnp.sqrt(Gamma_2) * jax.random.normal(y_key, x.shape)
        y = U @ temp

        # Bookkeeping
        log_p_y, grad_y = val_and_grad(y)
        U_y = U_t @ y
        U_grad_y = U_t @ grad_y

        # Acceptance step
        temp_x = Gamma_1 * (U_x / h_delta + 0.5 * U_grad_x)
        temp_y = Gamma_1 * (U_y / h_delta + 0.5 * U_grad_y)

        hxy = jnp.dot(U_x - temp_y, Gamma_3 * U_grad_y)
        hyx = jnp.dot(U_y - temp_x, Gamma_3 * U_grad_x)

        alpha = jnp.minimum(1, jnp.exp(log_p_y - log_p_x + hxy - hyx))
        accept = jax.random.uniform(u_key) < alpha

        proposed_state = MarginalState(y, log_p_y, grad_y, U_y, U_grad_y)
        state = jax.lax.cond(accept, lambda _: proposed_state, lambda _: state, None)
        info = MarginalInfo(alpha, accept, proposed_state)
        return state, info

    def init(x):
        log_p_x, grad_x = val_and_grad(x)
        return MarginalState(x, log_p_x, grad_x, U_t @ x, U_t @ grad_x)

    return init, step
