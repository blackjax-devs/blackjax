# Copyright 2020- The Blackjax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import blackjax.util

from typing import Callable, NamedTuple, Tuple

from blackjax.types import ArrayLikeTree, ArrayTree, PRNGKey
from blackjax import SamplingAlgorithm

import jax
import jax.numpy as jnp


class GibbsState(NamedTuple):
    position: ArrayTree
    logdensity: float
    logdensity_grad: ArrayTree
    noise_contraction: float
    noise_sigma: float
    count: int


def noiser(rng_key: PRNGKey, state: GibbsState):
    position, _, _, noise_contraction, noise_sigma, _ = state
    noise = blackjax.util.generate_gaussian_noise(rng_key, position, 0, noise_sigma)
    return jax.tree.map(lambda x, n: noise_contraction * x + n, position, noise)


def noiser_logpdf(state: GibbsState, sample_noised: ArrayTree, sample_clean: ArrayTree):
    mean = jax.tree.map(jnp.multiply, sample_clean, state.noise_contraction)
    return jax.scipy.stats.norm.logpdf(sample_noised, mean, state.noise_sigma**2).sum()


def init_denoising(
    rng_key: PRNGKey,
    noised_position: ArrayTree,
    state: GibbsState,
    logdensity_fn: Callable,
):
    position, _, _, noise_contraction, noise_sigma, count = state
    noise = blackjax.util.generate_gaussian_noise(
        rng_key, noised_position, 0, noise_sigma / noise_contraction
    )
    proposal_position = jax.tree.map(
        lambda x, n: x / noise_contraction + n, noised_position, noise
    )

    def gaussian_term(a, b, scale):
        gauss_terms = jax.tree.map(
            lambda x, y: -(jnp.linalg.norm(x - y) / (2 * scale**2)), a, b
        )

        return jax.tree_util.tree_reduce(jnp.add, gauss_terms)

    scaled_noised = jax.tree.map(lambda x: x / noise_contraction, noised_position)
    proposal_logratio = gaussian_term(
        position, scaled_noised, noise_sigma / noise_contraction
    ) - gaussian_term(
        proposal_position,
        scaled_noised,
        noise_sigma / noise_contraction,
    )
    scaled_proposal = jax.tree.map(lambda x: x * noise_contraction, proposal_position)
    scaled_position = jax.tree.map(lambda x: x * noise_contraction, position)
    noising_logratio = gaussian_term(
        scaled_proposal, noised_position, noise_sigma
    ) - gaussian_term(scaled_position, noised_position, noise_sigma)

    diff_logdensity = logdensity_fn(proposal_position) - logdensity_fn(position)

    log_accept = diff_logdensity + proposal_logratio + noising_logratio

    # accept proposal_position with probability prob_accept
    prob_accept = jnp.clip(jnp.exp(log_accept), 0.0, 1.0)

    do_accept = jax.random.bernoulli(rng_key, prob_accept)

    return jax.tree.map(
        lambda x, y: jnp.where(do_accept, x, y), proposal_position, position
    )


def denoise(
    rng_key, position: ArrayLikeTree, denoiser: SamplingAlgorithm, n_steps: int
):
    init_state = denoiser.init(position, rng_key)

    def body_fn(state, rng_key):
        new_state, info = denoiser.step(rng_key, state)
        return new_state, info

    keys = jax.random.split(rng_key, n_steps)
    state_denoised, info = jax.lax.scan(body_fn, init_state, keys)

    return state_denoised.position


def build_kernel():
    def kernel(
        rng_key: PRNGKey,
        state: GibbsState,
        logdensity_fn: Callable,
        n_steps: int,
        schedule: Callable[[int], Tuple[float, float]],
    ):
        _, _, _, noise_contraction, noise_sigma, count = state
        grad_fn = jax.value_and_grad(logdensity_fn)
        logdensity, logdensity_grad = grad_fn(state.position)

        key_noiser, key_init, key_denoiser = jax.random.split(rng_key, 3)
        noised_position = noiser(key_noiser, state)
        position = init_denoising(key_init, noised_position, state, logdensity_fn)

        def conditional_logprob(x):
            scaled_position = jax.tree.map(lambda x: x * noise_contraction, x)
            norm_tree = jax.tree.map(
                lambda x, y: jnp.sum((x - y) ** 2) / (2 * noise_sigma**2),
                scaled_position,
                noised_position,
            )
            return logdensity_fn(x) - jax.tree_util.tree_reduce(jnp.add, norm_tree)

        denoised = denoise(
            key_denoiser, position, blackjax.mala(conditional_logprob, 1e-2), n_steps
        )
        count = count + 1
        noise_contraction, noise_sigma = schedule(count)
        state = GibbsState(
            denoised, logdensity, logdensity_grad, noise_contraction, noise_sigma, count
        )
        return state

    return kernel


def as_top_level_api(
    logdensity_fn: Callable,
    n_steps: int = 10,
    schedule: Callable[[int], Tuple[float, float]] = lambda _: (0.9, 0.1),
) -> SamplingAlgorithm:
    kernel = build_kernel()

    def init(position: ArrayLikeTree, rng_key=None) -> GibbsState:
        del rng_key
        grad_fn = jax.value_and_grad(logdensity_fn)
        logdensity, logdensity_grad = grad_fn(position)

        contraction, noise_sigma = schedule(0)
        return GibbsState(
            position, logdensity, logdensity_grad, contraction, noise_sigma, 0
        )

    def step(rng_key: PRNGKey, state: GibbsState) -> tuple[GibbsState, None]:
        return kernel(rng_key, state, logdensity_fn, n_steps, schedule)

    return SamplingAlgorithm(init, step)
