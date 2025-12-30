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


from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from blackjax.base import SamplingAlgorithm
from blackjax.smc.base import update_and_take_last
from blackjax.smc.from_mcmc import unshared_parameters_and_step_fn
from blackjax.types import Array, ArrayLikeTree, ArrayTree, PRNGKey

__all__ = [
    "PersistentSMCState",
    "PersistentStateInfo",
    "init",
    "remove_padding",
    "compute_log_Z",
    "compute_log_persistent_weights",
    "resample_from_persistent",
    "compute_persistent_ess",
    "step",
    "build_kernel",
    "as_top_level_api",
]


class PersistentSMCState(NamedTuple):
    """State of the Persistent Sampling algorithm.

    Contains all particles from all iterations, their weights,
    log-likelihoods, log normalizing constants, tempering parameters and an
    index for the current iteration.
    Particles of the current iteration can be accessed via the `particles` property
    for convenience.

    NOTE: All arrays should be padded with zeros up the length of the
    tempering schedule + 1. This is to allow JIT compilation.

    Parameters
    ----------
    persistent_particles: ArrayLikeTree
        Particles from all iterations (padded with zeros to expected length of
        tempering schedule + 1).
    persistent_log_likelihoods: Array
        Log-likelihoods for all persistent particles, updated for current iteration.
        Shape is (n_schedule + 1, n_particles).
    persistent_log_Z: Array
        History of (log of) normalizing constants :math:`[log(Z_0), \\ldots, log(Z_t)]`,
        zero-padded for all iterations.
    tempering_schedule: Array
        History of tempering parameters :math:`[\\lambda_0, \\ldots, \\lambda_t]`,
        zero-padded.
    iteration: Array
        Current iteration index.

    Derived Properties
    ------------------
    particles: ArrayLikeTree
        Particles in current iteration (i.e. at index `iteration`).
    tempering_param: float | Array
        Tempering parameter in current iteration.
    log_Z: float | Array
        Log normalizing constant in current iteration.
    persistent_weights: Array
        Normalized weights for all persistent particles, updated for current iteration.
        Shape is (n_schedule + 1, n_particles), where n_schedule is the number of
        tempering steps. Normalized such that they sum to iteration * n_particles.
        Calculated using persistent_log_likelihoods, persistent_log_Z,
        tempering_schedule, and iteration.
        NOTE: The weights are calculated on-the-fly, rather than than stored during
        the sampling process, since the weights in the current iteration depend on
        the particles sampled at that iteration, while in the algorithm the weights are
        calculated before sampling the new particles.
    num_particles: int
        Number of particles.
    """

    persistent_particles: ArrayLikeTree
    persistent_log_likelihoods: Array
    persistent_log_Z: Array
    tempering_schedule: Array
    iteration: int | Array

    @property
    def particles(self) -> ArrayLikeTree:
        """Particles in current iteration."""
        return jax.tree.map(lambda x: x[self.iteration], self.persistent_particles)

    @property
    def tempering_param(self) -> float | Array:
        """Tempering parameter in current iteration."""
        return self.tempering_schedule[self.iteration]

    @property
    def log_Z(self) -> float | Array:
        """Log normalizing constant in current iteration."""
        return self.persistent_log_Z[self.iteration]

    @property
    def persistent_weights(self) -> Array:
        """Weights for all persistent particles in current iteration, normalized
        to sum to iteration * n_particles."""
        return jnp.exp(
            compute_log_persistent_weights(
                self.persistent_log_likelihoods,
                self.persistent_log_Z,
                self.tempering_schedule,
                self.iteration,
                include_current=True,
            )[0]
        )

    @property
    def num_particles(self) -> int:
        """Number of particles."""
        return self.persistent_log_likelihoods.shape[1]


class PersistentStateInfo(NamedTuple):
    """Information from one step of Persistent Sampling.

    Parameters
    ----------
    ancestors: Array
        The index of the particles selected by the resampling step.
    update_info: NamedTuple
        Additional information returned by the update function.
    """

    ancestors: Array
    update_info: NamedTuple


def init(
    particles: ArrayLikeTree,
    loglikelihood_fn: Callable,
    n_schedule: int | Array,
) -> PersistentSMCState:
    """Initialize the Persistent Sampling state.

    The arrays are padded with zeros to alow for JIT compilation.
    The dimension of the arrays is (n_schedule + 1, n_particles),
    where n_schedule is the number of tempering steps. The + 1 is to
    account for the initial prior distribution at iteration 0.

    Parameters
    ----------
    particles: PyTree
        Initial N particles (typically sampled from prior).
    loglikelihood_fn: Callable
        Log likelihood function.
    n_schedule: int | Array
        Number of steps in the tempering schedule.

    Returns
    -------
    PersistentSMCState
        Initial state, with
        - particles set to input particles,
        - weights set to uniform weights,
        - log-likelihoods set to the log-likelihoods of the input particles,
        - normalizing constant set to 1.0 (assume prior is normalized, this is
          important),
        - tempering parameters set to 0.0 (initial distribution is prior).
        - set iteration to 0.

        NOTE: All arrays in the PersistentSMCState are padded with zeros up
        to the length of the tempering schedule.
    """

    # Infer the number of particles from the size of the leading dimension of
    # the first leaf of the inputted PyTree.
    num_particles = jax.tree.flatten(particles)[0][0].shape[0]

    # Allocate arrays to store persistent particles and log-likelihoods, and
    # fill in the first entry with the initial values.
    padded_log_likelihoods = (
        jnp.zeros((n_schedule + 1, num_particles))
        .at[0]
        .set(jax.vmap(loglikelihood_fn)(particles))
    )
    padded_particles = jax.tree.map(
        lambda x: jnp.zeros((n_schedule + 1, *x.shape)).at[0].set(x), particles
    )

    return PersistentSMCState(
        padded_particles,
        padded_log_likelihoods,
        jnp.zeros(n_schedule + 1),  # log(1.0) = 0.0, so already set correctly
        jnp.zeros(n_schedule + 1),  # lambda_0 = 0.0, so already set correctly
        jnp.array(0),
    )


def remove_padding(state: PersistentSMCState) -> PersistentSMCState:
    """Remove padding from PersistentSMCState arrays up to current iteration.

    Parameters
    ----------
    state: PersistentSMCState
        The PersistentSMCState with padded arrays.

    Returns
    -------
    PersistentSMCState
        New PersistentSMCState with arrays trimmed to current iteration.
    """
    iteration = state.iteration
    return PersistentSMCState(
        persistent_particles=jax.tree.map(
            lambda x: x[: iteration + 1], state.persistent_particles
        ),
        persistent_log_likelihoods=state.persistent_log_likelihoods[: iteration + 1],
        persistent_log_Z=state.persistent_log_Z[: iteration + 1],
        tempering_schedule=state.tempering_schedule[: iteration + 1],
        iteration=state.iteration,
    )


def compute_log_Z(
    log_weights: Array,
    iteration: int | Array,
) -> Array:
    """Compute log normalizing constant from log weights.

    Implements Equation 16 from the Karamanis2025.

    Parameters
    ----------
    log_weights: Array
        Log of unnormalized weights for all persistent particles at current iteration.
    iteration: int | Array
        Current iteration index.

    Returns
    -------
    log_Z: float | Array
        Estimate of log of normalizing constant :math:`\\hat{Z}_{t}` at current
        iteration.

    """

    num_particles = log_weights.shape[1]
    log_normalization_constant = (
        logsumexp(log_weights) - jnp.log(num_particles) - jnp.log(iteration)
    )
    return log_normalization_constant


def compute_log_persistent_weights(
    persistent_log_likelihoods: Array,
    persistent_log_Z: Array,
    tempering_schedule: Array,
    iteration: int | Array,
    include_current: bool = False,
    normalize_to_one: bool = False,
) -> tuple[Array, Array]:
    """Compute importance weights for all persistent particles for
    current iteration.

    Implements Equations 14 and 15 from the Karamanis2025.

    NOTE: The returned weights are normalized such that they sum to
    :math:`(i \\times N)`, where i is the current iteration and N
    is the number of particles. They need to be renormalized to sum to 1.0
    before resampling, this can be done using the 'normalize_to_one' argument.

    Parameters
    ----------
    persistent_log_likelihoods: Array
        Log-likelihoods for all persistent particles (for all previous
        current iteration).
    persistent_log_Z: Array
        Log normalizing constants for all previous iterations.
    tempering_schedule: Array
        Tempering parameters up to current iteration.
    iteration: int | Array
        Current iteration index.
    include_current: bool, optional
        If `True`, include the current iteration in the weight computation (i.e.
        sum to t rather than t-1 in equations 14-16). This is useful when calculating
        the weights after the resampling step, where the current iteration's particles
        are already included in the persistent ensemble.
    normalize_to_one: bool, optional
        If `True`, normalize the weights to sum to 1.0. By default, the weights
        sum to (iteration * n_particles), as described in the paper.

    Returns
    -------
    normalized_log_weights: Array
        Log of normalized weights :math:`W^i_{tt'}` for all :math:`i \\times N`
        persistent particles at current iteration.
    new_log_Z: float
        Estimate of log of normalizing constant :math:`\\hat{Z}_{t}` at current
        iteration.

    """

    # create mask for all iterations up to current
    current_iteration = iteration + 1 if include_current else iteration
    mask = jnp.arange(persistent_log_likelihoods.shape[0])[:, None] < current_iteration

    # numerate in equation 15, masking out all iterations beyond current and
    # filling with nans
    log_numerator = jnp.where(
        mask,
        persistent_log_likelihoods * tempering_schedule[iteration],
        jnp.nan,
    )

    # denominator in equation 15
    # NOTE: This could be done using broadcasting rather than a loop, but the loop
    # seems more memory efficient. Otherwise, we would need to create a large
    # (n_schedule + 1, n_schedule + 1, n_particles) array.
    log_denominator = jax.lax.fori_loop(
        0,
        current_iteration,
        lambda i, val: jnp.where(
            mask,
            jnp.logaddexp(
                val,
                tempering_schedule[i] * persistent_log_likelihoods
                - persistent_log_Z[i],
            ),
            jnp.nan,
        ),
        jnp.full_like(persistent_log_likelihoods, -jnp.inf),  # initial = 0 in log-space
    ) + jnp.log(1.0 / current_iteration)

    # unnormalized weights, equation 15, fill nans with zeros (-inf in log-space)
    unnormalized_log_weights = jnp.nan_to_num(
        log_numerator - log_denominator,
        nan=-jnp.inf,
    )

    # normalization constant, equation 16
    log_normalization_constant = compute_log_Z(
        unnormalized_log_weights,
        current_iteration,
    )

    # normalized weights, equation 14
    log_weights = unnormalized_log_weights - log_normalization_constant
    # if requested, normalize weights to sum to 1.0 (instead of
    #  current_iteration * n_particles)
    if normalize_to_one:
        num_particles = log_weights.shape[1]
        log_weights = log_weights - jnp.log(current_iteration * num_particles)

    return log_weights, log_normalization_constant


def resample_from_persistent(
    rng_key: PRNGKey,
    persistent_particles: ArrayLikeTree,
    persistent_weights: Array,
    resample_fn: Callable,
) -> tuple[ArrayTree, Array]:
    """Resample N particles from the :math:`i \\times N`
    persistent ensemble, where i is the current iteration.

    Parameters
    ----------
    rng_key: PRNGKey
        JAX random key.
    persistent_particles: ArrayLikeTree
        Historical particles of the i previous iterations.
    persistent_weights: Array
        Normalized weights for all :math:`i \\times N` particles.
        NOTE: The weights need to sum to 1, this is different from
        the 'normalized' described by equation 14 in Karamanis2025
        amd computed by _compute_log_persistent_weights. These sum
        to :math:`(i \\times N)`, i.e. the current iteration times
        the number of particles (the current number of persistent
        particles in the current iteration).
    resample_fn: Callable
        Resampling function (from blackjax.smc.resampling)

    Returns
    -------
    resampled_particles: ArrayTree
        N particles resampled from persistent ensemble.
    resample_idx: Array
        Indices of the selected particles.
    """

    resample_idx = resample_fn(
        rng_key,
        persistent_weights.flatten(),
        num_samples=persistent_weights.shape[1],
    )

    # map index to particles, reshaping to (i * N, ...) first,
    # to match the flattened weights shape
    particles = jax.tree.map(
        lambda x: x.reshape(-1, *x.shape[2:])[resample_idx],
        persistent_particles,
    )
    return particles, resample_idx


def compute_persistent_ess(
    log_persistent_weights: Array,
    normalize_weights: bool = False,
) -> float | Array:
    """Calculate the effective sample size (ESS) of the persistent
    ensemble. Equation 17 from Karamanis2025.

    NOTE: For the second identity in equation 17 to hold, the
    weights must be normalized to sum to 1.0. This function normalizes
    the weights internally if `normalize_weights` is set to `True`.

    NOTE: The ESS can be > 1 for Persistent Sampling, unlike standard
    SMC.

    Parameters
    ----------
    log_persistent_weights: Array
        Normalized log weights for all persistent particles.
    normalize_weights: bool, optional
        If `True`, normalize the weights to sum to 1.0 before computing
        the ESS. By default, the weights are assumed to be normalized.

    Returns
    -------
    ess: float | Array
        Effective sample size of the persistent ensemble.
    """
    if normalize_weights:
        log_persistent_weights = log_persistent_weights - logsumexp(
            log_persistent_weights
        )

    return 1.0 / jnp.sum(jnp.exp(log_persistent_weights) ** 2)


def step(
    rng_key: PRNGKey,
    state: PersistentSMCState,
    lmbda: float | Array,
    loglikelihood_fn: Callable,
    update_fn: Callable,
    resample_fn: Callable,
    weight_fn: Callable = compute_log_persistent_weights,
) -> tuple[PersistentSMCState, PersistentStateInfo]:
    """One step of the Persistent Sampling algorithm, as
    described in algorithm 2 of Karamanis et al. (2025).

    Parameters
    ----------
    rng_key
        Key used for random number generation.
    state
        Current state of the PS sampler described by a PersistentSMCState.
    lmbda: float | Array
        New tempering parameter :math:`\\lambda_t` for current iteration.
    loglikelihood_fn: Callable
        Log likelihood function.
    update_fn: Callable
        MCMC kernel that takes in an array of keys and particles and returns
        updated particles along with any extra information.
    resample_fn: Callable
        Resampling function (from blackjax.smc.resampling). This function
        is passed to _resample_from_persistent to resample from the
        persistent ensemble.
    weight_fn
        Function that assigns a weight to the particles, by default
        _compute_log_persistent_weights, which implements equation 14-16 from
        Karamanis2025. Should return normalized log weights and log normalizing
        constant.

    Returns
    -------
    new_state: PersistentSMCState
        The updated PersistentSMCState. Updated fields are:
        - particles: particles from all iterations, with current iteration's
          particles added.
        - weights: normalized weights for all persistent particles at current
          iteration.
        - log_likelihoods: log-likelihoods for all persistent particles,
          with current iteration's log-likelihoods added.
        - log_Z: log normalizing constants, with current iteration's
          normalizing constant added.
        - tempering_schedule: tempering parameters, with current iteration's
          parameter added.
        - iteration: incremented by 1.
    info: PersistentStateInfo
        An `PersistentStateInfo` object that contains extra information about the PS
        transition. Contains:
        - ancestors: indices of the particles selected by the resampling step.
        - ess: effective sample size of the persistent ensemble.
        - update_info: any extra information returned by the update function.

    """

    # update iteration and split keys
    iteration = state.iteration + 1
    updating_key, resampling_key = jax.random.split(rng_key, 2)

    # update tempering schedule with new lambda
    tempering_schedule = state.tempering_schedule.at[iteration].set(lmbda)

    # compute weights
    log_weights, log_Z = weight_fn(
        state.persistent_log_likelihoods,
        state.persistent_log_Z,
        tempering_schedule,
        iteration,
        normalize_to_one=True,
    )
    weights = jnp.exp(log_weights)

    # resample particles
    num_particles = state.persistent_weights.shape[1]

    iteration_particles, resample_idx = resample_from_persistent(
        resampling_key,
        state.persistent_particles,
        weights,
        resample_fn,
    )

    # update particles with MCMC kernel
    keys = jax.random.split(updating_key, num_particles)
    iteration_particles, update_info = update_fn(
        keys,
        iteration_particles,
    )

    # calculate log likelihoods for new particles
    iteration_log_likelihoods = jax.vmap(loglikelihood_fn)(iteration_particles)

    # update state
    persistent_particles = jax.tree.map(
        lambda persistent, iteration_p: persistent.at[iteration].set(iteration_p),
        state.persistent_particles,
        iteration_particles,
    )

    persistent_log_Z = state.persistent_log_Z.at[iteration].set(log_Z)
    persistent_log_likelihoods = state.persistent_log_likelihoods.at[iteration].set(
        iteration_log_likelihoods
    )

    new_state = PersistentSMCState(
        persistent_particles=persistent_particles,
        persistent_log_likelihoods=persistent_log_likelihoods,
        persistent_log_Z=persistent_log_Z,
        tempering_schedule=tempering_schedule,
        iteration=iteration,
    )

    # calculate effective sample size
    return new_state, PersistentStateInfo(resample_idx, update_info)


def build_kernel(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    mcmc_step_fn: Callable,
    mcmc_init_fn: Callable,
    resampling_fn: Callable,
    update_strategy: Callable = update_and_take_last,
) -> Callable:
    """Build a Persistent Sampling kernel, with signature
    (rng_key,
    state,
    num_mcmc_steps,
    lmbda,
    mcmc_parameters,) -> (new_state, info).

    The function implements the Persistent Sampling algorithm as described
    in Karamanis et al. (2025), with a fixed tempering schedule. It
    functions similarly to tempered SMC (see blackjax.smc.tempered),
    but keeps track of all particles from all previous iterations. This
    can lead to a more stable posterior and marginal likelihood estimation
    at the cost of higher memory usage.

    Parameters
    ----------
    logprior_fn: Callable
        Log prior probability function.
        NOTE: This function must be normalized (:math:`Z_0 = 1`), in order
        for the weighting scheme to function correctly.
    loglikelihood_fn: Callable
        Log likelihood function.
    mcmc_step_fn: Callable
        Function that creates MCMC step from log-probability density function.
    mcmc_init_fn: Callable
        A function that creates a new mcmc state from a position and a
        log-probability density function.
    resampling_fn: Callable
        Resampling function (from blackjax.smc.resampling).
    update_strategy: Callable
        Strategy to update particles using MCMC kernels, by default
        'update_and_take_last' from blackjax.smc.base. The function signature must be
        (mcmc_init_fn,
        loggerposterior_fn,
        mcmc_step_fn,
        num_mcmc_steps,
        n_particles,) -> (mcmc_kernel, n_particles), like 'update_and_take_last'.
        The mcmc_kernel must have signature
        (rng_key, position, mcmc_parameters) -> (new_position, info).

    Returns
    -------
    kernel: Callable
        A callable that takes a rng_key, a PersistentSMCState, a tempering parameter
        lmbda, and a dictionary of mcmc_parameters, and that returns a the
        PersistentSMCState after the step along with information about the transition.
    """

    def update_fn(
        rng_key: PRNGKey,
        current_particles: ArrayLikeTree,
        num_mcmc_steps: int | Array,
        logposterior_fn: Callable,
        mcmc_parameters: dict,
        n_particles: int | Array,
    ) -> None:
        """Update function that runs MCMC kernels on the particles."""
        unshared_mcmc_parameters, shared_mcmc_step_fn = unshared_parameters_and_step_fn(
            mcmc_parameters, mcmc_step_fn
        )

        mcmc_kernel, _ = update_strategy(
            mcmc_init_fn,
            logposterior_fn,
            shared_mcmc_step_fn,
            num_mcmc_steps=num_mcmc_steps,
            n_particles=n_particles,
        )

        return mcmc_kernel(rng_key, current_particles, unshared_mcmc_parameters)

    def kernel(
        rng_key: PRNGKey,
        state: PersistentSMCState,
        num_mcmc_steps: int | Array,
        lmbda: float | Array,
        mcmc_parameters: dict,
    ) -> tuple[PersistentSMCState, PersistentStateInfo]:
        """Kernel to move the particles one step using the
        Persistent Sampling algorithm.

        Parameters
        ----------
        rng_key : PRNGKey
            Key used for random number generation.
        state : PersistentSMCState
            The sampling state from the previous iteration.
        num_mcmc_steps : int | Array
            Number of MCMC steps to apply to each particle.
        lmbda : float | Array
            Tempering parameter :math:`\\lambda_t` for current iteration.
        mcmc_parameters : dict
            The parameters for the mcmc kernel.

        Returns
        -------
        new_state : PersistentSMCState
            The new sampling state after one step of Persistent Sampling.
        info : PersistentStateInfo
            Additional information on the PS step.
        """

        def logposterior_fn(x: ArrayLikeTree) -> Array:
            """Tempered log-posterior function."""
            return logprior_fn(x) + lmbda * loglikelihood_fn(x)

        update_fn_wrapper = partial(
            update_fn,
            num_mcmc_steps=num_mcmc_steps,
            logposterior_fn=logposterior_fn,
            mcmc_parameters=mcmc_parameters,
            n_particles=state.persistent_weights.shape[1],
        )

        return step(
            rng_key,
            state,
            lmbda,
            loglikelihood_fn,
            update_fn_wrapper,
            resampling_fn,
        )

    return kernel


def as_top_level_api(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    n_schedule: int | Array,
    mcmc_step_fn: Callable,
    mcmc_init_fn: Callable,
    mcmc_parameters: dict,
    resampling_fn: Callable,
    num_mcmc_steps: int = 10,
    update_strategy: Callable = update_and_take_last,
) -> SamplingAlgorithm:
    """
    Implements the user interface for the Persistent Sampling
    kernel. See build_kernel for details.

    NOTE: For this algorithm, we need to keep track of all particles
    from all previous iterations. To do so in a JIT-compatible way,
    we need to know the number of tempering steps in advance, to
    preallocate arrays of the correct size. Therefore, the user must
    provide the number of steps in the tempering schedule via the
    `n_schedule` argument.
    Since all arrays are preallocated to (n_schedule + 1, n_particles),
    where the + 1 accounts for the initial value at iteration 0. The user
    must ensure that the tempering schedule used in the actual sampling
    matches n_schedule.
    A tempering schedule with many steps may lead to high memory usage.

    NOTE: The algorithm enforces the tempering schedule to start at 0.0,
    if the supplied schedule also starts at 0.0, the first step will be
    done twice.

    Parameters
    ----------
    logprior_fn : Callable
        The log-prior function of the model we wish to draw samples from.
        NOTE: This function must be normalized (:math:`Z_0 = 1`), in order
        for the weighting scheme to function correctly.
    loglikelihood_fn : Callable
        The log-likelihood function of the model we wish to draw samples from.
    n_schedule : int | Array
        Number of steps in the tempering schedule.
    mcmc_step_fn : Callable
        The MCMC step function used to update the particles.
    mcmc_init_fn : Callable
        The MCMC initialization function used to initialize the MCMC state
        from a position.
    mcmc_parameters : dict
        The parameters for the MCMC kernel.
    resampling_fn : Callable
        Resampling function (from blackjax.smc.resampling).
    num_mcmc_steps : int, optional
        Number of MCMC steps to apply to each particle at each iteration,
        by default 10.
    update_strategy : Callable, optional
        The strategy to update particles using MCMC kernels, by default
        'update_and_take_last' from blackjax.smc.base. See build_kernel for
        details.

    Returns
    -------
    SamplingAlgorithm
        A ``SamplingAlgorithm`` instance with init and step methods. See
        blackjax.base.SamplingAlgorithm for details.
        The init method has signature
        (position: ArrayLikeTree) -> PersistentSMCState
        The step method has signature
        (rng_key: PRNGKey, state: PersistentSMCState, lmbda: float | Array) ->
        (new_state: PersistentSMCState, info: PersistentStateInfo)
    """

    kernel = build_kernel(
        logprior_fn,
        loglikelihood_fn,
        mcmc_step_fn,
        mcmc_init_fn,
        resampling_fn,
        update_strategy,
    )

    def init_fn(position: ArrayLikeTree) -> PersistentSMCState:
        return init(position, loglikelihood_fn, n_schedule)

    def step_fn(
        rng_key: PRNGKey,
        state: PersistentSMCState,
        lmbda: float | Array,
    ) -> tuple[PersistentSMCState, PersistentStateInfo]:
        return kernel(
            rng_key,
            state,
            num_mcmc_steps,
            lmbda,
            mcmc_parameters,
        )

    return SamplingAlgorithm(init_fn, step_fn)  # type: ignore[arg-type]
