"""Public API for the window adaptation.

Exposing a public API for the window adaptation is slightly trickier as there
are more moving pieces than in the kernel.
"""
from typing import Callable, List, NamedTuple, Tuple, Union

import jax
import jax.numpy as jnp

import blackjax.adaptation.mass_matrix as adapt_mass_matrix
import blackjax.adaptation.schedules as schedules
import blackjax.adaptation.step_size as adapt_step_size
import blackjax.hmc as hmc
import blackjax.inference.base as base
import blackjax.nuts as nuts


class WindowAdaptationState(NamedTuple):
    dual_averaging_state: adapt_step_size.DualAveragingState
    inv_covariance_state: adapt_mass_matrix.InverseCovarianceState


def window_adaptation(
    rng_key: jax.random.PRNGKey,
    kernel_factory: Callable[[Callable, Union[hmc.HMCParameters, nuts.NUTSParameters]], Callable],
    logpdf: Callable,
    init_state: base.HMCState,
    init_parameters: Union[hmc.HMCParameters, nuts.NUTSParameters],
    num_warmup_steps: int = 500,
    is_mass_matrix_diagonal: bool = True,
):
    """Warmup scheme for sampling procedures based on euclidean manifold HMC.
    The schedule and algorithms used match Stan's [1]_ as closely as possible.

    Unlike several other libraries, we separate the warmup and sampling phases
    explicitly. This ensure a better modularity; a change in the warmup does
    not affect the sampling. It also allows users to run their own warmup
    should they want to.

    Stan's warmup consists in the three following phases:

    1. A fast adaptation window where only the step size is adapted using
    Nesterov's dual averaging scheme to match a target acceptance rate.
    2. A succession of slow adapation windows (where the size of a window
    is double that of the previous window) where both the mass matrix and the step size
    are adapted. The mass matrix is recomputed at the end of each window; the step
    size is re-initialized to a "reasonable" value.
    3. A last fast adaptation window where only the step size is adapted.

    Schematically:

    ```
    +---------+---+------+------------+------------------------+------+
    |  fast   | s | slow |   slow     |        slow            | fast |
    +---------+---+------+------------+------------------------+------+
    1         2   3      3            3                        3
    ```

    Step (1) consists in find a "reasonable" first step size that is used to
    initialize the dual averaging scheme. In (2) we initialize the mass matrix
    to the matrix. In (3) we compute the mass matrix to use in the kernel and
    re-initialize the mass matrix adaptation. The step size is still adapated
    in slow adaptation windows, and is not re-initialized between windows.

    Parameters
    ----------
    rng_key
        Key for the pseudo-random number generator.
    kernel_factory:
        A function which returns a transition kernel given a logpdf and a HMCParameters or
        NUTSParameters tuple. It can return an HMC kernel, a NUTS kernel and it does not
        matter which kernel will be used later. You can use NUTS to adapt HMC's parameters.
    logpdf
        Fuction that returns the log-probability of a given chain position.
    init_state
        Initial state of the chain.
    init_parameters:
        Initial parameters for the kernel.
    num_warmup_steps:
        Number of warmup steps to run.
    is_mass_matrix_diagonal
        Create and adapt a diagonal mass matrix if True, a dense matrix otherwise.

    Returns
    -------
    init
        Function that initializes the warmup.
    update
        Function that moves the warmup one step.
    final
        Function that returns the step size and mass matrix given a warmup state.
    """
    potential = lambda x: -logpdf(x)
    kernel_from_params = jax.partial(kernel_factory, potential)

    dual_averaging_init, dual_averaging_update = adapt_step_size.dual_averagin()
    (
        inv_covariance_init,  # wow that's black gone wild here
        inv_covariance_update,
        inv_covariance_final,
    ) = adapt_mass_matrix.inv_covariance(is_mass_matrix_diagonal)

    def adapt_init(
        rng_key: jax.random.PRNGKey,
        initial_state: base.HMCState,
        initial_parameters,
    ):
        mm_state = inv_covariance_init(initial_state)

        step_size = adapt_step_size.find_reasonable_step_size(
            rng_key,
            kernel_factory,
            initial_state,
            init_parameters,
        )
        da_state = dual_averaging_init(step_size)

        warmup_state = WindowAdaptationState(da_state, mm_state)

        return warmup_state

    def adapt_update(
        rng_key: jax.random.PRNGKey,
        chain_state: base.HMCState,
        warmup_state: WindowAdaptationState,
        warmup_stage: Tuple[int, bool],
    ) -> Tuple[base.HMCState, WindowAdaptationState, base.HMCInfo]:
        """Move the warmup by one step.
        We first create a new kernel with the current values of the step size
        and mass matrix and move the chain one step. Then, depending on the
        stage passed as an argument we execute either the fast or slow interval
        update. Finally we execute the final update of the slow interval depending
        on whether we are at the end of the window.
        Parameters
        ----------
        rng_key
            The key used in JAX's random number generator.
        stage
            The current stage of the warmup. 0 for the fast interval, 1 for the
            slow interval.
        is_middle_window_end
            True if this step is the last of a slow adaptation interval.
        chain_state
            Current state of the chain.
        warmup
            Current warmup state.
        Returns
        -------
        The updated states of the chain and the warmup.
        """
        stage, is_middle_window_end = warmup_stage
        dual_averaging_state, inv_covariance_state = warmup_state

        step_size = jnp.exp(dual_averaging_state.log_step_size)
        inv_mass_matrix = inv_covariance_state.inverse_mass_matrix
        kernel = kernel_from_params(
            init_parameters._replace(
                step_size=step_size, inv_mass_matrix=inv_mass_matrix
            )
        )

        chain_state, chain_info = kernel(rng_key, chain_state)

        warmup_state = jax.lax.switch(
            stage,
            (dual_averaging_update, inv_covariance_update),
            (rng_key, chain_state, chain_info, warmup_state),
        )

        warmup_state = jax.lax.cond(
            is_middle_window_end,
            inv_covariance_final,  # this is not exactly true, but a small detail here
            lambda x: x,
            warmup_state,
        )

        return chain_state, warmup_state, chain_info

    def adapt_final(
        warmup_state: WindowAdaptationState,
    ) -> Union[hmc.HMCParameters, nuts.NUTSParameters]:
        """Return the initial parameters with updated mass matrix and step size."""
        dual_averaging_state, inv_covariance_state = warmup_state
        step_size = jnp.exp(dual_averaging_state.log_step_size_avg)
        inv_mass_matrix = inv_covariance_state.inv_mass_matrix
        return init_parameters._replace(
            step_size=step_size, inv_mass_matrix=inv_mass_matrix
        )

    # Separating the code this way makes the flow more obvious.

    schedule = jnp.array(schedules.stan_window_adaptation)
    state, warmup_state = adapt_init(rng_key, init_state, init_parameters)

    def one_step(carry, warmup_stage):
        rng_key, state, warmup_state = carry
        _, rng_key = jax.random.split(rng_key)
        state, warmup_state, info = adapt_update(
            rng_key, state, warmup_state, warmup_stage
        )
        return (rng_key, state, warmup_state), (state, warmup_state, info)

    last_carry, warmup_chain = jax.lax.scan(
        one_step, (rng_key, state, warmup_state), schedule
    )
    _, last_state, last_warmup_state = last_carry

    parameters = adapt_final(warmup_state)

    return last_state, parameters, warmup_chain
