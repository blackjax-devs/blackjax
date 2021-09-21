"""Implementation of the Stan warmup for the HMC family of sampling algorithms."""
from typing import Any, Callable, List, NamedTuple, Tuple

import jax
import jax.numpy as jnp

from blackjax.adaptation.mass_matrix import (
    MassMatrixAdaptationState,
    mass_matrix_adaptation,
)
from blackjax.adaptation.step_size import (
    DualAveragingAdaptationState,
    dual_averaging_adaptation,
    find_reasonable_step_size,
)
from blackjax.inference.hmc.base import HMCState
from blackjax.types import Array, PRNGKey

__all__ = ["run", "stan_warmup"]


class StanWarmupState(NamedTuple):
    da_state: DualAveragingAdaptationState
    mm_state: MassMatrixAdaptationState


def run(
    rng_key,
    kernel_factory: Callable,
    initial_state: HMCState,
    num_steps: int = 1000,
    *,
    is_mass_matrix_diagonal: bool = True,
    initial_step_size: float = 1.0,
    target_acceptance_rate: float = 0.65,
) -> Tuple[HMCState, Tuple[float, Array], NamedTuple]:
    """Loop for the Stan warmup.

    Parameters
    ----------
    rng_key:
        Key for the pseudo-random number generator.
    kernel_factory:
        A function that takes a step size, and inverse mass matrix and returns
        a transition kernel.
    initial_state:
        The state from which to start the adaptation.
    num_steps:
        The number of times the kernel is run during the warmup.
    is_mass_matrix_diagonal:
        Indicates whether we should adapt a diagonal or full mass matrix.
    initial_step_size:
        The fist step size to use.
    target_acceptance_rate:
        Target acceptance rate for step size adaptation.

    Returns
    -------
        A tuple that contains the last state, a tuple with the step size and mass matrix, and a tuple
        that contains the whole adaptation chain.

    """
    init, update, final = stan_warmup(
        kernel_factory,
        is_mass_matrix_diagonal,
        target_acceptance_rate=target_acceptance_rate,
    )

    def one_step(carry, interval):
        rng_key, state, warmup_state = carry
        stage, is_middle_window_end = interval

        _, rng_key = jax.random.split(rng_key)
        state, warmup_state, info = update(
            rng_key, stage, is_middle_window_end, state, warmup_state
        )

        return ((rng_key, state, warmup_state), (state, warmup_state, info))

    schedule = jnp.array(stan_warmup_schedule(num_steps))

    warmup_state = init(rng_key, initial_state, initial_step_size)
    last_state, warmup_chain = jax.lax.scan(
        one_step, (rng_key, initial_state, warmup_state), schedule
    )
    _, last_chain_state, last_warmup_state = last_state

    step_size, inverse_mass_matrix = final(last_warmup_state)

    return last_chain_state, (step_size, inverse_mass_matrix), warmup_chain


def stan_warmup(
    kernel_factory: Callable,
    is_mass_matrix_diagonal: bool,
    target_acceptance_rate: float = 0.65,
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
    kernel_factory
        A function which returns a transition kernel given a step size and a
        mass matrix.
    is_mass_matrix_diagonal
        Create and adapt a diagonal mass matrix if True, a dense matrix otherwise.
    target_acceptance_rate:
        Target acceptance rate for step size adaptation.

    Returns
    -------
    init
        Function that initializes the warmup.
    update
        Function that moves the warmup one step.
    final
        Function that returns the step size and mass matrix given a warmup state.

    """
    fast_init, fast_update = fast_window(target_acceptance_rate)
    slow_init, slow_update, slow_final = slow_window(is_mass_matrix_diagonal)

    def init(
        rng_key: PRNGKey, initial_state: HMCState, initial_step_size: float
    ) -> StanWarmupState:
        """Initialize the warmup.

        To initialize the Stan warmup we create an identity mass matrix and use
        the `find_reasonable_step_size` procedure to initialize the dual
        averaging algorithm.

        """
        mm_state = slow_init(initial_state)

        kernel = lambda step_size: kernel_factory(
            step_size, mm_state.inverse_mass_matrix
        )
        step_size = find_reasonable_step_size(
            rng_key,
            kernel,
            initial_state,
            initial_step_size,
            target_acceptance_rate,
        )
        da_state = fast_init(step_size)

        warmup_state = StanWarmupState(da_state, mm_state)

        return warmup_state

    def update(
        rng_key: PRNGKey,
        stage: int,
        is_middle_window_end: bool,
        chain_state: HMCState,
        warmup_state: StanWarmupState,
    ) -> Tuple[HMCState, StanWarmupState, NamedTuple]:
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
        step_size = jnp.exp(warmup_state.da_state.log_step_size)
        inverse_mass_matrix = warmup_state.mm_state.inverse_mass_matrix
        kernel = kernel_factory(step_size, inverse_mass_matrix)

        chain_state, chain_info = kernel(rng_key, chain_state)

        warmup_state = jax.lax.switch(
            stage,
            (fast_update, slow_update),
            (rng_key, chain_state, chain_info, warmup_state),
        )

        warmup_state = jax.lax.cond(
            is_middle_window_end,
            slow_final,
            lambda x: x,
            warmup_state,
        )

        return chain_state, warmup_state, chain_info

    def final(warmup_state: StanWarmupState) -> Tuple[float, Array]:
        """Return the step size and mass matrix."""
        step_size = jnp.exp(warmup_state.da_state.log_step_size_avg)
        inverse_mass_matrix = warmup_state.mm_state.inverse_mass_matrix
        return step_size, inverse_mass_matrix

    return init, update, final


def fast_window(
    target_accept: float = 0.65,
) -> Tuple[Callable, Callable]:
    """First stage of the Stan warmup. The step size is adapted using
    Nesterov's dual averaging algorithms while the mass matrix stays the same.

    Parameters
    ----------
    target_accept:
        Target acceptance rate for step size adaptation using dual averaging.

    Returns
    -------
    A tuple of functions that respectively initialize the warmup state at the
    beginning of the window, and update the chain and warmup states within the
    window.

    """
    da_init, da_update, _ = dual_averaging_adaptation(target=target_accept)

    def init(initial_step_size: float) -> DualAveragingAdaptationState:
        da_state = da_init(initial_step_size)
        return da_state

    def update(
        fw_state: Tuple[Array, HMCState, Any, StanWarmupState]
    ) -> StanWarmupState:
        rng_key, state, info, warmup_state = fw_state

        new_da_state = da_update(warmup_state.da_state, info)
        new_warmup_state = StanWarmupState(new_da_state, warmup_state.mm_state)

        return new_warmup_state

    return init, update


def slow_window(
    is_mass_matrix_diagonal: bool = True,
    target_accept: float = 0.65,
) -> Tuple[Callable, Callable, Callable]:
    """Slow stage of the Stan warmup.

    In this stage we adapt the values of the mass matrix. The step size and the
    state of the mass matrix adaptation are re-initialized at the end of each
    window.

    Parameters
    ----------
    is_mass_matrix_diagonal
        Whether we want a diagonal mass matrix. Passed to the mass matrix adapation
        algorithm.
    target_accept:
        Target acceptance rate for step size adaptation using dual averaging.

    Returns
    -------
    A tuple of functions that respectively initialize the warmup state at the
    beginning of the window, update the chain and warmup states within the
    window, and update the warmup stage at the end of the window.

    """
    mm_init, mm_update, mm_final = mass_matrix_adaptation(is_mass_matrix_diagonal)
    da_init, da_update, da_final = dual_averaging_adaptation(target=target_accept)

    def init(state: HMCState) -> MassMatrixAdaptationState:
        """Initialize the mass matrix adaptation algorithm."""
        flat_position, _ = jax.flatten_util.ravel_pytree(state.position)
        n_dims = flat_position.shape[-1]
        mm_state = mm_init(n_dims)
        return mm_state

    def update(
        fs_state: Tuple[Array, HMCState, Any, StanWarmupState]
    ) -> StanWarmupState:
        """Move the warmup by one state when in a slow adaptation interval.

        Mass matrix adaptation and dual averaging states are both
        adapted in slow adaptation intervals, as indicated in Stan's
        reference manual.

        """
        rng_key, state, info, warmup_state = fs_state

        new_da_state = da_update(warmup_state.da_state, info)
        new_mm_state = mm_update(warmup_state.mm_state, state.position)
        new_warmup_state = StanWarmupState(new_da_state, new_mm_state)

        return new_warmup_state

    def final(warmup_state: StanWarmupState) -> StanWarmupState:
        """Update the parameters at the end of a slow adaptation window.

        We compute the value of the mass matrix and reset the mass matrix
        adapation's internal state since middle windows are "memoryless".

        """
        new_mm_state = mm_final(warmup_state.mm_state)
        new_da_state = da_init(da_final(warmup_state.da_state))
        new_warmup_state = StanWarmupState(new_da_state, new_mm_state)

        return new_warmup_state

    return init, update, final


def stan_warmup_schedule(
    num_steps: int,
    initial_buffer_size: int = 75,
    final_buffer_size: int = 50,
    first_window_size: int = 25,
) -> List[Tuple[int, bool]]:
    """Return the schedule for Stan's warmup.

    The schedule below is intended to be as close as possible to Stan's _[1].
    The warmup period is split into three stages:

    1. An initial fast interval to reach the typical set. Only the step size is
    adapted in this window.
    2. "Slow" parameters that require global information (typically covariance)
    are estimated in a series of expanding intervals with no memory; the step
    size is re-initialized at the end of each window. Each window is twice the
    size of the preceding window.
    3. A final fast interval during which the step size is adapted using the
    computed mass matrix.

    Schematically:

    ```
    +---------+---+------+------------+------------------------+------+
    |  fast   | s | slow |   slow     |        slow            | fast |
    +---------+---+------+------------+------------------------+------+
    ```

    The distinction slow/fast comes from the speed at which the algorithms
    converge to a stable value; in the common case, estimation of covariance
    requires more steps than dual averaging to give an accurate value. See _[1]
    for a more detailed explanation.

    Fast intervals are given the label 0 and slow intervals the label 1.

    Note
    ----
    It feels awkward to return a boolean that indicates whether the current
    step is the last step of a middle window, but not for other windows. This
    should probably be changed to "is_window_end" and we should manage the
    distinction upstream.

    Parameters
    ----------
    num_steps: int
        The number of warmup steps to perform.
    initial_buffer: int
        The width of the initial fast adaptation interval.
    first_window_size: int
        The width of the first slow adaptation interval.
    final_buffer_size: int
        The width of the final fast adaptation interval.

    Returns
    -------
    A list of tuples (window_label, is_middle_window_end).

    References
    ----------
    .. [1]: Stan Reference Manual v2.22
            Section 15.2 "HMC Algorithm"

    """
    schedule = []

    # Give up on mass matrix adaptation when the number of warmup steps is too small.
    if num_steps < 20:
        schedule += [(0, False)] * (num_steps - 1)
    else:
        # When the number of warmup steps is smaller that the sum of the provided (or default)
        # window sizes we need to resize the different windows.
        if initial_buffer_size + first_window_size + final_buffer_size > num_steps:
            initial_buffer_size = int(0.15 * num_steps)
            final_buffer_size = int(0.1 * num_steps)
            first_window_size = num_steps - initial_buffer_size - final_buffer_size

        # First stage: adaptation of fast parameters
        schedule += [(0, False)] * (initial_buffer_size - 1)
        schedule.append((0, False))

        # Second stage: adaptation of slow parameters in successive windows
        # doubling in size.
        final_buffer_start = num_steps - final_buffer_size

        next_window_size = first_window_size
        next_window_start = initial_buffer_size
        while next_window_start < final_buffer_start:
            current_start, current_size = next_window_start, next_window_size
            if 3 * current_size <= final_buffer_start - current_start:
                next_window_size = 2 * current_size
            else:
                current_size = final_buffer_start - current_start
            next_window_start = current_start + current_size
            schedule += [(1, False)] * (next_window_start - 1 - current_start)
            schedule.append((1, True))

        # Last stage: adaptation of fast parameters
        schedule += [(0, False)] * (num_steps - 1 - final_buffer_start)
        schedule.append((0, False))

    return schedule
