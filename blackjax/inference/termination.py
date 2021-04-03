from typing import NamedTuple

import jax
import jax.numpy as jnp

from blackjax.inference.integrators import IntegratorState
# from blackjax.inference.proposals import Trajectory


class IterativeUTurnState(NamedTuple):
    momentum: jnp.ndarray
    momentum_sum: jnp.ndarray
    idx_min: int
    idx_max: int


def iterative_uturn_numpyro(is_turning):
    """Numpyro style dynamic U-Turn criterion."""

    def new_state(num_dims, max_tree_depth):
        return IterativeUTurnState(
            jnp.zeros((max_tree_depth, num_dims)),
            jnp.zeros((max_tree_depth, num_dims)),
            0,
            0,
        )

    def update_criterion_state(
        checkpoints: IterativeUTurnState,
        trajectory,  #: Trajectory,
        state: IntegratorState,
        step: int,
    ):
        r_ckpts, r_sum_ckpts, _, _ = checkpoints
        ckpt_idx_min, ckpt_idx_max = _leaf_idx_to_ckpt_idxs(step)
        r, _ = jax.flatten_util.ravel_pytree(state.momentum)
        r_sum, _ = jax.flatten_util.ravel_pytree(trajectory.momentum_sum)
        r_ckpts, r_sum_ckpts = jax.lax.cond(
            step % 2 == 0,
            (r_ckpts, r_sum_ckpts),
            lambda x: (
                jax.ops.index_update(x[0], ckpt_idx_max, r),
                jax.ops.index_update(x[1], ckpt_idx_max, r_sum),
            ),
            (r_ckpts, r_sum_ckpts),
            lambda x: x,
        )
        return IterativeUTurnState(r_ckpts, r_sum_ckpts, ckpt_idx_min, ckpt_idx_max)

    def _leaf_idx_to_ckpt_idxs(n):
        """Find the checkpoint id from a step number."""
        # computes the number of non-zero bits except the last bit
        # e.g. 6 -> 2, 7 -> 2, 13 -> 2
        _, idx_max = jax.lax.while_loop(
            lambda nc: nc[0] > 0,
            lambda nc: (nc[0] >> 1, nc[1] + (nc[0] & 1)),
            (n >> 1, 0),
        )
        # computes the number of contiguous last non-zero bits
        # e.g. 6 -> 0, 7 -> 3, 13 -> 1
        _, num_subtrees = jax.lax.while_loop(
            lambda nc: (nc[0] & 1) != 0, lambda nc: (nc[0] >> 1, nc[1] + 1), (n, 0)
        )
        # TODO: explore the potential of setting idx_min=0 to allow more turning checks
        # It will be useful in case: e.g. assume a tree 0 -> 7 is a circle,
        # subtrees 0 -> 3, 4 -> 7 are half-circles, which two leaves might not
        # satisfy turning condition;
        # the full tree 0 -> 7 is a circle, which two leaves might also not satisfy
        # turning condition;
        # however, we can check the turning condition of the subtree 0 -> 5, which
        # likely satisfies turning condition because its trajectory 3/4 of a circle.
        # XXX: make sure that detailed balance is satisfied if we follow this direction
        idx_min = idx_max - num_subtrees + 1
        return idx_min, idx_max

    def _is_iterative_turning(checkpoints, trajectory, state):
        """Checks whether there is a U-turn in the iteratively built expanded trajectory.
        These checks only need to be performed as specific points.

        Does that include the robust U-turn check?
        """

        r, _ = jax.flatten_util.ravel_pytree(state.momentum)
        r_sum, _ = jax.flatten_util.ravel_pytree(trajectory.momentum_sum)
        r_ckpts, r_sum_ckpts, idx_min, idx_max = checkpoints

        def _body_fn(state):
            i, _ = state
            subtree_r_sum = r_sum - r_sum_ckpts[i] + r_ckpts[i]
            return i - 1, is_turning(r_ckpts[i], r, subtree_r_sum)

        _, turning = jax.lax.while_loop(
            lambda it: (it[0] >= idx_min) & ~it[1], _body_fn, (idx_max, False)
        )
        return turning

    return new_state, update_criterion_state, _is_iterative_turning
