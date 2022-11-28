# Copyright Contributors to the Numpyro project.
# Copyright 2020- The Blackjax Authors
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
from typing import NamedTuple

import jax
import jax.numpy as jnp

from blackjax.types import Array


class IterativeUTurnState(NamedTuple):
    momentum: Array
    momentum_sum: Array
    idx_min: int
    idx_max: int


def iterative_uturn_numpyro(is_turning):
    """Numpyro style dynamic U-Turn criterion."""

    def new_state(chain_state, max_num_doublings) -> IterativeUTurnState:
        flat, _ = jax.flatten_util.ravel_pytree(chain_state.position)
        num_dims = jnp.shape(flat)[0]
        return IterativeUTurnState(
            jnp.zeros((max_num_doublings, num_dims)),
            jnp.zeros((max_num_doublings, num_dims)),
            0,
            0,
        )

    def update_criterion_state(
        checkpoints: IterativeUTurnState,
        momentum_sum,
        momentum,
        step: int,
    ):
        r_ckpts, r_sum_ckpts, _, _ = checkpoints
        ckpt_idx_min, ckpt_idx_max = _leaf_idx_to_ckpt_idxs(step)
        r, _ = jax.flatten_util.ravel_pytree(momentum)
        r_sum, _ = jax.flatten_util.ravel_pytree(momentum_sum)
        r_ckpts, r_sum_ckpts = jax.lax.cond(
            step % 2 == 0,
            (r_ckpts, r_sum_ckpts),
            lambda x: (x[0].at[ckpt_idx_max].set(r), x[1].at[ckpt_idx_max].set(r_sum)),
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
        idx_min = idx_max - num_subtrees + 1
        return idx_min, idx_max

    def _is_iterative_turning(checkpoints, momentum_sum, momentum):
        """Checks whether there is a U-turn in the iteratively built expanded trajectory.

        These checks only need to be performed as specific points.

        """
        r, _ = jax.flatten_util.ravel_pytree(momentum)
        r_sum, _ = jax.flatten_util.ravel_pytree(momentum_sum)
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
