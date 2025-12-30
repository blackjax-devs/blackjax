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
"""All things solving for adaptive tempering."""
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from blackjax.types import Array


def dichotomy(
    fun: Callable,
    min_delta: float | Array,
    max_delta: float | Array,
    eps: float = 1e-4,
    max_iter: int = 100,
) -> Array:
    """Solves for delta by dichotomy.

    If max_delta is such that fun(max_delta) > 0, then we assume that max_delta
    can be used as an increment in the tempering.

    Parameters
    ----------
    fun: Callable
        The decreasing function to solve, we must have fun(min_delta) > 0,
        fun(max_delta) < 0
    min_delta: float
        Starting point of the interval search
    max_delta: float
        End point of the interval search
    eps: float
        Tolerance for :math:`|f(a) - f(b)|`
    max_iter: int
        Maximum of iterations in the dichotomy search

    Returns
    -------
    delta: Array, shape (,)
        The root of `fun`

    """

    def body(carry: tuple) -> tuple:
        i, a, b, f_a, f_b = carry

        mid = 0.5 * (a + b)
        f_mid = fun(mid)
        a, b, f_a, f_b = jax.lax.cond(
            f_mid < 0,
            lambda _: (a, mid, f_a, f_mid),
            lambda _: (mid, b, f_mid, f_b),
            None,
        )
        return i + 1, a, b, f_a, f_b

    def cond(carry: tuple) -> Array:
        i, a, b, f_a, f_b = carry
        return jnp.logical_and(i < max_iter, f_a - f_b > eps)

    f_min_delta, f_max_delta = fun(min_delta), fun(max_delta)

    def if_no_opt(_: Any) -> float | Array:
        return max_delta

    def if_opt(_: Any) -> float | Array:
        _, res_a, res_b, fun_res_a, fun_res_b = jax.lax.while_loop(
            cond, body, (0, min_delta, max_delta, f_min_delta, f_max_delta)
        )
        return res_a

    # if the upper end of the interval returns positive already, just return it,
    # otherwise search the optimum as long as the start of the interval is positive.
    return jax.lax.cond(
        f_max_delta > 0,
        if_no_opt,
        lambda _: jax.lax.cond(f_min_delta > 0, if_opt, lambda _: np.nan, None),
        None,
    )
