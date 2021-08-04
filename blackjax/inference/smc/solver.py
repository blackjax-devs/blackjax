"""All things solving for adaptive tempering."""

import jax
import numpy as np


def dichotomy(fun, _delta0, min_delta, max_delta, eps=1e-4, max_iter=100):
    """Solves for delta by dichotomy.

    If max_delta is such that fun(max_delta) > 0, then we assume that max_delta
    can be used as an increment in the tempering.

    Parameters
    ----------
    fun: Callable
        The decreasing function to solve, we must have fun(min_delta) > 0, fun(max_delta) < 0
    min_delta: float
        Starting point of the interval search
    max_delta: float
        End point of the interval search
    eps: float
        Tolerance for |f(a) - f(b)|
    max_iter: int
        Maximum of iterations in the dichotomy search

    Returns
    -------
    delta: jnp.ndarray, shape (,)
        The root of `fun`

    """

    def body(carry):
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

    def cond(carry):
        import jax.numpy as jnp

        i, a, b, f_a, f_b = carry
        return jnp.logical_and(i < max_iter, f_a - f_b > eps)

    f_min_delta, f_max_delta = fun(min_delta), fun(max_delta)

    if_no_opt = lambda _: max_delta

    def if_opt(_):
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
