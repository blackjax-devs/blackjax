# flake8: noqa
# type: ignore
# ensure that the blackjax implementation aligns with the original
# we copy the original implementation of MCLMC (from https://github.com/JakobRobnik/MicroCanonicalHMC) almost wholesale, in order to have a self-contained reference implementation

from enum import Enum
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.config import config
from scipy.fftpack import next_fast_len

import blackjax
from blackjax.adaptation.mclmc_adaptation import mclmc_find_L_and_step_size
from blackjax.mcmc.integrators import (
    esh_dynamics_momentum_update_one_step,
    noneuclidean_mclachlan,
)
from blackjax.mcmc.mclmc import build_kernel, init
from blackjax.util import run_inference_algorithm

config.update("jax_enable_x64", True)

lambda_c = 0.1931833275037836


class State(NamedTuple):
    """Dynamical state"""

    x: any
    u: any
    l: float
    g: any
    key: tuple


class Hyperparameters(NamedTuple):
    """Tunable parameters"""

    L: float
    eps: float
    sigma: any


def update_momentum(d):
    """The momentum updating map of the esh dynamics (see https://arxiv.org/pdf/2111.02434.pdf)
    similar to the implementation: https://github.com/gregversteeg/esh_dynamics
    There are no exponentials e^delta, which prevents overflows when the gradient norm is large.
    """

    def update(eps, u, g):
        g_norm = jnp.sqrt(jnp.sum(jnp.square(g)))
        e = -g / g_norm
        ue = jnp.dot(u, e)
        delta = eps * g_norm / (d - 1)
        zeta = jnp.exp(-delta)
        uu = e * (1 - zeta) * (1 + zeta + ue * (1 - zeta)) + 2 * zeta * u
        delta_r = delta - jnp.log(2) + jnp.log(1 + ue + (1 - ue) * zeta**2)
        return uu / jnp.sqrt(jnp.sum(jnp.square(uu))), delta_r * (d - 1)

    return update


def update_position(grad_nlogp):
    def update(eps, x, u, sigma):
        xx = x + eps * u * sigma
        ll, gg = grad_nlogp(xx)
        return xx, u, ll, gg

    return update


def minimal_norm(T, V):
    def step(x, u, g, eps, sigma):
        """Integrator from https://arxiv.org/pdf/hep-lat/0505020.pdf, see Equation 20."""

        # V T V T V
        uu, r1 = V(eps * lambda_c, u, g * sigma)
        xx, uu, ll, gg = T(0.5 * eps, x, uu, sigma)
        uu, r2 = V(eps * (1 - 2 * lambda_c), uu, gg * sigma)
        xx, uu, ll, gg = T(0.5 * eps, xx, uu, sigma)
        uu, r3 = V(eps * lambda_c, uu, gg * sigma)

        # kinetic energy change
        kinetic_change = r1 + r2 + r3

        return xx, uu, ll, gg, kinetic_change

    return step


class Target:
    """#Class for target distribution

      E.g.

      ```python
      Target(d=2, nlogp = lambda x: 0.5*jnp.sum(jnp.square(x)))
    ```

      defines a Gaussian.

    """

    def __init__(self, d, nlogp):
        self.d = d
        """dimensionality of the target distribution"""
        self.nlogp = nlogp
        """ negative log probability of target distribution (i.e. energy function)"""
        self.grad_nlogp = jax.value_and_grad(self.nlogp)
        """ function which computes nlogp and its gradient"""

    def transform(self, x):
        """a transformation of the samples from the target distribution"""
        return x

    def prior_draw(self, key):
        """**Args**: jax random key

        **Returns**: one random sample from the prior
        """

        raise Exception("Not implemented")


OutputType = Enum("Output", ["normal", "detailed", "ess"])
""" @private """


def mclmc(hamilton, partial, get_nu):
    def step(dyn, hyp):
        """One step of the generalized dynamics."""

        # Hamiltonian step
        x, u, l, g, kinetic_change = hamilton(
            x=dyn.x, u=dyn.u, g=dyn.g, eps=hyp.eps, sigma=hyp.sigma
        )

        # Langevin-like noise
        u, key = partial(u=u, random_key=dyn.key, nu=get_nu(hyp.L / hyp.eps))

        energy_change = kinetic_change + l - dyn.l

        return State(x, u, l, g, key), energy_change

    return step


def full_refresh(d):
    """Generates a random (isotropic) unit vector."""

    def rng(random_key):
        key, subkey = jax.random.split(random_key)
        u = jax.random.normal(jax.random.PRNGKey(0), shape=(d,))
        u /= jnp.sqrt(jnp.sum(jnp.square(u)))
        return u, key

    return rng


def partial_refresh(d):
    """Adds a small noise to u and normalizes."""

    def rng(u, random_key, nu):
        key, subkey = jax.random.split(random_key)
        z = nu * jax.random.normal(jax.random.PRNGKey(0), shape=(d,))

        return (u + z) / jnp.sqrt(jnp.sum(jnp.square(u + z))), key

    get_nu = lambda Nd: jnp.sqrt(
        (jnp.exp(2.0 / Nd) - 1.0) / d
    )  # MCHMC paper (Nd = L/eps)

    return rng, get_nu


class Sampler:
    """the MCHMC (q = 0 Hamiltonian) sampler"""

    def __init__(
        self,
        Target: Target,
        L=None,
        eps=None,
        integrator=minimal_norm,
        varEwanted=5e-4,
        diagonal_preconditioning=False,
        frac_tune1=0.1,
        frac_tune2=0.1,
        frac_tune3=0.1,
        boundary=None,
    ):
        self.Target = Target

        self.integrator = integrator

        hamiltonian_step = self.integrator(
            T=update_position(self.Target.grad_nlogp), V=update_momentum(self.Target.d)
        )
        self.step = mclmc(hamiltonian_step, *partial_refresh(self.Target.d))
        self.full_refresh = full_refresh(self.Target.d)

        self.hyp = Hyperparameters(
            L if L != None else jnp.sqrt(self.Target.d),
            eps if eps != None else jnp.sqrt(self.Target.d) * 0.25,
            jnp.ones(self.Target.d),
        )

        tune12var = tune12(
            self.step,
            self.Target.d,
            diagonal_preconditioning,
            jnp.array([frac_tune1, frac_tune2]),
            varEwanted,
            1.5,
            150,
        )
        tune3var = tune3(self.step, frac_tune3, 0.4)

        if frac_tune3 != 0.0:
            tune3var = tune3(self.step, frac=frac_tune3, Lfactor=0.4)
            self.schedule = [tune12var, tune3var]
        else:
            self.schedule = [
                tune12var,
            ]

    ### sampling routine ###

    def initialize(self, x_initial, random_key):
        ### random key ###
        if random_key is None:
            key = jax.random.PRNGKey(0)
        else:
            key = random_key

        ### initial conditions ###
        if x_initial is None:  # draw the initial x from the prior
            key, prior_key = jax.random.split(key)
            x = self.Target.prior_draw(prior_key)
        else:
            x = x_initial

        l, g = self.Target.grad_nlogp(x)

        u, key = self.full_refresh(key)
        # u = - g / jnp.sqrt(jnp.sum(jnp.square(g))) #initialize momentum in the direction of the gradient of log p

        return State(x, u, l, g, key)

    def sample(
        self,
        num_steps,
        num_chains=1,
        x_initial=None,
        random_key=None,
        output=OutputType.normal,
        thinning=1,
    ):
        """Args:
        num_steps: number of integration steps to take.

        num_chains: number of independent chains, defaults to 1. If different than 1, jax will parallelize the computation with the number of available devices (CPU, GPU, TPU),
        as returned by jax.local_device_count().

        x_initial: initial condition for x, shape: (d, ). Defaults to None in which case the initial condition is drawn from the prior distribution (self.Target.prior_draw).

        random_key: jax random seed, defaults to jax.random.PRNGKey(0)

        output: determines the output of the function:

                 'normal': samples
                     samples were transformed by the Target.transform to save memory and have shape: (num_samples, len(Target.transform(x)))

                 'detailed': samples, energy error at each step and -log p(x) at each step

                 'ess': Effective Sample Size per gradient evaluation, float.
                     In this case, ground truth E[x_i^2] and Var[x_i^2] should be known and defined as self.Target.second_moments and self.Target.variance_second_moments

                 Note: in all cases the hyperparameters that were used for sampling can be accessed through Sampler.hyp

         thinning: only one every 'thinning' steps is stored. Defaults to 1 (the output then contains (num_steps / thinning) samples)
                 This is not the recommended solution to save memory. It is better to use the transform functionality, when possible.
        """

        if output == OutputType.ess:
            for ground_truth in ["second_moments", "variance_second_moments"]:
                if not hasattr(self.Target, ground_truth):
                    raise AttributeError(
                        "Target."
                        + ground_truth
                        + " should be defined if you want to use output = ess."
                    )

        if num_chains == 1:
            results = self.single_chain_sample(
                num_steps, x_initial, random_key, output, thinning
            )  # the function which actually does the sampling
            return results
        else:
            num_cores = jax.local_device_count()
            if random_key is None:
                key = jax.random.PRNGKey(0)
            else:
                key = random_key

            if x_initial is None:  # draw the initial x from the prior
                keys_all = jax.random.split(key, num_chains * 2)
                x0 = jnp.array(
                    [
                        self.Target.prior_draw(keys_all[num_chains + i])
                        for i in range(num_chains)
                    ]
                )
                keys = keys_all[:num_chains]

            else:  # initial x is given
                x0 = jnp.copy(x_initial)
                keys = jax.random.split(key, num_chains)

            f = lambda i: self.single_chain_sample(
                num_steps, x0[i], keys[i], output, thinning
            )

            if num_cores != 1:  # run the chains on parallel cores
                parallel_function = jax.pmap(jax.vmap(f))
                results = parallel_function(
                    jnp.arange(num_chains).reshape(num_cores, num_chains // num_cores)
                )
                ### reshape results ###
                if type(results) is tuple:  # each chain returned a tuple
                    results_reshaped = []
                    for i in range(len(results)):
                        res = jnp.array(results[i])
                        results_reshaped.append(
                            res.reshape(
                                [
                                    num_chains,
                                ]
                                + [res.shape[j] for j in range(2, len(res.shape))]
                            )
                        )
                    return results_reshaped

                else:
                    return results.reshape(
                        [
                            num_chains,
                        ]
                        + [results.shape[j] for j in range(2, len(results.shape))]
                    )

            else:  # run chains serially on a single core
                results = jax.vmap(f)(jnp.arange(num_chains))

                return results

    def single_chain_sample(self, num_steps, x_initial, random_key, output, thinning):
        """sampling routine. It is called by self.sample"""
        ### initial conditions ###
        dyn = self.initialize(x_initial, random_key)

        hyp = self.hyp

        ### tuning ###
        dyn, hyp = run(dyn, hyp, self.schedule, num_steps)
        self.hyp = hyp

        ### sampling ###

        if output == OutputType.normal or output == OutputType.detailed:
            X, l, E = self.sample_normal(num_steps, dyn, hyp, thinning)
            if output == OutputType.detailed:
                return X, E, l
            else:
                return X

        elif output == OutputType.ess:
            return self.sample_ess(num_steps, dyn, hyp)

    def build_kernel(self, thinning: int):
        """kernel for sampling_normal"""

        def kernel_with_thinning(dyn, hyp):
            def substep(state, _):
                _dyn, energy_change = self.step(state[0], hyp)
                return (_dyn, energy_change), None

            return jax.lax.scan(substep, init=(dyn, 0.0), xs=None, length=thinning)[
                0
            ]  # do 'thinning' steps without saving

        if thinning == 1:
            return self.step
        else:
            return kernel_with_thinning

    def sample_normal(
        self, num_steps: int, _dyn: State, hyp: Hyperparameters, thinning: int
    ):
        """Stores transform(x) for each step."""

        kernel = self.build_kernel(thinning)

        def step(state, _):
            dyn, energy_change = kernel(state, hyp)

            return dyn, (self.Target.transform(dyn.x), dyn.l, energy_change)

        return jax.lax.scan(step, init=_dyn, xs=None, length=num_steps // thinning)[1]

    def sample_ess(self, num_steps: int, _dyn: State, hyp: Hyperparameters):
        """Stores the bias of the second moments for each step."""

        def step(state_track, useless):
            dyn, kalman_state = state_track
            dyn, _ = self.step(dyn, hyp)
            kalman_state = kalman_step(kalman_state)
            return (dyn, kalman_state), bias(kalman_state[1])

        def kalman_step(state, x):
            W, F2 = state
            F2 = (W * F2 + jnp.square(self.Target.transform(x))) / (
                W + 1
            )  # Update <f(x)> with a Kalman filter
            W += 1
            return W, F2

        def bias(x2):
            bias_d = (
                jnp.square(x2 - self.Target.second_moments)
                / self.Target.variance_second_moments
            )
            bavg2 = jnp.average(bias_d)
            # bmax2 = jnp.max(bias_d)
            return bavg2

        _, b = jax.lax.scan(
            step,
            init=(_dyn, (1, jnp.square(self.Target.transform(_dyn.x)))),
            xs=None,
            length=num_steps,
        )

        return b


def run(dyn, hyp, schedule, num_steps):
    _dyn, _hyp = dyn, hyp

    for program in schedule:
        _dyn, _hyp = program(_dyn, _hyp, num_steps)

    return _dyn, _hyp


def nan_reject(x, u, l, g, xx, uu, ll, gg, eps, eps_max, dK):
    """if there are nans, let's reduce the stepsize, and not update the state. The function returns the old state in this case."""

    nonans = jnp.all(jnp.isfinite(xx))
    _x, _u, _l, _g, _eps, _dk = jax.tree_util.tree_map(
        lambda new, old: jax.lax.select(nonans, jnp.nan_to_num(new), old),
        (xx, uu, ll, gg, eps_max, dK),
        (x, u, l, g, eps * 0.8, 0.0),
    )

    return nonans, _x, _u, _l, _g, _eps, _dk


def tune12(dynamics, d, diag_precond, frac, varEwanted=1e-3, sigma_xi=1.5, neff=150):
    gamma_forget = (neff - 1.0) / (neff + 1.0)

    def predictor(dyn_old, hyp, adaptive_state):
        """does one step with the dynamics and updates the prediction for the optimal stepsize
        Designed for the unadjusted MCHMC"""

        W, F, eps_max = adaptive_state

        # dynamics
        dyn_new, energy_change = dynamics(dyn_old, hyp)

        # step updating
        success, x, u, l, g, eps_max, energy_change = nan_reject(
            dyn_old.x,
            dyn_old.u,
            dyn_old.l,
            dyn_old.g,
            dyn_new.x,
            dyn_new.u,
            dyn_new.l,
            dyn_new.g,
            hyp.eps,
            eps_max,
            energy_change,
        )

        dyn = State(x, u, l, g, dyn_new.key)

        # Warning: var = 0 if there were nans, but we will give it a very small weight
        xi = (
            jnp.square(energy_change) / (d * varEwanted)
        ) + 1e-8  # 1e-8 is added to avoid divergences in log xi
        w = jnp.exp(
            -0.5 * jnp.square(jnp.log(xi) / (6.0 * sigma_xi))
        )  # the weight reduces the impact of stepsizes which are much larger on much smaller than the desired one.

        F = gamma_forget * F + w * (xi / jnp.power(hyp.eps, 6.0))
        W = gamma_forget * W + w
        eps = jnp.power(
            F / W, -1.0 / 6.0
        )  # We use the Var[E] = O(eps^6) relation here.
        eps = (eps < eps_max) * eps + (
            eps > eps_max
        ) * eps_max  # if the proposed stepsize is above the stepsize where we have seen divergences
        hyp_new = Hyperparameters(hyp.L, eps, hyp.sigma)

        return dyn, hyp_new, hyp_new, (W, F, eps_max), success

    def update_kalman(x, state, outer_weight, success, eps):
        """kalman filter to estimate the size of the posterior"""
        W, F1, F2 = state
        w = outer_weight * eps * success
        zero_prevention = 1 - outer_weight
        F1 = (W * F1 + w * x) / (
            W + w + zero_prevention
        )  # Update <f(x)> with a Kalman filter
        F2 = (W * F2 + w * jnp.square(x)) / (
            W + w + zero_prevention
        )  # Update <f(x)> with a Kalman filter
        W += w
        return (W, F1, F2)

    adap0 = (0.0, 0.0, jnp.inf)
    _step = predictor

    def step(state, outer_weight):
        """does one step of the dynamcis and updates the estimate of the posterior size and optimal stepsize"""
        dyn, hyp, _, adaptive_state, kalman_state = state
        dyn, hyp, hyp_final, adaptive_state, success = _step(dyn, hyp, adaptive_state)
        kalman_state = update_kalman(
            dyn.x, kalman_state, outer_weight, success, hyp.eps
        )

        return (dyn, hyp, hyp_final, adaptive_state, kalman_state), None

    def func(_dyn, _hyp, num_steps):
        num_steps1, num_steps2 = jnp.rint(num_steps * frac).astype(int)

        # we use the last num_steps2 to compute the diagonal preconditioner
        outer_weights = jnp.concatenate((jnp.zeros(num_steps1), jnp.ones(num_steps2)))

        # initial state

        kalman_state = (0.0, jnp.zeros(d), jnp.zeros(d))

        # run the steps
        state = jax.lax.scan(
            step,
            init=(_dyn, _hyp, _hyp, adap0, kalman_state),
            xs=outer_weights,
            length=num_steps1 + num_steps2,
        )[0]
        dyn, _, hyp, adap, kalman_state = state

        # determine L
        L = hyp.L
        sigma = hyp.sigma
        if num_steps2 != 0.0:
            _, F1, F2 = kalman_state
            variances = F2 - jnp.square(F1)
            L = jnp.sqrt(jnp.sum(variances))

            # optionally we do the diagonal preconditioning (and readjust the stepsize)
            if diag_precond:
                # diagonal preconditioning
                sigma = jnp.sqrt(variances)
                L = jnp.sqrt(d)

                # readjust the stepsize
                steps = num_steps2 // 3  # we do some small number of steps
                state = jax.lax.scan(
                    step, init=state, xs=jnp.ones(steps), length=steps
                )[0]
                dyn, _, hyp, adap, kalman_state = state
            else:
                sigma = hyp.sigma

        return dyn, Hyperparameters(L, hyp.eps, sigma)

    return func


def ess_corr(x):
    """Taken from: https://blackjax-devs.github.io/blackjax/diagnostics.html
    shape(x) = (num_samples, d)"""

    input_array = jnp.array(
        [
            x,
        ]
    )

    num_chains = 1  # input_array.shape[0]
    num_samples = input_array.shape[1]

    mean_across_chain = input_array.mean(axis=1, keepdims=True)
    # Compute autocovariance estimates for every lag for the input array using FFT.
    centered_array = input_array - mean_across_chain
    m = next_fast_len(2 * num_samples)
    ifft_ary = jnp.fft.rfft(centered_array, n=m, axis=1)
    ifft_ary *= jnp.conjugate(ifft_ary)
    autocov_value = jnp.fft.irfft(ifft_ary, n=m, axis=1)
    autocov_value = (
        jnp.take(autocov_value, jnp.arange(num_samples), axis=1) / num_samples
    )
    mean_autocov_var = autocov_value.mean(0, keepdims=True)
    mean_var0 = (
        jnp.take(mean_autocov_var, jnp.array([0]), axis=1)
        * num_samples
        / (num_samples - 1.0)
    )
    weighted_var = mean_var0 * (num_samples - 1.0) / num_samples
    weighted_var = jax.lax.cond(
        num_chains > 1,
        lambda _: weighted_var + mean_across_chain.var(axis=0, ddof=1, keepdims=True),
        lambda _: weighted_var,
        operand=None,
    )

    # Geyer's initial positive sequence
    num_samples_even = num_samples - num_samples % 2
    mean_autocov_var_tp1 = jnp.take(
        mean_autocov_var, jnp.arange(1, num_samples_even), axis=1
    )
    rho_hat = jnp.concatenate(
        [
            jnp.ones_like(mean_var0),
            1.0 - (mean_var0 - mean_autocov_var_tp1) / weighted_var,
        ],
        axis=1,
    )

    rho_hat = jnp.moveaxis(rho_hat, 1, 0)
    rho_hat_even = rho_hat[0::2]
    rho_hat_odd = rho_hat[1::2]

    mask0 = (rho_hat_even + rho_hat_odd) > 0.0
    carry_cond = jnp.ones_like(mask0[0])
    max_t = jnp.zeros_like(mask0[0], dtype=int)

    def positive_sequence_body_fn(state, mask_t):
        t, carry_cond, max_t = state
        next_mask = carry_cond & mask_t
        next_max_t = jnp.where(next_mask, jnp.ones_like(max_t) * t, max_t)
        return (t + 1, next_mask, next_max_t), next_mask

    (*_, max_t_next), mask = jax.lax.scan(
        positive_sequence_body_fn, (0, carry_cond, max_t), mask0
    )
    indices = jnp.indices(max_t_next.shape)
    indices = tuple([max_t_next + 1] + [indices[i] for i in range(max_t_next.ndim)])
    rho_hat_odd = jnp.where(mask, rho_hat_odd, jnp.zeros_like(rho_hat_odd))
    # improve estimation
    mask_even = mask.at[indices].set(rho_hat_even[indices] > 0)
    rho_hat_even = jnp.where(mask_even, rho_hat_even, jnp.zeros_like(rho_hat_even))

    # Geyer's initial monotone sequence
    def monotone_sequence_body_fn(rho_hat_sum_tm1, rho_hat_sum_t):
        update_mask = rho_hat_sum_t > rho_hat_sum_tm1
        next_rho_hat_sum_t = jnp.where(update_mask, rho_hat_sum_tm1, rho_hat_sum_t)
        return next_rho_hat_sum_t, (update_mask, next_rho_hat_sum_t)

    rho_hat_sum = rho_hat_even + rho_hat_odd
    _, (update_mask, update_value) = jax.lax.scan(
        monotone_sequence_body_fn, rho_hat_sum[0], rho_hat_sum
    )

    rho_hat_even_final = jnp.where(update_mask, update_value / 2.0, rho_hat_even)
    rho_hat_odd_final = jnp.where(update_mask, update_value / 2.0, rho_hat_odd)

    # compute effective sample size
    ess_raw = num_chains * num_samples
    tau_hat = (
        -1.0
        + 2.0 * jnp.sum(rho_hat_even_final + rho_hat_odd_final, axis=0)
        - rho_hat_even_final[indices]
    )

    tau_hat = jnp.maximum(tau_hat, 1 / np.log10(ess_raw))
    ess = ess_raw / tau_hat

    ### my part (combine all dimensions): ###
    neff = ess.squeeze() / num_samples
    return 1.0 / jnp.average(1 / neff)


def tune3(step, frac, Lfactor):
    """determine L by the autocorrelations (around 10 effective samples are needed for this to be accurate)"""

    def sample_full(num_steps, _dyn, hyp):
        """Stores full x for each step. Used in tune2."""

        def _step(state, useless):
            dyn_old = state
            dyn_new, _ = step(dyn_old, hyp)

            return dyn_new, dyn_new.x

        return jax.lax.scan(_step, init=_dyn, xs=None, length=num_steps)

    def func(dyn, hyp, num_steps):
        steps = jnp.rint(num_steps * frac).astype(int)

        dyn, X = sample_full(steps, dyn, hyp)
        ESS = ess_corr(X)  # num steps / effective sample size
        Lnew = (
            Lfactor * hyp.eps / ESS
        )  # = 0.4 * length corresponding to one effective sample

        return dyn, Hyperparameters(Lnew, hyp.eps, hyp.sigma)

    return func


# tests


def test_momentum_update():
    dim = 10
    logdensity_fn = lambda x: -0.5 * jnp.sum(jnp.square(x))
    step_size = 1e-3

    initial_position = jnp.ones(dim)
    initial_state = blackjax.mcmc.mclmc.init(
        initial_position, logdensity_fn, rng_key=jax.random.PRNGKey(0)
    )

    (
        blackjax_momentum,
        _,
        blackjax_kinetic_energy,
    ) = esh_dynamics_momentum_update_one_step(
        momentum=initial_state.momentum,
        logdensity_grad=initial_state.logdensity_grad,
        step_size=step_size,
        coef=1.0,
    )

    original_momentum, original_kinetic_energy = update_momentum(dim)(
        step_size, initial_state.momentum, -initial_state.logdensity_grad
    )

    assert jnp.allclose(blackjax_momentum, original_momentum)
    assert blackjax_kinetic_energy == original_kinetic_energy


# test that the non-euclidean integrator agrees with a simple implementation exactly
def test_non_euclidean_implementation():
    dim = 2
    logdensity_fn = lambda x: -0.5 * jnp.sum(jnp.square(x))
    step_size = 1e-3
    initial_position = jnp.array([1.0, 1.0])

    initial_state = blackjax.mcmc.mclmc.init(
        initial_position, logdensity_fn, rng_key=jax.random.PRNGKey(0)
    )
    # raise Exception(step(initial_state, step_size=1e-3))

    blackjax_result, blackjax_kinetic_change = noneuclidean_mclachlan(
        logdensity_fn=logdensity_fn
    )(initial_state, step_size=step_size)

    grad_nlogp = jax.value_and_grad(lambda x: -logdensity_fn(x))

    (
        original_position,
        original_momentum,
        original_nlogdensity,
        original_nlogdensity_grad,
        original_kinetic_change,
    ) = minimal_norm(
        T=update_position(grad_nlogp),
        V=update_momentum(dim),
    )(
        x=initial_state.position,
        u=initial_state.momentum,
        g=-initial_state.logdensity_grad,
        eps=step_size,
        sigma=1.0,
    )

    assert jnp.allclose(blackjax_result.position, original_position)
    assert jnp.allclose(blackjax_result.momentum, original_momentum)
    assert jnp.allclose(blackjax_result.logdensity, -original_nlogdensity)
    assert jnp.allclose(blackjax_result.logdensity_grad, -original_nlogdensity_grad)
    assert original_kinetic_change == blackjax_kinetic_change


def test_full_no_tuning():
    init_key = jax.random.PRNGKey(0)
    run_key = jax.random.PRNGKey(0)

    dim = 10
    logdensity_fn = lambda x: -0.5 * jnp.sum(jnp.square(x))
    step_size = 1e-3
    L = 1.0
    initial_position = jnp.ones(dim)
    num_steps = 10000

    initial_state = blackjax.mcmc.mclmc.init(
        position=initial_position, logdensity_fn=logdensity_fn, rng_key=init_key
    )

    sampling_alg = blackjax.mclmc(logdensity_fn, L=L, step_size=step_size)

    _, blackjax_samples, blackjax_info = run_inference_algorithm(
        rng_key=run_key,
        initial_state_or_position=initial_state,
        inference_algorithm=sampling_alg,
        num_steps=num_steps,
        transform=lambda x: x.position,
        seed=0,
    )

    target_simple = Target(d=dim, nlogp=lambda x: -logdensity_fn(x))
    original_mclmc_samples, original_energy_change, _ = Sampler(
        Target=target_simple,
        L=L,
        eps=step_size,
        frac_tune1=0.0,
        frac_tune2=0.0,
        frac_tune3=0.0,
    ).sample(
        num_steps,
        x_initial=initial_position,
        random_key=run_key,
        output=OutputType.detailed,
    )

    assert jnp.allclose(original_mclmc_samples, blackjax_samples)


def test_tune():
    num_steps = 1000
    num_chains = 1
    dim = 2
    key = jax.random.PRNGKey(0)

    initial_position = jnp.ones(dim)

    logdensity_fn = lambda x: -0.5 * jnp.sum(jnp.square(x))

    target_simple = Target(d=dim, nlogp=lambda x: -logdensity_fn(x))

    native_mclmc_sampler = Sampler(Target=target_simple)
    _ = native_mclmc_sampler.sample(
        num_steps, x_initial=initial_position, random_key=key
    )

    print(native_mclmc_sampler.hyp)

    kernel = build_kernel(
        logdensity_fn=logdensity_fn, integrator=noneuclidean_mclachlan
    )

    # run mclmc with tuning and get result
    initial_state = init(
        position=initial_position, logdensity_fn=logdensity_fn, rng_key=key
    )
    _, blackjax_mclmc_sampler_params = mclmc_find_L_and_step_size(
        mclmc_kernel=kernel,
        num_steps=num_steps,
        state=initial_state,
        rng_key=key,
        split_fn=lambda k, num: jnp.array([k] * num),
    )

    assert jnp.allclose(
        blackjax_mclmc_sampler_params.L, native_mclmc_sampler.hyp.L
    ) and jnp.allclose(
        blackjax_mclmc_sampler_params.step_size, native_mclmc_sampler.hyp.eps
    )
