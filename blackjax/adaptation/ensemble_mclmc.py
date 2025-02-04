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
# """Public API for the MCLMC Kernel"""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

import blackjax.adaptation.ensemble_umclmc as umclmc
from blackjax.adaptation.ensemble_umclmc import (
    equipartition_diagonal,
    equipartition_diagonal_loss,
)
from blackjax.adaptation.step_size import bisection_monotonic_fn
from blackjax.mcmc.adjusted_mclmc import build_kernel as build_kernel_malt
from blackjax.mcmc.hmc import HMCState
from blackjax.mcmc.integrators import (
    generate_isokinetic_integrator,
    mclachlan_coefficients,
    omelyan_coefficients,
)
from blackjax.util import run_eca


class AdaptationState(NamedTuple):

    steps_per_sample: float
    step_size: float
    stepsize_adaptation_state: (
        Any  # the state of the bisection algorithm to find a stepsize
    )
    iteration: int


build_kernel = lambda logdensity_fn, integrator, inverse_mass_matrix: lambda key, state, adap: build_kernel_malt(
    logdensity_fn=logdensity_fn,
    integrator=integrator,
    inverse_mass_matrix=inverse_mass_matrix,
)(
    rng_key=key,
    state=state,
    step_size=adap.step_size,
    num_integration_steps=adap.steps_per_sample,
    L_proposal_factor=1.25,
)


class Adaptation:
    def __init__(
        self,
        adaptation_state,
        num_adaptation_samples,  # amount of tuning in the adjusted phase before fixing params
        steps_per_sample,  # L/eps (same for each chain: currently fixed to 15)
        acc_prob_target=0.8,
        observables=lambda x: 0.0,  # just for diagnostics: some function of a given chain at given timestep
        observables_for_bias=lambda x: 0.0,  # just for diagnostics: the above, but averaged over all chains
        contract=lambda x: 0.0,  # just for diagnostics: observabiels for bias, contracted over dimensions
    ):
        self.num_adaptation_samples = num_adaptation_samples
        self.observables = observables
        self.observables_for_bias = observables_for_bias
        self.contract = contract

        # Determine the initial hyperparameters #

        # stepsize #
        # if we switched to the more accurate integrator we can use longer step size
        # integrator_factor = jnp.sqrt(10.) if mclachlan else 1.
        # Let's use the stepsize which will be optimal for the adjusted method. The energy variance after N steps scales as sigma^2 ~ N^2 eps^6 = eps^4 L^2
        # In the adjusted method we want sigma^2 = 2 mu = 2 * 0.41 = 0.82
        # With the current eps, we had sigma^2 = EEVPD * d for N = 1.
        # Combining the two we have EEVPD * d / 0.82 = eps^6 / eps_new^4 L^2
        # adjustment_factor = jnp.power(0.82 / (num_dims * adaptation_state.EEVPD), 0.25) / jnp.sqrt(steps_per_sample)
        step_size = adaptation_state.step_size  # * integrator_factor * adjustment_factor

        # steps_per_sample = (int)(jnp.max(jnp.array([Lfull / step_size, 1])))

        # Initialize the dual averaging adaptation #
        # da_init_fn, self.epsadap_update, _ = dual_averaging_adaptation(target= acc_prob_target)
        # stepsize_adaptation_state = da_init_fn(step_size)

        # Initialize the bisection for finding the step size
        stepsize_adaptation_state, self.epsadap_update = bisection_monotonic_fn(
            acc_prob_target
        )

        self.initial_state = AdaptationState(
            steps_per_sample, step_size, stepsize_adaptation_state, 0
        )

    def summary_statistics_fn(self, state, info, rng_key):
        return {
            "acceptance_probability": info.acceptance_rate,
            "equipartition_diagonal": equipartition_diagonal(
                state
            ),  # metric for bias: equipartition theorem gives todo...
            "observables": self.observables(state.position),
            "observables_for_bias": self.observables_for_bias(state.position),
        }

    def update(self, adaptation_state, Etheta):
        acc_prob = Etheta["acceptance_probability"]
        equi_diag = equipartition_diagonal_loss(Etheta["equipartition_diagonal"])
        true_bias = self.contract(Etheta["observables_for_bias"])  # remove

        info_to_be_stored = {
            "L": adaptation_state.step_size * adaptation_state.steps_per_sample,
            "steps_per_sample": adaptation_state.steps_per_sample,
            "step_size": adaptation_state.step_size,
            "acc_prob": acc_prob,
            "equi_diag": equi_diag,
            "bias": true_bias,
            "observables": Etheta["observables"],
        }

        # Bisection to find step size
        stepsize_adaptation_state, step_size = self.epsadap_update(
            adaptation_state.stepsize_adaptation_state,
            adaptation_state.step_size,
            acc_prob,
        )

        return (
            AdaptationState(
                adaptation_state.steps_per_sample,
                step_size,
                stepsize_adaptation_state,
                adaptation_state.iteration + 1,
            ),
            info_to_be_stored,
        )


def bias(model):
    """should be transfered to benchmarks/"""

    def observables(position):
        return jnp.square(model.transform(position))

    def contract(sampler_E_x2):
        bsq = jnp.square(sampler_E_x2 - model.E_x2) / model.Var_x2
        return jnp.array([jnp.max(bsq), jnp.average(bsq)])

    return observables, contract


def while_steps_num(cond):
    if jnp.all(cond):
        return len(cond)
    else:
        return jnp.argmin(cond) + 1


def emaus(
    model,
    num_steps1,  # max number in phase 1
    num_steps2,  # fixed number in phase 2
    num_chains,
    mesh,
    rng_key,
    alpha=1.9,  # L = \sqrt{d}*\alpha*vars
    save_frac=0.2,  # to end stage one, the fraction of stage 1 samples used to estimate fluctuation. min is: save_frac*num_steps1
    C=0.1,  # constant in stage 1 that determines step size (eq (9) in paper)
    early_stop=True,  # for stage 1
    r_end=5e-3,  # stage1 parameters
    diagonal_preconditioning=True,
    integrator_coefficients=None,  # (for stage 2)
    steps_per_sample=10,
    acc_prob=None,
    observables=lambda x: None,
    ensemble_observables=None,
    diagnostics=True
):
    
    """
    model: the target density object
    num_steps1: number of steps in the first phase
    num_steps2: number of steps in the second phase
    num_chains: number of chains
    mesh: the mesh object, used for distributing the computation across cpus and nodes
    rng_key: the random key
    alpha: L = \sqrt{d}*\alpha*variances
    save_frac: the fraction of samples used to estimate the fluctuation in the first phase
    C: constant in stage 1 that determines step size (eq (9) of EMAUS paper)
    early_stop: whether to stop the first phase early
    r_end
    diagonal_preconditioning: whether to use diagonal preconditioning
    integrator_coefficients: the coefficients of the integrator
    steps_per_sample: the number of steps per sample
    acc_prob: the acceptance probability
    observables: the observables (for diagnostic use)
    ensemble_observables:  observable calculated over the ensemble (for diagnostic use)
    diagnostics: whether to return diagnostics
    """

    observables_for_bias, contract = bias(model)
    key_init, key_umclmc, key_mclmc = jax.random.split(rng_key, 3)

    # initialize the chains
    initial_state = umclmc.initialize(
        key_init, model.logdensity_fn, model.sample_init, num_chains, mesh
    )

    # burn-in with the unadjusted method #
    kernel = umclmc.build_kernel(model.logdensity_fn)
    save_num = (int)(jnp.rint(save_frac * num_steps1))
    adap = umclmc.Adaptation(
        model.ndims,
        alpha=alpha,
        bias_type=3,
        save_num=save_num,
        C=C,
        power=3.0 / 8.0,
        r_end=r_end,
        observables=observables,
        observables_for_bias=observables_for_bias,
        contract=contract,
    )
    
    final_state, final_adaptation_state, info1 = run_eca(
        key_umclmc,
        initial_state,
        kernel,
        adap,
        num_steps1,
        num_chains,
        mesh,
        ensemble_observables,
        early_stop=early_stop,
    )

    # refine the results with the adjusted method #
    _acc_prob = acc_prob
    if integrator_coefficients is None:
        high_dims = model.ndims > 200
        _integrator_coefficients = (
            omelyan_coefficients if high_dims else mclachlan_coefficients
        )
        if acc_prob is None:
            _acc_prob = 0.9 if high_dims else 0.7

    else:
        _integrator_coefficients = integrator_coefficients
        if acc_prob is None:
            _acc_prob = 0.9

    integrator = generate_isokinetic_integrator(_integrator_coefficients)
    gradient_calls_per_step = (
        len(_integrator_coefficients) // 2
    )  # scheme = BABAB..AB scheme has len(scheme)//2 + 1 Bs. The last doesn't count because that gradient can be reused in the next step.

    if diagonal_preconditioning:
        inverse_mass_matrix = jnp.sqrt(final_adaptation_state.inverse_mass_matrix)

        # scale the stepsize so that it reflects averag scale change of the preconditioning
        average_scale_change = jnp.sqrt(jnp.average(inverse_mass_matrix))
        final_adaptation_state = final_adaptation_state._replace(
            step_size=final_adaptation_state.step_size / average_scale_change
        )

    else:
        inverse_mass_matrix = 1.0

    kernel = build_kernel(
        model.logdensity_fn, integrator, inverse_mass_matrix=inverse_mass_matrix
    )
    initial_state = HMCState(
        final_state.position, final_state.logdensity, final_state.logdensity_grad
    )
    num_samples = num_steps2 // (gradient_calls_per_step * steps_per_sample)
    num_adaptation_samples = (
        num_samples // 2
    )  # number of samples after which the stepsize is fixed.

    adap = Adaptation(
        final_adaptation_state,
        num_adaptation_samples,
        steps_per_sample,
        _acc_prob,
        observables=observables,
        observables_for_bias=observables_for_bias,
        contract=contract,
    )

    final_state, final_adaptation_state, info2 = run_eca(
        key_mclmc,
        initial_state,
        kernel,
        adap,
        num_samples,
        num_chains,
        mesh,
        ensemble_observables,
    )

    if diagnostics:
        info = {"phase_1" : info1, "phase_2" : info2}
    else:
        info = None

    return info, gradient_calls_per_step, _acc_prob, final_state
