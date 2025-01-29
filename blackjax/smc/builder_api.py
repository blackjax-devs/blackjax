import functools

import blackjax.smc.adaptive_tempered
from blackjax import SamplingAlgorithm, inner_kernel_tuning
from blackjax.smc import (
    adaptive_tempered,
    from_mcmc,
    partial_posteriors_path,
    pretuning,
    resampling,
    solver,
    tempered,
)
from blackjax.smc.base import update_and_take_last
from blackjax.smc.from_mcmc import build_kernel as smc_from_mcmc
from blackjax.smc.pretuning import build_kernel
from blackjax.smc.waste_free import waste_free_smc


class SMCSamplerBuilder:
    """
    SMC is a meta-algorithm in the sense that can be constructed in
    different ways by composing inner components. The aim of this API
    is to foster modifying such compositions easily.
    """

    def __init__(self, resampling_fn=resampling.systematic):
        self.step_structure = None
        self.update_strategy = None
        self.mcmc_parameter_update_fn = None
        self.pretune_fn = None
        self.resampling_fn = resampling_fn

    # Different ways of building the sequence of distributions
    def adaptive_tempering_sequence(
            self, target_ess, logprior_fn, loglikelihood_fn, root_solver=solver.dichotomy
    ):
        self.step_structure = "adaptive_tempering"
        self.step_structure_algorithm = blackjax.smc.adaptive_tempered.build_kernel
        self.step_structure_init = blackjax.smc.tempered.init
        self.logprior_fn = logprior_fn
        self.loglikelihood_fn = loglikelihood_fn
        self.root_solver = root_solver
        self.target_ess = target_ess
        return self

    def tempering_from_sequence(self, logprior_fn, loglikelihood_fn):
        self.step_structure = "tempering"
        self.step_structure_algorithm = blackjax.smc.tempered.build_kernel
        self.step_structure_init = blackjax.smc.tempered.init
        self.logprior_fn = logprior_fn
        self.loglikelihood_fn = loglikelihood_fn
        return self

    def partial_posteriors_sequence(self, partial_logposterior_factory):
        self.step_structure = "partial_posteriors"
        self.step_structure_algorithm = (
            blackjax.smc.partial_posteriors_path.build_kernel
        )
        self.partial_logposterior_factory = partial_logposterior_factory
        return self

    # Inner kernel construction
    def inner_kernel(self, init, step, inner_kernel_params):
        self._inner_kernel_step = step
        self._inner_kernel_init = init
        self._inner_kernel_params = inner_kernel_params
        return self

    # Ways of updating the particles
    def mutate_waste_free(self, n_particles, p):
        if self.update_strategy is not None:
            raise ValueError("Can't use two update strategies at the same time")
        self.update_strategy = waste_free_smc(n_particles, p)
        return self

    def mutate_and_take_last(self, mcmc_steps):
        if self.update_strategy is not None:
            raise ValueError("Can't use two update strategies at the same time")
        self.update_strategy = functools.partial(
            update_and_take_last, num_mcmc_steps=mcmc_steps
        )
        return self

    # Ways of tuning or pre-tuning the inner kernel parameters
    def with_inner_kernel_tuning(self, mcmc_parameter_update_fn):
        if self.mcmc_parameter_update_fn is not None:
            raise ValueError(
                "Can't call inner_kernel_tuning twice, consider merging all calls into one"
            )

        self.mcmc_parameter_update_fn = mcmc_parameter_update_fn
        return self

    def with_pretuning(self, pretune_fn):
        if self.pretune_fn is not None:
            raise ValueError(
                "Can't call pretune twice, consider merging all calls into one"
            )
        self.pretune_fn = pretune_fn
        return self

    def build(self):
        if self.update_strategy is None:
            raise ValueError(
                "You must choose an update strategy, either waste_free() or mutate_and_take_last()"
            )

        if self.step_structure is None:
            raise ValueError(
                "You must either call adaptive_tempering(), "
                "tempering_sequence()"
                " or partial_posteriors_path()"
            )

        if self.step_structure == "adaptive_tempering":
            init, step = self._adaptive_tempered_from_parameters()
        elif self.step_structure == "tempering":
            init, step = self._tempered_from_parameters()
        elif self.step_structure == "partial_posteriors":
            init, step = self._partial_posterior_from_parameters()
        else:
            raise NotImplementedError(
                "The SMCBuilder API supports three ways of structuring SMC"
                "steps: adaptive tempering, fixed-sequence tempering or "
                "partial posteriors (data tempering). "
            )

        if self.mcmc_parameter_update_fn is None and self.pretune_fn is None:
            # no tuning or pretuning is used
            return SamplingAlgorithm(init, step(self._inner_kernel_params))

        if self.mcmc_parameter_update_fn is not None and self.pretune_fn is None:
            # only tuning
            def new_init(position):
                return inner_kernel_tuning.init(
                    init, position, self._inner_kernel_params
                )

            return SamplingAlgorithm(
                new_init,
                inner_kernel_tuning.build_kernel(step, self.mcmc_parameter_update_fn),
            )
        if self.mcmc_parameter_update_fn is None and self.pretune_fn is not None:
            # only pretune
            return self._build_pretuning()

        # Both Pretune and Tune
        raise NotImplementedError("Tuning and pretuning used together hasn't been implemented yet")

    def _adaptive_tempered_from_parameters(self):
        def from_parameteres(inner_kernel_params):
            mutation_step = from_mcmc.build_kernel(
                self._inner_kernel_step,
                self._inner_kernel_init,
                self.resampling_fn,
                inner_kernel_params,
                self.update_strategy,
            )

            tempered_kernel = tempered.build_kernel(
                self.logprior_fn, self.loglikelihood_fn, mutation_step
            )

            step = adaptive_tempered.build_kernel(
                self.loglikelihood_fn,
                self.target_ess,
                self.root_solver,
                tempered_kernel,
            )
            return step

        init = tempered.init
        return init, from_parameteres

    def _tempered_from_parameters(self):
        def from_parameteres(inner_kernel_params):
            mutation_step = from_mcmc.build_kernel(
                self._inner_kernel_step,
                self._inner_kernel_init,
                self.resampling_fn,
                inner_kernel_params,
                self.update_strategy,
            )

            tempered_kernel = tempered.build_kernel(
                self.logprior_fn, self.loglikelihood_fn, mutation_step
            )
            return tempered_kernel

        init = tempered.init
        return init, from_parameteres

    def _partial_posterior_from_parameters(self):
        def from_parameters(params):
            update_particles = from_mcmc.build_kernel(
                self._inner_kernel_step,
                self._inner_kernel_init,
                self.resampling_fn,
                params,
                self.update_strategy,
            )
            return partial_posteriors_path.build_kernel(
                self.partial_logposterior_factory, update_particles
            )

        return (partial_posteriors_path.init, from_parameters)

    def _build_pretuning(self):
        def delegate(rng_key, state, logposterior_fn, log_weights_fn, mcmc_parameteres):
            return smc_from_mcmc(
                self._inner_kernel_step,
                self._inner_kernel_init,
                self.resampling_fn,
                mcmc_parameteres,
                self.update_strategy,
            )(
                rng_key,
                state,
                logposterior_fn,
                log_weights_fn,
            )

        if self.step_structure == "adaptive_tempering":

            def smc_algorithm_from_params(mcmc_parameters, pretuned_step):
                tempered_kernel = blackjax.smc.tempered.build_kernel(
                    logprior_fn=self.logprior_fn,
                    loglikelihood_fn=self.loglikelihood_fn,
                    update_particles=functools.partial(
                        pretuned_step, mcmc_parameters=mcmc_parameters
                    ),
                )

                return blackjax.smc.adaptive_tempered.build_kernel(
                    loglikelihood_fn=self.loglikelihood_fn,
                    target_ess=self.target_ess,
                    root_solver=self.root_solver,
                    tempered_kernel=tempered_kernel,
                )

        elif self.step_structure == "tempering":

            def smc_algorithm_from_params(mcmc_parameters, pretuned_step):
                return blackjax.smc.tempered.build_kernel(
                    logprior_fn=self.logprior_fn,
                    loglikelihood_fn=self.loglikelihood_fn,
                    update_particles=functools.partial(
                        pretuned_step, mcmc_parameters=mcmc_parameters
                    ),
                )

        kernel = build_kernel(smc_algorithm_from_params, self.pretune_fn, delegate)

        def init_fn(position, rng_key=None):
            del rng_key
            return pretuning.init(
                blackjax.smc.tempered.init, position, self._inner_kernel_params
            )

        return SamplingAlgorithm(init_fn, kernel)

    def _tune_and_pretune(self):
        def pt(
                logprior_fn,
                loglikelihood_fn,
                mcmc_step_fn,
                mcmc_init_fn,
                mcmc_parameters,
                resampling_fn,
                num_mcmc_steps,
                initial_parameter_value,
                target_ess,
        ):
            return blackjax.pretuning(
                blackjax.adaptive_tempered_smc,
                logprior_fn,
                loglikelihood_fn,
                mcmc_step_fn,
                mcmc_init_fn,
                resampling_fn,
                num_mcmc_steps,
                target_ess=self.target_ess,
                pretune_fn=self.pretune,
            )

        kernel = blackjax.smc.inner_kernel_tuning.build_kernel(
            pt,
            self.logprior_fn,
            self.loglikelihood_fn,
            self._inner_kernel_step,
            self._inner_kernel_init,
            self.resampling_fn,
            self.mcmc_parameter_update_fn,
            initial_parameter_value=self._inner_kernel_params,
            target_ess=self.target_ess,
            smc_returns_state_with_parameter_override=True,
        )

        def init2(position):
            return blackjax.smc.inner_kernel_tuning.init(
                blackjax.adaptive_tempered_smc.init, position, initial_parameters
            )

        return SamplingAlgorithm(init2, kernel)
