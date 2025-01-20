import functools

from blackjax import SamplingAlgorithm, inner_kernel_tuning
from blackjax.smc import adaptive_tempered, from_mcmc, resampling, solver, tempered, partial_posteriors_path
from blackjax.smc.base import update_and_take_last
from blackjax.smc.waste_free import waste_free_smc


class SMCSamplerBuilder:
    def __init__(self):
        self.step_structure = None
        self.update_strategy = None
        self.mcmc_parameter_update_fn = None

    # Different ways of building the sequence of distributions
    def adaptive_tempering(
            self, target_ess, logprior_fn, loglikelihood_fn, root_solver=solver.dichotomy
    ):
        self.step_structure = "adaptive_tempering"
        self.logprior_fn = logprior_fn
        self.loglikelihood_fn = loglikelihood_fn
        self.root_solver = root_solver
        self.target_ess = target_ess
        return self

    def tempering_from_sequence(self, logprior_fn, loglikelihood_fn):
        self.step_structure = "tempering"
        self.logprior_fn = logprior_fn
        self.loglikelihood_fn = loglikelihood_fn
        return self

    def partial_posteriors(self, partial_logposterior_factory):
        self.step_structure = "partial_posteriors"
        self.partial_logposterior_factory = partial_logposterior_factory
        return self

    # Inner kernel construction
    def inner_kernel(self, init, step, inner_kernel_params):
        self._inner_kernel_step = step
        self._inner_kernel_init = init
        self._inner_kernel_params = inner_kernel_params
        return self

    # Ways of updating the particles
    def waste_free(self, n_particles, p):
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
            raise ValueError("Can't call inner_kernel_tuning twice, consider merging all calls into one")

        self.mcmc_parameter_update_fn = mcmc_parameter_update_fn
        return self

    def pretune(self):
        pass

    def build(self, resampling_fn=resampling.systematic):
        self.resampling_fn = resampling_fn
        if self.update_strategy is None:
            raise ValueError(
                "You must choose an update strategy, either waste_free() or mutate_and_take_last()"
            )

        if self.step_structure is None:
            raise ValueError(
                "You must either call adaptive_tempering(), "
                "tempering_sequence()"
                " or partial_posteriors_path()")

        if self.step_structure == "adaptive_tempering":
            init, step = self._build_adaptive_tempered()

        elif self.step_structure == "tempering":
            init, step = self._build_tempered_from_parameters()
        elif self.step_structure == "partial_posteriors":
            init, step = self._build_partial_posterior_from_parameters()

        else:
            raise NotImplementedError(
                "The SMCBuilder API supports three ways of structuring SMC"
                "steps: adaptive tempering, fixed-sequence tempering or "
                "partial posteriors (data tempering). "
            )

        if self.mcmc_parameter_update_fn is not None:
            def new_init(position):
                return inner_kernel_tuning.init(init, position, self._inner_kernel_params)

            return SamplingAlgorithm(new_init,
                                     inner_kernel_tuning.build_kernel(step, self.mcmc_parameter_update_fn))
        else:
            step = step(self._inner_kernel_params)
        return SamplingAlgorithm(init, step)

    def _build_adaptive_tempered(self):
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

    def _build_tempered_from_parameters(self):
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

    def _build_partial_posterior_from_parameters(self):
        def from_parameters(params):
            update_particles =  from_mcmc.build_kernel(
                    self._inner_kernel_step,
                    self._inner_kernel_init,
                    self.resampling_fn,
                    params,
                    self.update_strategy,
                )
            return partial_posteriors_path.build_kernel(self.partial_logposterior_factory, update_particles)

        return (partial_posteriors_path.init, from_parameters)


