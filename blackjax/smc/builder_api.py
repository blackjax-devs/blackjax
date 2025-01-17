import functools

from blackjax import SamplingAlgorithm
from blackjax.smc import adaptive_tempered, from_mcmc, resampling, solver, tempered
from blackjax.smc.base import update_and_take_last
from blackjax.smc.waste_free import waste_free_smc


class SMCSamplerBuilder:
    def __init__(self):
        self.step_structure = None
        self.update_strategy = None

    def inner_kernel(self, init, step, inner_kernel_params):
        self._inner_kernel_step = step
        self._inner_kernel_init = init
        self._inner_kernel_params = inner_kernel_params
        print(inner_kernel_params)
        return self

    def tempering_from_sequence(self, logprior_fn, loglikelihood_fn):
        self.step_structure = "tempering"
        self.logprior_fn = logprior_fn
        self.loglikelihood_fn = loglikelihood_fn
        return self

    def adaptive_tempering(
        self, target_ess, logprior_fn, loglikelihood_fn, root_solver=solver.dichotomy
    ):
        self.step_structure = "adaptive_tempering"
        self.logprior_fn = logprior_fn
        self.loglikelihood_fn = loglikelihood_fn
        self.root_solver = root_solver
        self.target_ess = target_ess
        return self

    def partial_posteriors(self):
        pass

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

    def pretune(self):
        pass

    def inner_kernel_tuning(self):
        pass

    def build(self, resampling_fn=resampling.systematic):
        if self.update_strategy is None:
            raise ValueError(
                "You must choose an update strategy, either waste_free() or mutate_and_take_last()"
            )
        if self.step_structure == "adaptive_tempering":
            mutation_step = from_mcmc.build_kernel(
                self._inner_kernel_step,
                self._inner_kernel_init,
                resampling_fn,
                self._inner_kernel_params,
                self.update_strategy,
            )

            tempered_kernel = tempered.build_kernel(
                self.logprior_fn, self.loglikelihood_fn, mutation_step
            )
            return SamplingAlgorithm(
                tempered.init,
                adaptive_tempered.build_kernel(
                    self.loglikelihood_fn,
                    self.target_ess,
                    self.root_solver,
                    tempered_kernel,
                ),
            )
        elif self.step_structure == "tempering":
            mutation_step = from_mcmc.build_kernel(
                self._inner_kernel_step,
                self._inner_kernel_init,
                resampling_fn,
                self._inner_kernel_params,
                self.update_strategy,
            )

            tempered_kernel = tempered.build_kernel(
                self.logprior_fn, self.loglikelihood_fn, mutation_step
            )
            return SamplingAlgorithm(tempered.init, tempered_kernel)
        else:
            raise NotImplementedError(
                "The SMCBuilder API supports three ways of structuring SMC"
                "steps: adaptive tempering, fixed-sequence tempering or "
                "partial posteriors (data tempering). "
            )
