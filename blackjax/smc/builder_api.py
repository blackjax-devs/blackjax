import functools

from blackjax import SamplingAlgorithm
from blackjax.smc import adaptive_tempered, tempered, from_mcmc
from blackjax.smc.base import update_and_take_last


class SMCSamplerBuilder:
    def __init__(self):
        self.step_structure = None


    def inner_kernel(self, init, step, inner_kernel_params):
        self._inner_kernel_step = step
        self._inner_kernel_init = init
        self._inner_kernel_params = inner_kernel_params
        print(inner_kernel_params)
        return self
    def tempering_from_sequence(self):
        pass

    def adaptive_tempering(self, target_ess, solver, logprior_fn , loglikelihood_fn):
        self.step_structure = "adaptive_tempering"
        self.logprior_fn = logprior_fn
        self.loglikelihood_fn = loglikelihood_fn
        self.solver=solver
        self.target_ess = target_ess
        return self
    def partial_posteriors(self):
        pass


    def waste_free(self):
        pass

    def mutate_and_take_last(self, mcmc_steps):
        self.update_strategy = functools.partial(update_and_take_last, num_mcmc_steps=mcmc_steps)
        return self

    def pretune(self):
        pass

    def inner_kernel_tuning(self):
        pass

    def build(self, resampling):
        if self.step_structure == "adaptive_tempering":
            mutation_step = from_mcmc.build_kernel(
                self._inner_kernel_step,
                self._inner_kernel_init,
                resampling,
                self._inner_kernel_params,
                self.update_strategy,
            )

            tempered_kernel = tempered.build_kernel(self.logprior_fn,
                                                    self.loglikelihood_fn,
                                                    mutation_step
                                                    )
            return SamplingAlgorithm(tempered.init,
                                     adaptive_tempered.build_kernel(self.loglikelihood_fn,
                                                                    self.target_ess,
                                                                    self.solver,
                                                                    tempered_kernel))


        else:
            raise NotImplementedError
