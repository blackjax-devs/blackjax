import dataclasses
from typing import Callable, Union

from blackjax._version import __version__

from .adaptation.chees_adaptation import chees_adaptation
from .adaptation.mclmc_adaptation import mclmc_find_L_and_step_size
from .adaptation.meads_adaptation import meads_adaptation
from .adaptation.pathfinder_adaptation import pathfinder_adaptation
from .adaptation.window_adaptation import window_adaptation
from .base import SamplingAlgorithm
from .diagnostics import effective_sample_size as ess
from .diagnostics import potential_scale_reduction as rhat
from .mcmc.barker import barker_proposal
from .mcmc.dynamic_hmc import dynamic_hmc
from .mcmc.elliptical_slice import elliptical_slice
from .mcmc.ghmc import ghmc
from .mcmc.mala import mala
from .mcmc.marginal_latent_gaussian import mgrad_gaussian
from .mcmc.mclmc import mclmc
from .mcmc.periodic_orbital import orbital_hmc
from .mcmc.random_walk import additive_step_random_walk, irmh, rmh
from .mcmc.rmhmc import rmhmc
from .optimizers import dual_averaging, lbfgs
from .sgmcmc.csgld import csgld
from .sgmcmc.sghmc import sghmc
from .sgmcmc.sgld import sgld
from .sgmcmc.sgnht import sgnht
from .smc.adaptive_tempered import adaptive_tempered_smc
from .smc.inner_kernel_tuning import inner_kernel_tuning
from .smc.tempered import tempered_smc
from .vi.meanfield_vi import meanfield_vi
from .vi.pathfinder import pathfinder
from .vi.schrodinger_follmer import schrodinger_follmer
from .vi.svgd import svgd
from .mcmc import hmc as _hmc
from .mcmc import nuts as _nuts


@dataclasses.dataclass
class SamplingAlgorithmFactory:
    differentiable_callable: Callable
    init: Callable
    build_kernel: Callable

    def __call__(self, *args, **kwargs) -> SamplingAlgorithm:
        return self.differentiable_callable(*args, **kwargs)


hmc = SamplingAlgorithmFactory(_hmc.as_sampling_algorithm, _hmc.init, _hmc.build_kernel)
nuts = SamplingAlgorithmFactory(_nuts.as_sampling_algorithm, _nuts.init, _nuts.build_kernel)

hmc_family = [hmc, nuts]

__all__ = [
    "__version__",
    "dual_averaging",  # optimizers
    "lbfgs",
    "dynamic_hmc",
    "rmhmc",
    "mala",
    "mgrad_gaussian",
    "nuts",
    "orbital_hmc",
    "additive_step_random_walk",
    "rmh",
    "irmh",
    "mclmc",
    "elliptical_slice",
    "ghmc",
    "barker_proposal",
    "sgld",  # stochastic gradient mcmc
    "sghmc",
    "sgnht",
    "csgld",
    "window_adaptation",  # mcmc adaptation
    "meads_adaptation",
    "chees_adaptation",
    "pathfinder_adaptation",
    "mclmc_find_L_and_step_size",  # mclmc adaptation
    "adaptive_tempered_smc",  # smc
    "tempered_smc",
    "inner_kernel_tuning",
    "meanfield_vi",  # variational inference
    "pathfinder",
    "schrodinger_follmer",
    "svgd",
    "ess",  # diagnostics
    "rhat",
]
