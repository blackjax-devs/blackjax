import dataclasses
from typing import Callable

from blackjax._version import __version__

#from .adaptation.chees_adaptation import chees_adaptation
#from .adaptation.mclmc_adaptation import mclmc_find_L_and_step_size
#from .adaptation.meads_adaptation import meads_adaptation
#from .adaptation.pathfinder_adaptation import pathfinder_adaptation
#from .adaptation.window_adaptation import window_adaptation
from .base import SamplingAlgorithm, VIAlgorithm
#from .diagnostics import effective_sample_size as ess
#from .diagnostics import potential_scale_reduction as rhat
#from .mcmc import barker
#from .mcmc import dynamic_hmc as _dynamic_hmc
#from .mcmc import elliptical_slice as _elliptical_slice
#from .mcmc import ghmc as _ghmc
from .mcmc import hmc as _hmc
#from .mcmc import mala as _mala
#from .mcmc import marginal_latent_gaussian
#from .mcmc import mclmc as _mclmc
#from .mcmc import nuts as _nuts
#from .mcmc import periodic_orbital, random_walk
#from .mcmc import rmhmc as _rmhmc
#from .mcmc.random_walk import additive_step_random_walk as _additive_step_random_walk
#from .mcmc.random_walk import (
#    irmh_as_top_level_api,
#    normal_random_walk,
#    rmh_as_top_level_api,
#)
#from .optimizers import dual_averaging, lbfgs
#from .sgmcmc import csgld as _csgld
#from .sgmcmc import sghmc as _sghmc
#from .sgmcmc import sgld as _sgld
#from .sgmcmc import sgnht as _sgnht
from .smc import adaptive_tempered
from .smc import inner_kernel_tuning as _inner_kernel_tuning
from .smc import tempered
#from .vi import meanfield_vi as _meanfield_vi
#from .vi import pathfinder as _pathfinder
#from .vi import schrodinger_follmer as _schrodinger_follmer
#from .vi import svgd as _svgd
#from .vi.pathfinder import PathFinderAlgorithm

"""
The above three classes exist as a backwards compatible way of exposing both the high level, differentiable
factory and the low level components, which may not be differentiable. Moreover, this design allows for the lower
level to be mostly functional programming in nature and reducing boilerplate code.
"""


@dataclasses.dataclass
class GenerateSamplingAPI:
    differentiable: Callable
    init: Callable
    build_kernel: Callable

    def __call__(self, *args, **kwargs) -> SamplingAlgorithm:
        return self.differentiable(*args, **kwargs)

    def register_factory(self, name, callable):
        setattr(self, name, callable)


@dataclasses.dataclass
class GenerateVariationalAPI:
    differentiable: Callable
    init: Callable
    step: Callable
    sample: Callable

    def __call__(self, *args, **kwargs) -> VIAlgorithm:
        return self.differentiable(*args, **kwargs)


##class GeneratePathfinderAPI:
 #   differentiable: Callable
 ##   approximate: Callable
  #  sample: Callable

 #   def __call__(self, *args, **kwargs) -> PathFinderAlgorithm:
        #return self.differentiable(*args, **kwargs)


def generate_top_level_api_from(module):
    return GenerateSamplingAPI(
        module.as_top_level_api, module.init, module.build_kernel
    )


# MCMC
hmc = generate_top_level_api_from(_hmc)

# SMC
adaptive_tempered_smc = generate_top_level_api_from(adaptive_tempered)
tempered_smc = generate_top_level_api_from(tempered)
inner_kernel_tuning = generate_top_level_api_from(_inner_kernel_tuning)

smc_family = [tempered_smc, adaptive_tempered_smc]
"Step_fn returning state has a .particles attribute"

# variational inference
