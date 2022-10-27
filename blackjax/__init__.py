from .diagnostics import effective_sample_size as ess
from .diagnostics import potential_scale_reduction as rhat
from .kernels import (
    adaptive_tempered_smc,
    elliptical_slice,
    ghmc,
    hmc,
    irmh,
    mala,
    meads,
    mgrad_gaussian,
    nuts,
    orbital_hmc,
    pathfinder,
    pathfinder_adaptation,
    rmh,
    sghmc,
    sgld,
    tempered_smc,
    window_adaptation,
)
from .optimizers import dual_averaging, lbfgs

__all__ = [
    "dual_averaging",  # optimizers
    "lbfgs",
    "hmc",  # mcmc
    "mala",
    "mgrad_gaussian",
    "nuts",
    "orbital_hmc",
    "rmh",
    "irmh",
    "elliptical_slice",
    "ghmc",
    "meads",
    "sgld",  # stochastic gradient mcmc
    "sghmc",
    "window_adaptation",  # mcmc adaptation
    "pathfinder_adaptation",
    "adaptive_tempered_smc",  # smc
    "tempered_smc",
    "pathfinder",  # variational inference
    "ess",  # diagnostics
    "rhat",
]

from . import _version

__version__ = _version.get_versions()["version"]
