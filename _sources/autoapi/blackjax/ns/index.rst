blackjax.ns
===========

.. py:module:: blackjax.ns

.. autoapi-nested-parse::

   Nested sampling algorithms.

   Nested sampling is a Monte Carlo method for Bayesian computation, used for
   evidence (marginal likelihood) estimation and posterior sampling.

   Available modules:

   - `base`: Core components for Nested Sampling.
   - `adaptive`: Adaptive nested sampling combining SMC-style adaptive tempering
     with per-step inner-kernel parameter tuning and evidence tracking.
   - `nss`: Nested slice sampling, with hit-and-run (``build_kernel``) or
     slice-within-Gibbs (``build_swig_kernel``) inner kernels.
   - `integrator`: NSIntegrator for tracking evidence integration.
   - `utils`: Utility functions for processing nested sampling results.
   - `from_mcmc`: Utilities to build nested sampling algorithms from MCMC kernels.



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/blackjax/ns/adaptive/index
   /autoapi/blackjax/ns/base/index
   /autoapi/blackjax/ns/from_mcmc/index
   /autoapi/blackjax/ns/integrator/index
   /autoapi/blackjax/ns/nss/index
   /autoapi/blackjax/ns/utils/index


