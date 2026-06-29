blackjax.ns.utils
=================

.. py:module:: blackjax.ns.utils

.. autoapi-nested-parse::

   Utility functions for setting up and post-processing Nested Sampling runs.

   .. rubric:: References

   .. [1] Skilling, J. (2006). "Nested sampling for general Bayesian computation."
          Bayesian Analysis, 1(4), 833-859. https://doi.org/10.1214/06-BA127
   .. [2] Fowlie, A., Handley, W., & Su, L. (2021). "Nested sampling with plateaus."
          Monthly Notices of the Royal Astronomical Society, 503(1), 1199-1205.
          https://doi.org/10.1093/mnras/stab590



Functions
---------

.. autoapisummary::

   blackjax.ns.utils.log1mexp
   blackjax.ns.utils.compute_num_live
   blackjax.ns.utils.logX
   blackjax.ns.utils.log_weights
   blackjax.ns.utils.finalise
   blackjax.ns.utils.ess
   blackjax.ns.utils.sample
   blackjax.ns.utils.get_first_row
   blackjax.ns.utils.uniform_prior


Module Contents
---------------

.. py:function:: log1mexp(x: blackjax.types.Array) -> blackjax.types.Array

   Computes log(1 - exp(x)) in a numerically stable way.


.. py:function:: compute_num_live(info: blackjax.ns.base.NSInfo) -> blackjax.types.Array

   Compute the effective number of live points at each death contour (Fowlie, Handley & Su, 2021).

   When doing batch deletions, the jump in energy level can be smoothed by
   transforming 1 jump of size k into k jumps of size 1. This function computes
   the effective population size associated with this transformation.

   Expects the **complete finalised output** -- the dead points together with the
   final live particles (e.g. from :func:`finalise`): it relies on every
   particle's birth event being present. Called on a dead-only subset that omits
   the initial live particles' births, the live counts are wrong (1 instead of
   ``N`` for a standard run).

   :returns: An array where each element `num_live[j]` is the effective number of live
             points `m*_i` when the j-th particle (in the sorted list of dead particles)
             was considered "dead".
   :rtype: Array


.. py:function:: logX(rng_key: blackjax.types.PRNGKey, dead_info: blackjax.ns.base.NSInfo, shape: int = 100) -> tuple[blackjax.types.Array, blackjax.types.Array]

   Simulate the stochastic evolution of log prior volumes (Skilling, 2006).

   Wraps the effective population size in `compute_num_live`, along with stochastic
   simulation of the log prior shrinkage associated with each deleted particle.


   :param rng_key: A JAX PRNG key for generating uniform random variates.
   :param dead_info: An `NSInfo` object (or compatible PyTree) containing `loglikelihood_birth`
                     and `loglikelihood` for all dead particles accumulated during an NS run.
                     It's assumed these particles are already sorted by their death log-likelihood.
   :param shape: The shape of Monte Carlo samples to generate for the stochastic
                 log-volume sequence. Each sample represents one possible path of
                 volume shrinkage. Default is 100.

   :returns:

             - `logX_cumulative`: An array of shape `(num_dead_particles, shape)`
               containing `shape` simulated sequences of cumulative log prior volumes `log(X_i)`.
             - `log_dX_elements`: An array of shape `(num_dead_particles, shape)`
               containing `shape` simulated sequences of log prior volume elements `log(dX_i)`.
               `dX_i` is the trapezoidal volume element `(X_{i-1} - X_{i+1}) / 2`.
   :rtype: tuple[Array, Array]


.. py:function:: log_weights(rng_key: blackjax.types.PRNGKey, dead_info: blackjax.ns.base.NSInfo, shape: int = 100, beta: float = 1.0) -> blackjax.types.Array

   Calculate the log importance weights for Nested Sampling results.

   :param rng_key: A JAX PRNG key for simulating `log(dX_i)`.
   :param dead_info: An `NSInfo` object (or compatible PyTree) containing `loglikelihood_birth`
                     and `loglikelihood` for all dead particles.
   :param shape: The shape of Monte Carlo samples to use for simulating `log(dX_i)`.
                 Default is 100.
   :param beta: The inverse temperature. Typically 1.0 for standard evidence calculation.
                Allows for reweighting to different temperatures.

   :returns: An array of log importance weights, shape `(num_dead_particles, *shape)`.
             The original order of particles in `dead_info` is preserved.
   :rtype: Array


.. py:function:: finalise(live: blackjax.ns.base.NSState, dead: list[blackjax.ns.base.NSInfo], update_info: bool = True) -> blackjax.ns.base.NSInfo

   Combines the history of dead particle information with the final live points.

   :param live: The final `NSState` of the Nested Sampler, containing the live particles.
   :param dead: A list of `NSInfo` objects, where each object contains information
                about the particles that "died" at one step of the NS algorithm.
   :param update_info: Whether to concatenate the `update_info` from each element of `dead`.
                       If False, the returned `update_info` is None. Default is True.

   :returns: A single `NSInfo` whose `particles` field concatenates all dead
             particles with the final live particles. When ``update_info=True`` the
             `update_info` field concatenates the `update_info` from each element of
             `dead` only -- no entry is added for the final live points, so it is
             shorter than `particles` by the number of live points. When
             ``update_info=False`` the `update_info` field is None.
   :rtype: NSInfo


.. py:function:: ess(rng_key: blackjax.types.PRNGKey, dead: blackjax.ns.base.NSInfo) -> blackjax.types.Array

   Computes the Effective Sample Size (ESS) from log-weights.

   :param rng_key: A JAX PRNG key, used by `log_weights`.
   :param dead: An `NSInfo` object containing the full set of dead (and final live)
                particles, typically the output of `finalise`.

   :returns: The mean Effective Sample Size, a scalar float.
   :rtype: Array


.. py:function:: sample(rng_key: blackjax.types.PRNGKey, dead: blackjax.ns.base.NSInfo, shape: int = 1000) -> blackjax.types.ArrayTree

   Resamples particles according to their importance weights.

   :param rng_key: A JAX PRNG key, used by `log_weights` and for resampling.
   :param dead: An `NSInfo` object containing the full set of dead (and final live)
                particles, typically the output of `finalise`.
   :param shape: The number of resampled particles to draw. Default is 1000.

   :returns: A PyTree of resampled particles, where each leaf has `shape`.
   :rtype: ArrayTree


.. py:function:: get_first_row(x: blackjax.types.ArrayTree) -> blackjax.types.ArrayTree

   Extracts the first "row" (element along the leading axis) of each leaf in a PyTree.

   This is typically used to get a single particle's structure or values from
   a PyTree representing a collection of particles, where the leading dimension
   of each leaf array corresponds to the particle index.

   :param x: A PyTree of arrays, where each leaf array has a leading dimension.

   :returns: A PyTree with the same structure as `x`, but where each leaf is the
             first slice `leaf[0]` of the corresponding leaf in `x`.
   :rtype: ArrayTree


.. py:function:: uniform_prior(rng_key: blackjax.types.PRNGKey, num_live: int, bounds: dict[str, tuple[float, float]]) -> tuple[blackjax.types.ArrayTree, Callable]

   Sample initial particles and build a log-prior for a box-uniform prior.

   :param rng_key: A JAX PRNG key for random number generation.
   :param num_live: The number of live particles to sample.
   :param bounds: A dictionary mapping parameter names to their bounds (tuples of min and max).
                  Each parameter will be sampled uniformly within these bounds.
                  Example: {'param1': (0.0, 1.0), 'param2': (-5.0, 5.0)}

   :returns:

             - `particles`: A PyTree of sampled parameters, where each leaf has shape `(num_live,)`.
             - `logprior_fn`: A function that computes the log-prior probability
               for a given set of parameters.
   :rtype: tuple


