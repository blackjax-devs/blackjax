blackjax.vi.multipathfinder
===========================

.. py:module:: blackjax.vi.multipathfinder


Classes
-------

.. autoapisummary::

   blackjax.vi.multipathfinder.MultipathfinderState


Functions
---------

.. autoapisummary::

   blackjax.vi.multipathfinder.multi_approximate
   blackjax.vi.multipathfinder.psis_weights
   blackjax.vi.multipathfinder.as_top_level_api


Module Contents
---------------

.. py:class:: MultipathfinderState



   State returned by multi-path Pathfinder.

   path_states
       One ``PathfinderState`` per independent L-BFGS run.
   samples
       Approximate posterior samples drawn from each path's best
       approximation, shape ``(n_paths, num_samples, ...)``.
   logp
       Log target density evaluated at the per-path samples,
       shape ``(n_paths, num_samples)``.
   logq
       Log approximation density at the per-path samples,
       shape ``(n_paths, num_samples)``.


   .. py:attribute:: path_states
      :type:  blackjax.vi.pathfinder.PathfinderState


   .. py:attribute:: samples
      :type:  blackjax.types.ArrayTree


   .. py:attribute:: logp
      :type:  blackjax.types.Array


   .. py:attribute:: logq
      :type:  blackjax.types.Array


.. py:function:: multi_approximate(rng_key: blackjax.types.PRNGKey, logdensity_fn: Callable, initial_positions: blackjax.types.ArrayLikeTree, num_samples: int = 200, *, maxiter: int = 30, maxcor: int = 10, maxls: int = 1000, gtol: float = 1e-08, ftol: float = 1e-05) -> tuple[MultipathfinderState, blackjax.vi.pathfinder.PathfinderInfo]

   Multi-path Pathfinder variational inference.

   Runs single-path Pathfinder independently from each of the supplied
   initial positions (Algorithm 2 in :cite:p:`zhang2022pathfinder`), then
   collects the per-path samples and log densities needed for importance
   weighting via :func:`psis_weights`.

   :param rng_key: PRNG key.
   :param logdensity_fn: (Un-normalised) log density of the target distribution.
   :param initial_positions: Starting points for each L-BFGS run.  Must be a pytree where the
                             leading axis indexes the ``n_paths`` paths; e.g. an array of shape
                             ``(n_paths, d)``.
   :param num_samples: Number of samples drawn per path to estimate ELBO and log weights.
   :param maxiter: Maximum L-BFGS iterations per path.
   :param maxcor: L-BFGS history size.
   :param maxls: Maximum line-search steps per iteration.
   :param gtol: Gradient norm convergence tolerance.
   :param ftol: Function value convergence tolerance.

   :returns: * A ``MultipathfinderState`` (all path states, per-path samples, and log densities)
             * and a ``PathfinderInfo`` wrapping all per-path ``PathfinderState`` objects.


.. py:function:: psis_weights(state: MultipathfinderState) -> tuple[blackjax.types.Array, blackjax.types.Array]

   Compute Pareto-Smoothed Importance Sampling (PSIS) weights.

   Thin wrapper around :func:`blackjax.util.psis_weights` that extracts the
   log importance ratios from a :class:`MultipathfinderState`.

   :param state: Output of :func:`multi_approximate`.

   :returns: * *log_weights* -- Normalised log importance weights, shape ``(n_paths * num_samples,)``.
             * *pareto_k* -- Pareto shape parameter estimate (scalar ``Array``).  Values below 0.5
               indicate reliable importance sampling; above 0.7 may indicate
               unreliable estimates.


.. py:function:: as_top_level_api(logdensity_fn: Callable) -> blackjax.base.VIAlgorithm

   High-level multi-path Pathfinder interface.

   Returns a ``VIAlgorithm`` whose ``init`` runs multi-path Pathfinder and
   whose ``sample`` draws importance-resampled approximate posterior samples
   using PSIS weights.

   :param logdensity_fn: (Un-normalised) log density of the target distribution.

   :rtype: A ``VIAlgorithm``.


