blackjax.vi.pathfinder
======================

.. py:module:: blackjax.vi.pathfinder


Classes
-------

.. autoapisummary::

   blackjax.vi.pathfinder.PathfinderState


Functions
---------

.. autoapisummary::

   blackjax.vi.pathfinder.approximate
   blackjax.vi.pathfinder.sample
   blackjax.vi.pathfinder.as_top_level_api


Module Contents
---------------

.. py:class:: PathfinderState



   State of the Pathfinder algorithm.

   Pathfinder locates normal approximations to the target density along a
   quasi-Newton optimization path, with local covariance estimated using
   the inverse Hessian estimates produced by the L-BFGS optimizer.
   PathfinderState stores for an iteration of the L-BFGS optimizer the
   resulting ELBO and all factors needed to sample from the approximated
   target density.

   position:
       position
   grad_position:
       gradient of target distribution wrt position
   alpha, beta, gamma:
       factored representation of the inverse hessian
   elbo:
       ELBO of approximation wrt target distribution



   .. py:attribute:: elbo
      :type:  blackjax.types.Array


   .. py:attribute:: position
      :type:  blackjax.types.ArrayTree


   .. py:attribute:: grad_position
      :type:  blackjax.types.ArrayTree


   .. py:attribute:: alpha
      :type:  blackjax.types.Array


   .. py:attribute:: beta
      :type:  blackjax.types.Array


   .. py:attribute:: gamma
      :type:  blackjax.types.Array


.. py:function:: approximate(rng_key: blackjax.types.PRNGKey, logdensity_fn: Callable, initial_position: blackjax.types.ArrayLikeTree, num_samples: int = 200, *, maxiter=30, maxcor=10, maxls=1000, gtol=1e-08, ftol=1e-05, **lbfgs_kwargs) -> tuple[PathfinderState, PathfinderInfo]

   Pathfinder variational inference algorithm.

   Pathfinder locates normal approximations to the target density along a
   quasi-Newton optimization path, with local covariance estimated using
   the inverse Hessian estimates produced by the L-BFGS optimizer.

   Function implements the algorithm 3 in :cite:p:`zhang2022pathfinder`:

   :param rng_key: PRNG key
   :param logdensity_fn: (un-normalized) log densify function of target distribution to take
                         approximate samples from
   :param initial_position: starting point of the L-BFGS optimization routine
   :param num_samples: number of samples to draw to estimate ELBO
   :param maxiter: Maximum number of iterations of the L-BFGS algorithm.
   :param maxcor: Maximum number of metric corrections of the L-BFGS algorithm ("history
                  size")
   :param ftol: The L-BFGS algorithm terminates the minimization when `(f_k - f_{k+1}) <
                ftol`
   :param gtol: The L-BFGS algorithm terminates the minimization when `|g_k|_norm < gtol`
   :param maxls: The maximum number of line search steps (per iteration) for the L-BFGS
                 algorithm
   :param \*\*lbfgs_kwargs: other keyword arguments passed to `jaxopt.LBFGS`.

   :returns: * *A PathfinderState with information on the iteration in the optimization path*
             * *whose approximate samples yields the highest ELBO, and PathfinderInfo that*
             * *contains all the states traversed.*


.. py:function:: sample(rng_key: blackjax.types.PRNGKey, state: PathfinderState, num_samples: int | tuple[] | tuple[int] = ()) -> blackjax.types.ArrayTree

   Draw from the Pathfinder approximation of the target distribution.

   :param rng_key: PRNG key
   :param state: PathfinderState containing information for sampling
   :param num_samples: Number of samples to draw

   :rtype: Samples drawn from the approximate Pathfinder distribution


.. py:function:: as_top_level_api(logdensity_fn: Callable) -> blackjax.base.VIAlgorithm

   Implements the (basic) user interface for the pathfinder kernel.

   Pathfinder locates normal approximations to the target density along a
   quasi-Newton optimization path, with local covariance estimated using
   the inverse Hessian estimates produced by the L-BFGS optimizer.
   Pathfinder returns draws from the approximation with the lowest estimated
   Kullback-Leibler (KL) divergence to the true posterior.

   As Pathfinder is a one-shot algorithm, the returned ``VIAlgorithm.step``
   is a no-op; all computation happens inside ``VIAlgorithm.init``.

   :param logdensity_fn: A function that represents the log-density of the model we want
                         to sample from.

   :rtype: A ``VIAlgorithm``.


