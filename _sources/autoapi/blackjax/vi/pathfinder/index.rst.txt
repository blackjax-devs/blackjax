:py:mod:`blackjax.vi.pathfinder`
================================

.. py:module:: blackjax.vi.pathfinder


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   blackjax.vi.pathfinder.PathfinderState



Functions
~~~~~~~~~

.. autoapisummary::

   blackjax.vi.pathfinder.approximate
   blackjax.vi.pathfinder.sample



.. py:class:: PathfinderState



   State of the Pathfinder algorithm.

   Pathfinder locates normal approximations to the target density along a
   quasi-Newton optimization path, with local covariance estimated using
   the inverse Hessian estimates produced by the L-BFGS optimizer.
   PathfinderState stores for an interation fo the L-BFGS optimizer the
   resulting ELBO and all factors needed to sample from the approximated
   target density.

   position:
       position
   grad_position:
       gradient of target distribution wrt position
   alpha, beta, gamma:
       factored rappresentation of the inverse hessian
   elbo:
       ELBO of approximation wrt target distribution


   .. py:attribute:: elbo
      :type: blackjax.types.Array

      

   .. py:attribute:: position
      :type: blackjax.types.PyTree

      

   .. py:attribute:: grad_position
      :type: blackjax.types.PyTree

      

   .. py:attribute:: alpha
      :type: blackjax.types.Array

      

   .. py:attribute:: beta
      :type: blackjax.types.Array

      

   .. py:attribute:: gamma
      :type: blackjax.types.Array

      


.. py:function:: approximate(rng_key: blackjax.types.PRNGKey, logdensity_fn: Callable, initial_position: blackjax.types.PyTree, num_samples: int = 200, *, maxiter=30, maxcor=10, maxls=1000, gtol=1e-08, ftol=1e-05) -> Tuple[PathfinderState, PathfinderInfo]

   Pathfinder variational inference algorithm.

   Pathfinder locates normal approximations to the target density along a
   quasi-Newton optimization path, with local covariance estimated using
   the inverse Hessian estimates produced by the L-BFGS optimizer.

   Function implements the algorithm 3 in :cite:p:`zhang2022pathfinder`:

   :param rng_key: PRPNG key
   :param logdensity_fn: (un-normalized) log densify function of target distribution to take
                         approximate samples from
   :param initial_position: starting point of the L-BFGS optimization routine
   :param num_samples: number of samples to draw to estimate ELBO
   :param maxiter: Maximum number of iterations of the LGBFS algorithm.
   :param maxcor: Maximum number of metric corrections of the LGBFS algorithm ("history
                  size")
   :param ftol: The LGBFS algorithm terminates the minimization when `(f_k - f_{k+1}) <
                ftol`
   :param gtol: The LGBFS algorithm terminates the minimization when `|g_k|_norm < gtol`
   :param maxls: The maximum number of line search steps (per iteration) for the LGBFS
                 algorithm

   :returns: * *A PathfinderState with information on the iteration in the optimization path*
             * *whose approximate samples yields the highest ELBO, and PathfinderInfo that*
             * *contains all the states traversed.*


.. py:function:: sample(rng_key: blackjax.types.PRNGKey, state: PathfinderState, num_samples: Union[int, Tuple[], Tuple[int]] = ()) -> blackjax.types.PyTree

   Draw from the Pathfinder approximation of the target distribution.

   :param rng_key: PRNG key
   :param state: PathfinderState containing information for sampling
   :param num_samples: Number of samples to draw

   :rtype: Samples drawn from the approximate Pathfinder distribution


