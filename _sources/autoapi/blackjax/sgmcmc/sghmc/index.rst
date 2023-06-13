:py:mod:`blackjax.sgmcmc.sghmc`
===============================

.. py:module:: blackjax.sgmcmc.sghmc

.. autoapi-nested-parse::

   Public API for the Stochastic gradient Hamiltonian Monte Carlo kernel.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   blackjax.sgmcmc.sghmc.sghmc



Functions
~~~~~~~~~

.. autoapisummary::

   blackjax.sgmcmc.sghmc.init
   blackjax.sgmcmc.sghmc.build_kernel



.. py:function:: init(position: blackjax.types.ArrayLikeTree) -> blackjax.types.ArrayLikeTree


.. py:function:: build_kernel(alpha: float = 0.01, beta: float = 0) -> Callable

   Stochastic gradient Hamiltonian Monte Carlo (SgHMC) algorithm.


.. py:class:: sghmc


   Implements the (basic) user interface for the SGHMC kernel.

   The general sghmc kernel builder (:meth:`blackjax.sgmcmc.sghmc.build_kernel`, alias
   `blackjax.sghmc.build_kernel`) can be cumbersome to manipulate. Since most users
   only need to specify the kernel parameters at initialization time, we
   provide a helper function that specializes the general kernel.

   .. rubric:: Example

   To initialize a SGHMC kernel one needs to specify a schedule function, which
   returns a step size at each sampling step, and a gradient estimator
   function. Here for a constant step size, and `data_size` data samples:

   .. code::

       grad_estimator = blackjax.sgmcmc.gradients.grad_estimator(logprior_fn, loglikelihood_fn, data_size)

   We can now initialize the sghmc kernel and the state. Like HMC, SGHMC needs the user to specify a number of integration steps.

   .. code::

       sghmc = blackjax.sghmc(grad_estimator, num_integration_steps)

   Assuming we have an iterator `batches` that yields batches of data we can
   perform one step:

   .. code::

       step_size = 1e-3
       minibatch = next(batches)
       new_position = sghmc.step(rng_key, position, minibatch, step_size)

   Kernels are not jit-compiled by default so you will need to do it manually:

   .. code::

      step = jax.jit(sghmc.step)
      new_position, info = step(rng_key, position, minibatch, step_size)

   :param grad_estimator: A function that takes a position, a batch of data and returns an estimation
                          of the gradient of the log-density at this position.

   :rtype: A ``SamplingAlgorithm``.

   .. py:attribute:: init

      

   .. py:attribute:: build_kernel

      


