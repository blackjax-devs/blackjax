:py:mod:`blackjax.sgmcmc.sgld`
==============================

.. py:module:: blackjax.sgmcmc.sgld

.. autoapi-nested-parse::

   Public API for the Stochastic gradient Langevin Dynamics kernel.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   blackjax.sgmcmc.sgld.sgld



Functions
~~~~~~~~~

.. autoapisummary::

   blackjax.sgmcmc.sgld.init
   blackjax.sgmcmc.sgld.build_kernel



.. py:function:: init(position: blackjax.types.ArrayLikeTree) -> blackjax.types.ArrayLikeTree


.. py:function:: build_kernel() -> Callable

   Stochastic gradient Langevin Dynamics (SgLD) algorithm.


.. py:class:: sgld


   Implements the (basic) user interface for the SGLD kernel.

   The general sgld kernel builder (:meth:`blackjax.sgmcmc.sgld.build_kernel`, alias
   `blackjax.sgld.build_kernel`) can be cumbersome to manipulate. Since most users
   only need to specify the kernel parameters at initialization time, we
   provide a helper function that specializes the general kernel.

   .. rubric:: Example

   To initialize a SGLD kernel one needs to specify a schedule function, which
   returns a step size at each sampling step, and a gradient estimator
   function. Here for a constant step size, and `data_size` data samples:

   .. code::

       grad_fn = blackjax.sgmcmc.gradients.grad_estimator(logprior_fn, loglikelihood_fn, data_size)

   We can now initialize the sgld kernel and the state:

   .. code::

       sgld = blackjax.sgld(grad_fn)

   Assuming we have an iterator `batches` that yields batches of data we can
   perform one step:

   .. code::

       step_size = 1e-3
       minibatch = next(batches)
       new_position = sgld.step(rng_key, position, minibatch, step_size)

   Kernels are not jit-compiled by default so you will need to do it manually:

   .. code::

      step = jax.jit(sgld.step)
      new_position, info = step(rng_key, position, minibatch, step_size)

   :param grad_estimator: A function that takes a position, a batch of data and returns an estimation
                          of the gradient of the log-density at this position.

   :rtype: A ``SamplingAlgorithm``.

   .. py:attribute:: init

      

   .. py:attribute:: build_kernel

      


