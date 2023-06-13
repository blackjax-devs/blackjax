:py:mod:`blackjax.sgmcmc.sgnht`
===============================

.. py:module:: blackjax.sgmcmc.sgnht

.. autoapi-nested-parse::

   Public API for the Stochastic gradient Nosé-Hoover Thermostat kernel.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   blackjax.sgmcmc.sgnht.SGNHTState
   blackjax.sgmcmc.sgnht.sgnht



Functions
~~~~~~~~~

.. autoapisummary::

   blackjax.sgmcmc.sgnht.init
   blackjax.sgmcmc.sgnht.build_kernel



.. py:class:: SGNHTState




   State of the SGNHT algorithm.

   :param position: Current position in the sample space.
   :param momentum: Current momentum in the sample space.
   :param xi: Scalar thermostat controlling kinetic energy.

   .. py:attribute:: position
      :type: blackjax.types.ArrayTree

      

   .. py:attribute:: momentum
      :type: blackjax.types.ArrayTree

      

   .. py:attribute:: xi
      :type: float

      


.. py:function:: init(position: blackjax.types.ArrayLikeTree, rng_key: blackjax.types.PRNGKey, xi: float) -> SGNHTState


.. py:function:: build_kernel(alpha: float = 0.01, beta: float = 0) -> Callable

   Stochastic gradient Nosé-Hoover Thermostat (SGNHT) algorithm.


.. py:class:: sgnht


   Implements the (basic) user interface for the SGNHT kernel.

   The general sgnht kernel (:meth:`blackjax.sgmcmc.sgnht.build_kernel`, alias
   `blackjax.sgnht.build_kernel`) can be cumbersome to manipulate. Since most users
   only need to specify the kernel parameters at initialization time, we
   provide a helper function that specializes the general kernel.

   .. rubric:: Example

   To initialize a SGNHT kernel one needs to specify a schedule function, which
   returns a step size at each sampling step, and a gradient estimator
   function. Here for a constant step size, and `data_size` data samples:

   .. code::

       grad_estimator = blackjax.sgmcmc.gradients.grad_estimator(logprior_fn, loglikelihood_fn, data_size)

   We can now initialize the sgnht kernel and the state.

   .. code::

       sgnht = blackjax.sgnht(grad_estimator)
       state = sgnht.init(rng_key, position)

   Assuming we have an iterator `batches` that yields batches of data we can
   perform one step:

   .. code::

       step_size = 1e-3
       minibatch = next(batches)
       new_state = sgnht.step(rng_key, state, minibatch, step_size)

   Kernels are not jit-compiled by default so you will need to do it manually:

   .. code::

      step = jax.jit(sgnht.step)
      new_state = step(rng_key, state, minibatch, step_size)

   :param grad_estimator: A function that takes a position, a batch of data and returns an estimation
                          of the gradient of the log-density at this position.

   :rtype: A ``SamplingAlgorithm``.

   .. py:attribute:: init

      

   .. py:attribute:: build_kernel

      


