blackjax.ns.from_mcmc
=====================

.. py:module:: blackjax.ns.from_mcmc

.. autoapi-nested-parse::

   NS particle-update strategies that wrap a generic MCMC kernel under the
   likelihood constraint.



Classes
-------

.. autoapisummary::

   blackjax.ns.from_mcmc.ConstrainedMCMCInfo


Functions
---------

.. autoapisummary::

   blackjax.ns.from_mcmc.update_with_mcmc_take_last
   blackjax.ns.from_mcmc.reject_constrained_step
   blackjax.ns.from_mcmc.build_kernel


Module Contents
---------------

.. py:class:: ConstrainedMCMCInfo



   Info for a constrained MCMC proposal.

   .. attribute:: info

      The underlying MCMC info (e.g., RWInfo for random walk).

   .. attribute:: is_accepted

      True if both the MCMC proposal was accepted and the proposed
      point is above the likelihood threshold.


   .. py:attribute:: info
      :type:  NamedTuple


   .. py:attribute:: is_accepted
      :type:  jax.numpy.ndarray


.. py:function:: update_with_mcmc_take_last(constrained_mcmc_step_fn, num_mcmc_steps, num_delete)

   An update strategy for NS that uses MCMC to update the particles.
   For now we will not keep the states as they will be too large to store.
   Similar to the update_and_take_last from SMC.

   :param constrained_mcmc_step_fn: Wrapped MCMC step function that enforces the NS likelihood constraint.
   :param num_mcmc_steps: Number of MCMC proposals per particle.
   :param num_delete: Number of particles to replace per step.

   :returns: * *An update function that proposes new particles by running the constrained*
             * *MCMC kernel from survivor start points and returns the final states and*
             * *infos.*


.. py:function:: reject_constrained_step(init_state_fn: Callable, logdensity_fn: Callable, mcmc_init_fn: Callable, mcmc_step_fn: Callable) -> Callable

   Constrained inner step wrapping a generic MCMC kernel (propose-then-reject).

   Proposes one ``mcmc_step_fn`` move and accepts it only if the MCMC step
   accepted AND the proposed point is above the likelihood threshold; otherwise
   the particle stays put. The complement to
   :func:`slice_constrained_step` for kernels that cannot gate
   the constraint inside their own proposal.

   :param init_state_fn: Builds a particle state from a position and birth log-likelihood.
   :param logdensity_fn: Log-density of the (unconstrained) target passed to the MCMC kernel.
   :param mcmc_init_fn: Initializes the wrapped MCMC state from a position and ``logdensity_fn``.
   :param mcmc_step_fn: One step of the wrapped MCMC kernel,
                        ``(rng_key, mcmc_state, logdensity_fn, **params) -> (mcmc_state, info)``.

   :returns: * A constrained inner step ``(rng_key, state, loglikelihood_0, \*\*params) ->
             * (new_state, ConstrainedMCMCInfo)``.


.. py:function:: build_kernel(constrained_step_fn: Callable, num_inner_steps: int, update_inner_kernel_params_fn: Callable, num_delete: int = 1, delete_fn: Callable = default_delete_fn, update_strategy: Callable = update_with_mcmc_take_last) -> Callable

   Build a Nested Sampling kernel from a constrained inner step.

   The generic NS engine: run ``constrained_step_fn`` (a move that reports its
   in-contour ``is_accepted``) for ``num_inner_steps`` from survivor start points,
   take the last, and accumulate the evidence via the adaptive kernel. Build the
   step with :func:`reject_constrained_step` (generic MCMC) or, for the slice
   family, :func:`slice_constrained_step`.

   :param constrained_step_fn: Constrained inner step ``(rng_key, state, loglikelihood_0, **params) ->
                               (new_state, info)``.
   :param num_inner_steps: Number of inner steps per particle replacement.
   :param update_inner_kernel_params_fn: Recomputes the inner-kernel parameters from the live points each step.
   :param num_delete: Number of particles replaced per NS iteration.
   :param delete_fn: Selects which particles to delete (default: the lowest-likelihood ones).
   :param update_strategy: Inner-kernel factory ``(constrained_step_fn, num_inner_steps,
                           num_delete) -> update_fn`` (default:
                           :func:`update_with_mcmc_take_last`). The returned ``update_fn`` is
                           called once per NS iteration as::

                               update_fn(rng_key, state, loglikelihood_0, **step_parameters)
                                   -> (new_particles, update_info)

                           where ``loglikelihood_0`` is the likelihood contour being replaced and
                           ``step_parameters`` is the current ``state.inner_kernel_params``. It
                           must return exactly ``num_delete`` particle states whose pytree
                           structure and dtypes match ``state.particles``, since they are written
                           back by index, and each must satisfy ``loglikelihood >
                           loglikelihood_0`` with ``loglikelihood_birth`` set to
                           ``loglikelihood_0``. Driving the particles with ``constrained_step_fn``
                           gives the last two properties for free. ``update_info`` is passed
                           through to ``NSInfo.update_info`` unchanged and may be any pytree.

   :rtype: A Nested Sampling kernel ``kernel(rng_key, state) -> (new_state, info)``.


