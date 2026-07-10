blackjax.adaptation.mclmc_adaptation
====================================

.. py:module:: blackjax.adaptation.mclmc_adaptation

.. autoapi-nested-parse::

   Algorithms to adapt the MCLMC kernel parameters, namely step size and L.



Classes
-------

.. autoapisummary::

   blackjax.adaptation.mclmc_adaptation.MCLMCAdaptationState


Functions
---------

.. autoapisummary::

   blackjax.adaptation.mclmc_adaptation.mclmc_find_L_and_step_size
   blackjax.adaptation.mclmc_adaptation.make_L_step_size_adaptation
   blackjax.adaptation.mclmc_adaptation.make_adaptation_L
   blackjax.adaptation.mclmc_adaptation.handle_nans


Module Contents
---------------

.. py:class:: MCLMCAdaptationState



   Represents the tunable parameters for MCLMC adaptation.

   L
       The momentum decoherent rate for the MCLMC algorithm.
   step_size
       The step size used for the MCLMC algorithm.
   inverse_mass_matrix
       A matrix used for preconditioning.


   .. py:attribute:: L
      :type:  float


   .. py:attribute:: step_size
      :type:  float


   .. py:attribute:: inverse_mass_matrix
      :type:  float


.. py:function:: mclmc_find_L_and_step_size(mclmc_kernel, num_steps, state, rng_key, logdensity_fn=None, frac_tune1=0.1, frac_tune2=0.1, frac_tune3=0.1, desired_energy_var=0.0005, trust_in_estimate=1.5, num_effective_samples=150, diagonal_preconditioning=True, params=None, l_factor=0.4)

   Finds the optimal value of the parameters for the MCLMC algorithm.

   :param mclmc_kernel: The kernel function built by ``mclmc.build_kernel``.  Its call signature
                        must be ``kernel(rng_key, state, logdensity_fn, inverse_mass_matrix, L,
                        step_size)``, matching the standard BlackJAX kernel pattern.
   :param num_steps: The number of MCMC steps that will subsequently be run, after tuning.
   :param state: The initial state of the MCMC algorithm.
   :param rng_key: The random number generator key.
   :param logdensity_fn: The log-density function of the target distribution.
   :param frac_tune1: The fraction of tuning for the first step of the adaptation.
   :param frac_tune2: The fraction of tuning for the second step of the adaptation.
   :param frac_tune3: The fraction of tuning for the third step of the adaptation.
   :param desired_energy_var: The desired energy variance for the MCMC algorithm.
   :param trust_in_estimate: The trust in the estimate of optimal stepsize.
   :param num_effective_samples: The number of effective samples for the MCMC algorithm.
   :param diagonal_preconditioning: Whether to do diagonal preconditioning (i.e. a mass matrix)
   :param params: Initial params to start tuning from (optional)
   :param l_factor: The factor scaling the estimated autocorrelation length to obtain momentum decoherence length L.

   :returns: * *final_state* -- The final integrator state after the three tuning phases.
             * *final_params* -- An ``MCLMCAdaptationState`` containing the adapted ``L``,
               ``step_size``, and ``inverse_mass_matrix``.
             * *total_num_tuning_integrator_steps* -- The total number of integrator steps consumed across all three
               tuning phases (frac_tune1 + frac_tune2 + frac_tune3 of
               ``num_steps``).

   .. rubric:: Example

   .. code-block:: python

       kernel = blackjax.mcmc.mclmc.build_kernel(integrator=integrator)

       (
           blackjax_state_after_tuning,
           blackjax_mclmc_sampler_params,
           num_tuning_steps,
       ) = blackjax.mclmc_find_L_and_step_size(
           mclmc_kernel=kernel,
           logdensity_fn=logdensity_fn,
           num_steps=num_steps,
           state=initial_state,
           rng_key=tune_key,
           diagonal_preconditioning=preconditioning,
       )

   .. rubric:: Notes

   **Live divergence monitoring (jax-tap >= 0.3.0)**

   The internal tuning scan exposes a per-step divergence flag as its ``ys``
   output (``True`` = divergence on that step).  Users who install
   ``jax-tap >= 0.3.0`` can observe this stream with no changes to BlackJAX::

       import jaxtap  # pip install "jax-tap>=0.3.0"

       with jaxtap.record(
           select_ys=lambda ys: ys[0],  # the single divergence-flag leaf
           alert_ys=lambda e: "divergence" if e.value else None,
           alert_ys_once=True,  # one stderr line then silence; drop for per-step
       ) as rec:
           state, params, _ = blackjax.mclmc_find_L_and_step_size(
               mclmc_kernel=kernel, num_steps=N, state=init_state,
               rng_key=key, logdensity_fn=logdensity_fn,
           )
       divergence_steps = [
           e.step for e in rec.events if e.kind == "output" and e.value
       ]

   **Checking for degenerate warmup**

   BlackJAX does not emit runtime warnings; checking is the user's
   responsibility.

   *Before calling* — verify the initial gradient is finite::

       from jax.flatten_util import ravel_pytree
       ok = jnp.all(jnp.isfinite(ravel_pytree(state.logdensity_grad)[0]))
       # finite logdensity + non-finite gradient = model/solver/support issue (#973)

   *After calling* — a collapsed warmup leaves ``step_size`` orders of magnitude
   below the posterior scale; healthy and frozen runs differ by ~6 orders::

       ratio = final_params.step_size * num_steps / final_params.L
       # ratio ≈ 1 → healthy;  ratio << 1 → likely frozen


.. py:function:: make_L_step_size_adaptation(kernel, logdensity_fn, dim, frac_tune1, frac_tune2, diagonal_preconditioning, desired_energy_var=0.001, trust_in_estimate=1.5, num_effective_samples=150)

   Adapts the stepsize and L of the MCLMC kernel. Designed for unadjusted MCLMC


.. py:function:: make_adaptation_L(kernel, logdensity_fn, frac, l_factor)

   determine L by the autocorrelations (around 10 effective samples are needed for this to be accurate)


.. py:function:: handle_nans(previous_state, next_state, step_size, step_size_max, kinetic_change, kernel_nonans, key)

   Adaptation-level NaN handler.

   If the kernel reported a divergence (via its truthful ``info.nonans`` after
   #969 fix), reduce ``step_size_max`` and return the pre-step state.  The
   kernel's own ``handle_nans`` already sanitises ``next_state`` for both
   divergence signatures:

   * Case-1: NaN position or momentum (position overshoot through a hard boundary).
   * Case-2: finite position + momentum but NaN ``logdensity`` (dominant under
     ``velocity_verlet`` at moderate overshoot on bounded targets).

   :param kernel_nonans: ``info.nonans`` from the MCLMC kernel — truthful after the #969 fix.

   :returns: ``True`` when the step was clean (no divergence and finite energy change).
   :rtype: success


