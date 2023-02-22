:py:mod:`blackjax.mcmc.marginal_latent_gaussian`
================================================

.. py:module:: blackjax.mcmc.marginal_latent_gaussian

.. autoapi-nested-parse::

   Public API for marginal latent Gaussian sampling.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   blackjax.mcmc.marginal_latent_gaussian.MarginalState
   blackjax.mcmc.marginal_latent_gaussian.MarginalInfo



Functions
~~~~~~~~~

.. autoapisummary::

   blackjax.mcmc.marginal_latent_gaussian.init_and_kernel



.. py:class:: MarginalState



   State of the RMH chain.

   x
       Current position of the chain.
   log_p_x
       Current value of the log-likelihood of the model
   grad_x
       Current value of the gradient of the log-likelihood of the model
   U_x
       Auxiliary attributes
   U_grad_x


   .. py:attribute:: position
      :type: blackjax.types.Array

      

   .. py:attribute:: logdensity
      :type: float

      

   .. py:attribute:: logdensity_grad
      :type: blackjax.types.Array

      

   .. py:attribute:: U_x
      :type: blackjax.types.Array

      

   .. py:attribute:: U_grad_x
      :type: blackjax.types.Array

      


.. py:class:: MarginalInfo



   Additional information on the RMH chain.

   This additional information can be used for debugging or computing
   diagnostics.

   acceptance_rate
       The acceptance probability of the transition, linked to the energy
       difference between the original and the proposed states.
   is_accepted
       Whether the proposed position was accepted or the original position
       was returned.
   proposal
       The state proposed by the proposal.


   .. py:attribute:: acceptance_rate
      :type: float

      

   .. py:attribute:: is_accepted
      :type: bool

      

   .. py:attribute:: proposal
      :type: MarginalState

      


.. py:function:: init_and_kernel(logdensity_fn, covariance, mean=None)

   Build the marginal version of the auxiliary gradient-based sampler

   :returns: * *A kernel that takes a rng_key and a Pytree that contains the current state*
             * *of the chain and that returns a new state of the chain along with*
             * *information about the transition.*
             * *An init function.*


