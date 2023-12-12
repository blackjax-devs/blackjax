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
   blackjax.mcmc.marginal_latent_gaussian.mgrad_gaussian



Functions
~~~~~~~~~

.. autoapisummary::

   blackjax.mcmc.marginal_latent_gaussian.init
   blackjax.mcmc.marginal_latent_gaussian.build_kernel



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
       Gradient of the auxiliary attributes


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

      


.. py:function:: init(position, logdensity_fn, U_t)

   Initialize the marginal version of the auxiliary gradient-based sampler.

   :param position: The initial position of the chain.
   :param logdensity_fn: The logarithm of the likelihood function for the latent Gaussian model.
   :param U_t: The unitary array of the covariance matrix.


.. py:function:: build_kernel(cov_svd: CovarianceSVD)

   Build the marginal version of the auxiliary gradient-based sampler.

   :param cov_svd: The singular value decomposition of the covariance matrix.

   :returns: * *A kernel that takes a rng_key and a Pytree that contains the current state*
             * *of the chain and that returns a new state of the chain along with*
             * *information about the transition.*


.. py:class:: mgrad_gaussian


   Implements the marginal sampler for latent Gaussian model of :cite:p:`titsias2018auxiliary`.

   It uses a first order approximation to the log_likelihood of a model with Gaussian prior.
   Interestingly, the only parameter that needs calibrating is the "step size" delta,
   which can be done very efficiently.
   Calibrating it to have an acceptance rate of roughly 50% is a good starting point.

   .. rubric:: Examples

   A new marginal latent Gaussian MCMC kernel for a model q(x) ∝ exp(f(x)) N(x; m, C)
   can be initialized and used for a given "step size" delta with the following code:

   .. code::

       mgrad_gaussian = blackjax.mgrad_gaussian(f, C, mean=m, step_size=delta)
       state = mgrad_gaussian.init(zeros)  # Starting at the mean of the prior
       new_state, info = mgrad_gaussian.step(rng_key, state)

   We can JIT-compile the step function for better performance

   .. code::

       step = jax.jit(mgrad_gaussian.step)
       new_state, info = step(rng_key, state)

   :param logdensity_fn: The logarithm of the likelihood function for the latent Gaussian model.
   :param covariance: The covariance of the prior Gaussian density.
   :param mean: Mean of the prior Gaussian density. Default is zero.
   :type mean: optional

   :rtype: A ``SamplingAlgorithm``.

   .. py:attribute:: init

      

   .. py:attribute:: build_kernel

      


