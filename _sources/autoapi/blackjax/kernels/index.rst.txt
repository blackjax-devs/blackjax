:py:mod:`blackjax.kernels`
==========================

.. py:module:: blackjax.kernels

.. autoapi-nested-parse::

   Blackjax high-level interface with sampling algorithms.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   blackjax.kernels.adaptive_tempered_smc
   blackjax.kernels.tempered_smc
   blackjax.kernels.hmc
   blackjax.kernels.mala
   blackjax.kernels.nuts
   blackjax.kernels.mgrad_gaussian
   blackjax.kernels.sgld
   blackjax.kernels.sghmc
   blackjax.kernels.csgld
   blackjax.kernels.rmh
   blackjax.kernels.irmh
   blackjax.kernels.orbital_hmc
   blackjax.kernels.elliptical_slice
   blackjax.kernels.ghmc
   blackjax.kernels.pathfinder
   blackjax.kernels.meanfield_vi



Functions
~~~~~~~~~

.. autoapisummary::

   blackjax.kernels.window_adaptation
   blackjax.kernels.meads_adaptation
   blackjax.kernels.pathfinder_adaptation



.. py:class:: adaptive_tempered_smc

   Implements the (basic) user interface for the Adaptive Tempered SMC kernel.

   :param logprior_fn: The log-prior function of the model we wish to draw samples from.
   :param loglikelihood_fn: The log-likelihood function of the model we wish to draw samples from.
   :param mcmc_step_fn: The MCMC step function used to update the particles.
   :param mcmc_init_fn: The MCMC init function used to build a MCMC state from a particle position.
   :param mcmc_parameters: The parameters of the MCMC step function.
   :param resampling_fn: The function used to resample the particles.
   :param target_ess: The number of effective sample size to aim for at each step.
   :param root_solver: The solver used to adaptively compute the temperature given a target number
                       of effective samples.
   :param num_mcmc_steps: The number of times the MCMC kernel is applied to the particles per step.

   :rtype: A ``MCMCSamplingAlgorithm``.

   .. py:attribute:: init

      

   .. py:attribute:: kernel

      


.. py:class:: tempered_smc

   Implements the (basic) user interface for the Adaptive Tempered SMC kernel.

   :param logprior_fn: The log-prior function of the model we wish to draw samples from.
   :param loglikelihood_fn: The log-likelihood function of the model we wish to draw samples from.
   :param mcmc_step_fn: The MCMC step function used to update the particles.
   :param mcmc_init_fn: The MCMC init function used to build a MCMC state from a particle position.
   :param mcmc_parameters: The parameters of the MCMC step function.
   :param resampling_fn: The function used to resample the particles.
   :param num_mcmc_steps: The number of times the MCMC kernel is applied to the particles per step.

   :rtype: A ``MCMCSamplingAlgorithm``.

   .. py:attribute:: init

      

   .. py:attribute:: kernel

      


.. py:class:: hmc

   Implements the (basic) user interface for the HMC kernel.

   The general hmc kernel (:meth:`blackjax.mcmc.hmc.kernel`, alias `blackjax.hmc.kernel`) can be
   cumbersome to manipulate. Since most users only need to specify the kernel
   parameters at initialization time, we provide a helper function that
   specializes the general kernel.

   We also add the general kernel and state generator as an attribute to this class so
   users only need to pass `blackjax.hmc` to SMC, adaptation, etc. algorithms.

   .. rubric:: Examples

   A new HMC kernel can be initialized and used with the following code:

   .. code::

       hmc = blackjax.hmc(logdensity_fn, step_size, inverse_mass_matrix, num_integration_steps)
       state = hmc.init(position)
       new_state, info = hmc.step(rng_key, state)

   Kernels are not jit-compiled by default so you will need to do it manually:

   .. code::

      step = jax.jit(hmc.step)
      new_state, info = step(rng_key, state)

   Should you need to you can always use the base kernel directly:

   .. code::

      import blackjax.mcmc.integrators as integrators

      kernel = blackjax.hmc.kernel(integrators.mclachlan)
      state = blackjax.hmc.init(position, logdensity_fn)
      state, info = kernel(rng_key, state, logdensity_fn, step_size, inverse_mass_matrix, num_integration_steps)

   :param logdensity_fn: The log-density function we wish to draw samples from.
   :param step_size: The value to use for the step size in the symplectic integrator.
   :param inverse_mass_matrix: The value to use for the inverse mass matrix when drawing a value for
                               the momentum and computing the kinetic energy.
   :param num_integration_steps: The number of steps we take with the symplectic integrator at each
                                 sample step before returning a sample.
   :param divergence_threshold: The absolute value of the difference in energy between two states above
                                which we say that the transition is divergent. The default value is
                                commonly found in other libraries, and yet is arbitrary.
   :param integrator: (algorithm parameter) The symplectic integrator to use to integrate the trajectory.

   :rtype: A ``MCMCSamplingAlgorithm``.

   .. py:attribute:: init

      

   .. py:attribute:: kernel

      


.. py:class:: mala

   Implements the (basic) user interface for the MALA kernel.

   The general mala kernel (:meth:`blackjax.mcmc.mala.kernel`, alias `blackjax.mala.kernel`) can be
   cumbersome to manipulate. Since most users only need to specify the kernel
   parameters at initialization time, we provide a helper function that
   specializes the general kernel.

   We also add the general kernel and state generator as an attribute to this class so
   users only need to pass `blackjax.mala` to SMC, adaptation, etc. algorithms.

   .. rubric:: Examples

   A new MALA kernel can be initialized and used with the following code:

   .. code::

       mala = blackjax.mala(logdensity_fn, step_size)
       state = mala.init(position)
       new_state, info = mala.step(rng_key, state)

   Kernels are not jit-compiled by default so you will need to do it manually:

   .. code::

      step = jax.jit(mala.step)
      new_state, info = step(rng_key, state)

   Should you need to you can always use the base kernel directly:

   .. code::

      kernel = blackjax.mala.kernel(logdensity_fn)
      state = blackjax.mala.init(position, logdensity_fn)
      state, info = kernel(rng_key, state, logdensity_fn, step_size)

   :param logdensity_fn: The log-density function we wish to draw samples from.
   :param step_size: The value to use for the step size in the symplectic integrator.

   :rtype: A ``MCMCSamplingAlgorithm``.

   .. py:attribute:: init

      

   .. py:attribute:: kernel

      


.. py:class:: nuts

   Implements the (basic) user interface for the nuts kernel.

   .. rubric:: Examples

   A new NUTS kernel can be initialized and used with the following code:

   .. code::

       nuts = blackjax.nuts(logdensity_fn, step_size, inverse_mass_matrix)
       state = nuts.init(position)
       new_state, info = nuts.step(rng_key, state)

   We can JIT-compile the step function for more speed:

   .. code::

       step = jax.jit(nuts.step)
       new_state, info = step(rng_key, state)

   You can always use the base kernel should you need to:

   .. code::

      import blackjax.mcmc.integrators as integrators

      kernel = blackjax.nuts.kernel(integrators.yoshida)
      state = blackjax.nuts.init(position, logdensity_fn)
      state, info = kernel(rng_key, state, logdensity_fn, step_size, inverse_mass_matrix)

   :param logdensity_fn: The log-density function we wish to draw samples from.
   :param step_size: The value to use for the step size in the symplectic integrator.
   :param inverse_mass_matrix: The value to use for the inverse mass matrix when drawing a value for
                               the momentum and computing the kinetic energy.
   :param max_num_doublings: The maximum number of times we double the length of the trajectory before
                             returning if no U-turn has been obserbed or no divergence has occured.
   :param divergence_threshold: The absolute value of the difference in energy between two states above
                                which we say that the transition is divergent. The default value is
                                commonly found in other libraries, and yet is arbitrary.
   :param integrator: (algorithm parameter) The symplectic integrator to use to integrate the trajectory.

   :rtype: A ``MCMCSamplingAlgorithm``.

   .. py:attribute:: init

      

   .. py:attribute:: kernel

      


.. py:class:: mgrad_gaussian

   Implements the marginal sampler for latent Gaussian model of :cite:p:`titsias2018auxiliary`.

   It uses a first order approximation to the log_likelihood of a model with Gaussian prior.
   Interestingly, the only parameter that needs calibrating is the "step size" delta, which can be done very efficiently.
   Calibrating it to have an acceptance rate of roughly 50% is a good starting point.

   .. rubric:: Examples

   A new marginal latent Gaussian MCMC kernel for a model q(x) ∝ exp(f(x)) N(x; m, C) can be initialized and
   used for a given "step size" delta with the following code:

   .. code::

       mgrad_gaussian = blackjax.mgrad_gaussian(f, C, use_inverse=False, mean=m)
       state = mgrad_gaussian.init(zeros)  # Starting at the mean of the prior
       new_state, info = mgrad_gaussian.step(rng_key, state, delta)

   We can JIT-compile the step function for better performance

   .. code::

       step = jax.jit(mgrad_gaussian.step)
       new_state, info = step(rng_key, state, delta)

   :param logdensity_fn: The logarithm of the likelihood function for the latent Gaussian model.
   :param covariance: The covariance of the prior Gaussian density.
   :param mean: Mean of the prior Gaussian density. Default is zero.
   :type mean: optional

   :rtype: A ``MCMCSamplingAlgorithm``.


.. py:class:: sgld

   Implements the (basic) user interface for the SGLD kernel.

   The general sgld kernel (:meth:`blackjax.mcmc.sgld.kernel`, alias
   `blackjax.sgld.kernel`) can be cumbersome to manipulate. Since most users
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
       state = sgld.init(position)

   Assuming we have an iterator `batches` that yields batches of data we can
   perform one step:

   .. code::

       step_size = 1e-3
       minibatch = next(batches)
       new_state = sgld.step(rng_key, state, minibatch, step_size)

   Kernels are not jit-compiled by default so you will need to do it manually:

   .. code::

      step = jax.jit(sgld.step)
      new_state, info = step(rng_key, state, minibatch, step_size)

   :param grad_estimator: A function that takes a position, a batch of data and returns an estimation
                          of the gradient of the log-density at this position.

   :rtype: A ``MCMCSamplingAlgorithm``.

   .. py:attribute:: kernel

      


.. py:class:: sghmc

   Implements the (basic) user interface for the SGHMC kernel.

   The general sghmc kernel (:meth:`blackjax.mcmc.sghmc.kernel`, alias
   `blackjax.sghmc.kernel`) can be cumbersome to manipulate. Since most users
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
       state = sghmc.init(position)

   Assuming we have an iterator `batches` that yields batches of data we can
   perform one step:

   .. code::

       step_size = 1e-3
       minibatch = next(batches)
       new_state = sghmc.step(rng_key, state, minibatch, step_size)

   Kernels are not jit-compiled by default so you will need to do it manually:

   .. code::

      step = jax.jit(sghmc.step)
      new_state, info = step(rng_key, state, minibatch, step_size)

   :param grad_estimator: A function that takes a position, a batch of data and returns an estimation
                          of the gradient of the log-density at this position.

   :rtype: A ``MCMCSamplingAlgorithm``.

   .. py:attribute:: kernel

      


.. py:class:: csgld

   Implements the (basic) user interface for the Contour SGLD kernel.

   :param logdensity_estimator_fn: A function that returns an estimation of the model's logdensity given
                                   a position and a batch of data.
   :param zeta: Hyperparameter that controls the geometric property of the flattened
                density. If `zeta=0` the function reduces to the SGLD step function.
   :param temperature: Temperature parameter.
   :param num_partitions: The number of partitions we divide the energy landscape into.
   :param energy_gap: The difference in energy :math:`\Delta u` between the successive
                      partitions. Can be determined by running e.g. an optimizer to determine
                      the range of energies. `num_partition` * `energy_gap` should match this
                      range.
   :param min_energy: A rough estimate of the minimum energy in a dataset, which should be
                      strictly smaller than the exact minimum energy! e.g. if the minimum
                      energy of a dataset is 3456, we can set min_energy to be any value
                      smaller than 3456. Set it to 0 is acceptable, but not efficient enough.
                      the closer the gap between min_energy and 3456 is, the better.

   :rtype: A ``MCMCSamplingAlgorithm``.

   .. py:attribute:: init

      

   .. py:attribute:: kernel

      


.. py:function:: window_adaptation(algorithm: Union[hmc, nuts], logdensity_fn: Callable, is_mass_matrix_diagonal: bool = True, initial_step_size: float = 1.0, target_acceptance_rate: float = 0.8, progress_bar: bool = False, **extra_parameters) -> blackjax.base.AdaptationAlgorithm

   Adapt the value of the inverse mass matrix and step size parameters of
   algorithms in the HMC fmaily.

   Algorithms in the HMC family on a euclidean manifold depend on the value of
   at least two parameters: the step size, related to the trajectory
   integrator, and the mass matrix, linked to the euclidean metric.

   Good tuning is very important, especially for algorithms like NUTS which can
   be extremely inefficient with the wrong parameter values. This function
   provides a general-purpose algorithm to tune the values of these parameters.
   Originally based on Stan's window adaptation, the algorithm has evolved to
   improve performance and quality.

   :param algorithm: The algorithm whose parameters are being tuned.
   :param logdensity_fn: The log density probability density function from which we wish to
                         sample.
   :param is_mass_matrix_diagonal: Whether we should adapt a diagonal mass matrix.
   :param initial_step_size: The initial step size used in the algorithm.
   :param target_acceptance_rate: The acceptance rate that we target during step size adaptation.
   :param progress_bar: Whether we should display a progress bar.
   :param \*\*extra_parameters: The extra parameters to pass to the algorithm, e.g. the number of
                                integration steps for HMC.

   :rtype: A function that runs the adaptation and returns an `AdaptationResult` object.


.. py:function:: meads_adaptation(logdensity_fn: Callable, num_chains: int) -> blackjax.base.AdaptationAlgorithm

   Adapt the parameters of the Generalized HMC algorithm.

   The Generalized HMC algorithm depends on three parameters, each controlling
   one element of its behaviour: step size controls the integrator's dynamics,
   alpha controls the persistency of the momentum variable, and delta controls
   the deterministic transformation of the slice variable used to perform the
   non-reversible Metropolis-Hastings accept/reject step.

   The step size parameter is chosen to ensure the stability of the velocity
   verlet integrator, the alpha parameter to make the influence of the current
   state on future states of the momentum variable to decay exponentially, and
   the delta parameter to maximize the acceptance of proposal but with good
   mixing properties for the slice variable. These characteristics are targeted
   by controlling heuristics based on the maximum eigenvalues of the correlation
   and gradient matrices of the cross-chain samples, under simpifyng assumptions.

   Good tuning is fundamental for the non-reversible Generalized HMC sampling
   algorithm to explore the target space efficienty and output uncorrelated, or
   as uncorrelated as possible, samples from the target space. Furthermore, the
   single integrator step of the algorithm lends itself for fast sampling
   on parallel computer architectures.

   :param logdensity_fn: The log density probability density function from which we wish to sample.
   :param num_chains: Number of chains used for cross-chain warm-up training.

   :returns: * *A function that returns the last cross-chain state, a sampling kernel with the*
             * *tuned parameter values, and all the warm-up states for diagnostics.*


.. py:class:: rmh

   Implements the (basic) user interface for the gaussian random walk kernel

   .. rubric:: Examples

   A new Gaussian Random Walk kernel can be initialized and used with the following code:

   .. code::

       rmh = blackjax.rmh(logdensity_fn, sigma)
       state = rmh.init(position)
       new_state, info = rmh.step(rng_key, state)

   We can JIT-compile the step function for better performance

   .. code::

       step = jax.jit(rmh.step)
       new_state, info = step(rng_key, state)

   :param logdensity_fn: The log density probability density function from which we wish to sample.
   :param sigma: The value of the covariance matrix of the gaussian proposal distribution.

   :rtype: A ``MCMCSamplingAlgorithm``.

   .. py:attribute:: init

      

   .. py:attribute:: kernel

      


.. py:class:: irmh

   Implements the (basic) user interface for the independent RMH.

   .. rubric:: Examples

   A new kernel can be initialized and used with the following code:

   .. code::

       rmh = blackjax.irmh(logdensity_fn, proposal_distribution)
       state = rmh.init(position)
       new_state, info = rmh.step(rng_key, state)

   We can JIT-compile the step function for better performance

   .. code::

       step = jax.jit(rmh.step)
       new_state, info = step(rng_key, state)

   :param logdensity_fn: The log density probability density function from which we wish to sample.
   :param proposal_distribution: A Callable that takes a random number generator and produces a new proposal. The
                                 proposal is independent of the sampler's current state.

   :rtype: A ``MCMCSamplingAlgorithm``.

   .. py:attribute:: init

      

   .. py:attribute:: kernel

      


.. py:class:: orbital_hmc

   Implements the (basic) user interface for the Periodic orbital MCMC kernel.

   Each iteration of the periodic orbital MCMC outputs ``period`` weighted samples from
   a single Hamiltonian orbit connecting the previous sample and momentum (latent) variable
   with precision matrix ``inverse_mass_matrix``, evaluated using the ``bijection`` as an
   integrator with discretization parameter ``step_size``.

   .. rubric:: Examples

   A new Periodic orbital MCMC kernel can be initialized and used with the following code:

   .. code::

       per_orbit = blackjax.orbital_hmc(logdensity_fn, step_size, inverse_mass_matrix, period)
       state = per_orbit.init(position)
       new_state, info = per_orbit.step(rng_key, state)

   We can JIT-compile the step function for better performance

   .. code::

       step = jax.jit(per_orbit.step)
       new_state, info = step(rng_key, state)

   :param logdensity_fn: The logarithm of the probability density function we wish to draw samples from.
   :param step_size: The value to use for the step size in for the symplectic integrator to buid the orbit.
   :param inverse_mass_matrix: The value to use for the inverse mass matrix when drawing a value for
                               the momentum and computing the kinetic energy.
   :param period: The number of steps used to build the orbit.
   :param bijection: (algorithm parameter) The symplectic integrator to use to build the orbit.

   :rtype: A ``MCMCSamplingAlgorithm``.

   .. py:attribute:: init

      

   .. py:attribute:: kernel

      


.. py:class:: elliptical_slice

   Implements the (basic) user interface for the Elliptical Slice sampling kernel.

   .. rubric:: Examples

   A new Elliptical Slice sampling kernel can be initialized and used with the following code:

   .. code::

       ellip_slice = blackjax.elliptical_slice(loglikelihood_fn, cov_matrix)
       state = ellip_slice.init(position)
       new_state, info = ellip_slice.step(rng_key, state)

   We can JIT-compile the step function for better performance

   .. code::

       step = jax.jit(ellip_slice.step)
       new_state, info = step(rng_key, state)

   :param loglikelihood_fn: Only the log likelihood function from the posterior distributon we wish to sample.
   :param cov_matrix: The value of the covariance matrix of the gaussian prior distribution from the posterior we wish to sample.

   :rtype: A ``MCMCSamplingAlgorithm``.

   .. py:attribute:: init

      

   .. py:attribute:: kernel

      


.. py:class:: ghmc

   Implements the (basic) user interface for the Generalized HMC kernel.

   The Generalized HMC kernel performs a similar procedure to the standard HMC
   kernel with the difference of a persistent momentum variable and a non-reversible
   Metropolis-Hastings step instead of the standard Metropolis-Hastings acceptance
   step.

   This means that the sampling of the momentum variable depends on the previous
   momentum, the rate of persistence depends on the alpha parameter, and that the
   Metropolis-Hastings accept/reject step is done through slice sampling with a
   non-reversible slice variable also dependent on the previous slice, the determinisitc
   transformation is defined by the delta parameter.

   The Generalized HMC does not have a trajectory length parameter, it always performs
   one iteration of the velocity verlet integrator with a given step size, making
   the algorithm a good candiate for running many chains in parallel.

   .. rubric:: Examples

   A new Generalized HMC kernel can be initialized and used with the following code:

   .. code::

       ghmc_kernel = blackjax.ghmc(logdensity_fn, step_size, alpha, delta)
       state = ghmc_kernel.init(rng_key, position)
       new_state, info = ghmc_kernel.step(rng_key, state)

   We can JIT-compile the step function for better performance

   .. code::

       step = jax.jit(ghmc_kernel.step)
       new_state, info = step(rng_key, state)

   :param logdensity_fn: The log-density function we wish to draw samples from.
   :param step_size: A PyTree of the same structure as the target PyTree (position) with the
                     values used for as a step size for each dimension of the target space in
                     the velocity verlet integrator.
   :param alpha: The value defining the persistence of the momentum variable.
   :param delta: The value defining the deterministic translation of the slice variable.
   :param divergence_threshold: The absolute value of the difference in energy between two states above
                                which we say that the transition is divergent. The default value is
                                commonly found in other libraries, and yet is arbitrary.
   :param noise_gn: A function that takes as input the slice variable and outputs a random
                    variable used as a noise correction of the persistent slice update.
                    The parameter defaults to a random variable with a single atom at 0.

   :rtype: A ``MCMCSamplingAlgorithm``.

   .. py:attribute:: init

      

   .. py:attribute:: kernel

      


.. py:class:: pathfinder

   Implements the (basic) user interface for the pathfinder kernel.

   Pathfinder locates normal approximations to the target density along a
   quasi-Newton optimization path, with local covariance estimated using
   the inverse Hessian estimates produced by the L-BFGS optimizer.
   Pathfinder returns draws from the approximation with the lowest estimated
   Kullback-Leibler (KL) divergence to the true posterior.

   Note: all the heavy processing in performed in the init function, step
   function is just a drawing a sample from a normal distribution

   :param logdensity_fn: A function that represents the log-density of the model we want
                         to sample from.

   :rtype: A ``VISamplingAlgorithm``.

   .. py:attribute:: approximate

      

   .. py:attribute:: sample

      


.. py:function:: pathfinder_adaptation(algorithm: Union[hmc, nuts], logdensity_fn: Callable, initial_step_size: float = 1.0, target_acceptance_rate: float = 0.8, **extra_parameters) -> blackjax.base.AdaptationAlgorithm

   Adapt the value of the inverse mass matrix and step size parameters of
   algorithms in the HMC fmaily.

   :param algorithm: The algorithm whose parameters are being tuned.
   :param logdensity_fn: The log density probability density function from which we wish to sample.
   :param initial_step_size: The initial step size used in the algorithm.
   :param target_acceptance_rate: The acceptance rate that we target during step size adaptation.
   :param \*\*extra_parameters: The extra parameters to pass to the algorithm, e.g. the number of
                                integration steps for HMC.

   :returns: * *A function that returns the last chain state and a sampling kernel with the*
             * *tuned parameter values from an initial state.*


.. py:class:: meanfield_vi

   High-level implementation of Mean-Field Variational Inference.

   :param logdensity_fn: A function that represents the log-density function associated with
                         the distribution we want to sample from.
   :param optimizer: Optax optimizer to use to optimize the ELBO.
   :param num_samples: Number of samples to take at each step to optimize the ELBO.

   :rtype: A ``VIAlgorithm``.

   .. py:attribute:: init

      

   .. py:attribute:: step

      

   .. py:attribute:: sample

      


