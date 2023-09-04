:py:mod:`blackjax.mcmc.nuts`
============================

.. py:module:: blackjax.mcmc.nuts

.. autoapi-nested-parse::

   Public API for the NUTS Kernel



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   blackjax.mcmc.nuts.NUTSInfo
   blackjax.mcmc.nuts.nuts



Functions
~~~~~~~~~

.. autoapisummary::

   blackjax.mcmc.nuts.build_kernel



Attributes
~~~~~~~~~~

.. autoapisummary::

   blackjax.mcmc.nuts.init


.. py:data:: init

   

.. py:class:: NUTSInfo




   Additional information on the NUTS transition.

   This additional information can be used for debugging or computing
   diagnostics.

   momentum:
       The momentum that was sampled and used to integrate the trajectory.
   is_divergent
       Whether the difference in energy between the original and the new state
       exceeded the divergence threshold.
   is_turning
       Whether the sampling returned because the trajectory started turning
       back on itself.
   energy:
       Energy of the transition.
   trajectory_leftmost_state
       The leftmost state of the full trajectory.
   trajectory_rightmost_state
       The rightmost state of the full trajectory.
   num_trajectory_expansions
       Number of subtrajectory samples that were taken.
   num_integration_steps
       Number of integration steps that were taken. This is also the number of
       states in the full trajectory.
   acceptance_rate
       average acceptance probabilty across entire trajectory


   .. py:attribute:: momentum
      :type: blackjax.types.ArrayTree

      

   .. py:attribute:: is_divergent
      :type: bool

      

   .. py:attribute:: is_turning
      :type: bool

      

   .. py:attribute:: energy
      :type: float

      

   .. py:attribute:: trajectory_leftmost_state
      :type: blackjax.mcmc.integrators.IntegratorState

      

   .. py:attribute:: trajectory_rightmost_state
      :type: blackjax.mcmc.integrators.IntegratorState

      

   .. py:attribute:: num_trajectory_expansions
      :type: int

      

   .. py:attribute:: num_integration_steps
      :type: int

      

   .. py:attribute:: acceptance_rate
      :type: float

      


.. py:function:: build_kernel(integrator: Callable = integrators.velocity_verlet, divergence_threshold: int = 1000)

   Build an iterative NUTS kernel.

   This algorithm is an iteration on the original NUTS algorithm :cite:p:`hoffman2014no`
   with two major differences:

   - We do not use slice samplig but multinomial sampling for the proposal
     :cite:p:`betancourt2017conceptual`;
   - The trajectory expansion is not recursive but iterative :cite:p:`phan2019composable`,
     :cite:p:`lao2020tfp`.

   The implementation can seem unusual for those familiar with similar
   algorithms. Indeed, we do not conceptualize the trajectory construction as
   building a tree. We feel that the tree lingo, inherited from the recursive
   version, is unnecessarily complicated and hides the more general concepts
   upon which the NUTS algorithm is built.

   NUTS, in essence, consists in sampling a trajectory by iteratively choosing
   a direction at random and integrating in this direction a number of times
   that doubles at every step. From this trajectory we continuously sample a
   proposal. When the trajectory turns on itself or when we have reached the
   maximum trajectory length we return the current proposal.

   :param integrator: The simplectic integrator used to build trajectories.
   :param divergence_threshold: The absolute difference in energy above which we consider
                                a transition "divergent".


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

      kernel = blackjax.nuts.build_kernel(integrators.yoshida)
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

   :rtype: A ``SamplingAlgorithm``.

   .. py:attribute:: init

      

   .. py:attribute:: build_kernel

      


