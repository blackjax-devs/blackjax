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



Functions
~~~~~~~~~

.. autoapisummary::

   blackjax.mcmc.nuts.kernel



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
      :type: blackjax.types.PyTree

      

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

      


.. py:function:: kernel(integrator: Callable = integrators.velocity_verlet, divergence_threshold: int = 1000, max_num_doublings: int = 10)

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
   :param max_num_doublings: The maximum number of times we expand the trajectory by
                             doubling the number of steps if the trajectory does not
                             turn onto itself.


