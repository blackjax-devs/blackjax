:py:mod:`blackjax.mcmc.trajectory`
==================================

.. py:module:: blackjax.mcmc.trajectory

.. autoapi-nested-parse::

   Procedures to build trajectories for algorithms in the HMC family.

   To propose a new state, algorithms in the HMC family generally proceed by :cite:p:`betancourt2017conceptual`:

   1. Sampling a trajectory starting from the initial point;
   2. Sampling a new state from this sampled trajectory.

   Step (1) ensures that the process is reversible and thus that detailed balance
   is respected. The traditional implementation of HMC does not sample a
   trajectory, but instead takes a fixed number of steps in the same direction and
   flips the momentum of the last state.

   We distinguish here between two different methods to sample trajectories: static
   and dynamic sampling. In the static setting we sample trajectories with a fixed
   number of steps, while in the dynamic setting the total number of steps is
   determined by a dynamic termination criterion. Traditional HMC falls in the
   former category, NUTS in the latter.

   There are also two methods to sample proposals from these trajectories. In the
   static setting we first build the trajectory and then sample a proposal from
   this trajectory. In the progressive setting we update the proposal as the
   trajectory is being sampled. While the former is faster, we risk saturating the
   memory by keeping states that will subsequently be discarded.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   blackjax.mcmc.trajectory.Trajectory
   blackjax.mcmc.trajectory.DynamicIntegrationState
   blackjax.mcmc.trajectory.DynamicExpansionState



Functions
~~~~~~~~~

.. autoapisummary::

   blackjax.mcmc.trajectory.append_to_trajectory
   blackjax.mcmc.trajectory.reorder_trajectories
   blackjax.mcmc.trajectory.merge_trajectories
   blackjax.mcmc.trajectory.static_integration
   blackjax.mcmc.trajectory.dynamic_progressive_integration
   blackjax.mcmc.trajectory.dynamic_recursive_integration
   blackjax.mcmc.trajectory.dynamic_multiplicative_expansion



.. py:class:: Trajectory



   .. py:attribute:: leftmost_state
      :type: blackjax.mcmc.integrators.IntegratorState

      

   .. py:attribute:: rightmost_state
      :type: blackjax.mcmc.integrators.IntegratorState

      

   .. py:attribute:: momentum_sum
      :type: blackjax.types.PyTree

      

   .. py:attribute:: num_states
      :type: int

      


.. py:function:: append_to_trajectory(trajectory: Trajectory, state: blackjax.mcmc.integrators.IntegratorState) -> Trajectory

   Append a state to the (right of the) trajectory to form a new trajectory.


.. py:function:: reorder_trajectories(direction: int, trajectory: Trajectory, new_trajectory: Trajectory) -> Tuple[Trajectory, Trajectory]

   Order the two trajectories depending on the direction.


.. py:function:: merge_trajectories(left_trajectory: Trajectory, right_trajectory: Trajectory)


.. py:function:: static_integration(integrator: Callable, direction: int = 1) -> Callable

   Generate a trajectory by integrating several times in one direction.


.. py:class:: DynamicIntegrationState



   .. py:attribute:: step
      :type: int

      

   .. py:attribute:: proposal
      :type: blackjax.mcmc.proposal.Proposal

      

   .. py:attribute:: trajectory
      :type: Trajectory

      

   .. py:attribute:: termination_state
      :type: NamedTuple

      


.. py:function:: dynamic_progressive_integration(integrator: Callable, kinetic_energy: Callable, update_termination_state: Callable, is_criterion_met: Callable, divergence_threshold: float)

   Integrate a trajectory and update the proposal sequentially in one direction
   until the termination criterion is met.

   :param integrator: The symplectic integrator used to integrate the hamiltonian trajectory.
   :param kinetic_energy: Function to compute the current value of the kinetic energy.
   :param update_termination_state: Updates the state of the termination mechanism.
   :param is_criterion_met: Determines whether the termination criterion has been met.
   :param divergence_threshold: Value of the difference of energy between two consecutive states above
                                which we say a transition is divergent.


.. py:function:: dynamic_recursive_integration(integrator: Callable, kinetic_energy: Callable, uturn_check_fn: Callable, divergence_threshold: float, use_robust_uturn_check: bool = False)

   Integrate a trajectory and update the proposal recursively in Python
   until the termination criterion is met.

   This is the implementation of Algorithm 6 from :cite:p:`hoffman2014no` with multinomial sampling.
   The implemenation here is mostly for validating the progressive implementation
   to make sure the two are equivalent. The recursive implementation should not
   be used for actually sampling as it cannot be jitted and thus likely slow.

   :param integrator: The symplectic integrator used to integrate the hamiltonian trajectory.
   :param kinetic_energy: Function to compute the current value of the kinetic energy.
   :param uturn_check_fn: Determines whether the termination criterion has been met.
   :param divergence_threshold: Value of the difference of energy between two consecutive states above which we say a transition is divergent.
   :param use_robust_uturn_check: Bool to indicate whether to perform additional U turn check between two trajectory.


.. py:class:: DynamicExpansionState



   .. py:attribute:: step
      :type: int

      

   .. py:attribute:: proposal
      :type: blackjax.mcmc.proposal.Proposal

      

   .. py:attribute:: trajectory
      :type: Trajectory

      

   .. py:attribute:: termination_state
      :type: NamedTuple

      


.. py:function:: dynamic_multiplicative_expansion(trajectory_integrator: Callable, uturn_check_fn: Callable, max_num_expansions: int = 10, rate: int = 2) -> Callable

   Sample a trajectory and update the proposal sequentially
   until the termination criterion is met.

   The trajectory is sampled with the following procedure:
   1. Pick a direction at random;
   2. Integrate `num_step` steps in this direction;
   3. If the integration has stopped prematurely, do not update the proposal;
   4. Else if the trajectory is performing a U-turn, return current proposal;
   5. Else update proposal, `num_steps = num_steps ** rate` and repeat from (1).

   :param trajectory_integrator: A function that runs the symplectic integrators and returns a new proposal
                                 and the integrated trajectory.
   :param uturn_check_fn: Function used to check the U-Turn criterion.
   :param step_size: The step size used by the symplectic integrator.
   :param max_num_expansions: The maximum number of trajectory expansions until the proposal is
                              returned.
   :param rate: The rate of the geometrical expansion. Typically 2 in NUTS, this is why
                the literature often refers to "tree doubling".


