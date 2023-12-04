:py:mod:`blackjax.mcmc.proposal`
================================

.. py:module:: blackjax.mcmc.proposal


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   blackjax.mcmc.proposal.Proposal



Functions
~~~~~~~~~

.. autoapisummary::

   blackjax.mcmc.proposal.safe_energy_diff
   blackjax.mcmc.proposal.proposal_generator
   blackjax.mcmc.proposal.progressive_uniform_sampling
   blackjax.mcmc.proposal.progressive_biased_sampling
   blackjax.mcmc.proposal.compute_asymmetric_acceptance_ratio
   blackjax.mcmc.proposal.static_binomial_sampling
   blackjax.mcmc.proposal.nonreversible_slice_sampling



Attributes
~~~~~~~~~~

.. autoapisummary::

   blackjax.mcmc.proposal.TrajectoryState


.. py:data:: TrajectoryState

   

.. py:class:: Proposal




   Proposal for the next chain step for MCMC with trajectory building (e.g., NUTS).

   state:
       The trajectory state that corresponds to this proposal.
   energy:
       The total energy that corresponds to this proposal.
   weight:
       Weight of the proposal. It is equal to the logarithm of the sum of the canonical
       densities of each state :math:`e^{-H(z)}` along the trajectory.
   sum_log_p_accept:
       cumulated Metropolis-Hastings acceptance probability across entire trajectory.


   .. py:attribute:: state
      :type: TrajectoryState

      

   .. py:attribute:: energy
      :type: float

      

   .. py:attribute:: weight
      :type: float

      

   .. py:attribute:: sum_log_p_accept
      :type: float

      


.. py:function:: safe_energy_diff(initial_energy: float, new_energy: float) -> float


.. py:function:: proposal_generator(energy_fn: Callable) -> tuple[Callable, Callable]

   :param energy_fn: A function that computes the energy associated to a given state

   :returns: * *Two functions, one to generate an initial proposal when no step has been taken,*
             * *another to generate proposals after each step.*


.. py:function:: progressive_uniform_sampling(rng_key: blackjax.types.PRNGKey, proposal: Proposal, new_proposal: Proposal) -> Proposal


.. py:function:: progressive_biased_sampling(rng_key: blackjax.types.PRNGKey, proposal: Proposal, new_proposal: Proposal) -> Proposal

   Baised proposal sampling :cite:p:`betancourt2017conceptual`.

   Unlike uniform sampling, biased sampling favors new proposals. It thus
   biases the transition away from the trajectory's initial state.



.. py:function:: compute_asymmetric_acceptance_ratio(transition_energy_fn: Callable) -> Callable

   Generate a meta function to compute the transition between two states.

   In particular, both states are used to compute the energies to consider in weighting
   the proposal, to account for asymmetries.

   :param transition_energy_fn: A function that computes the energy of a transition from an initial state
                                to a new state, given some optional keyword arguments.

   :rtype: A functions to compute the acceptance ratio .


.. py:function:: static_binomial_sampling(rng_key: blackjax.types.PRNGKey, log_p_accept: float, proposal, new_proposal)

   Accept or reject a proposal.

   In the static setting, the probability with which the new proposal is
   accepted is a function of the difference in energy between the previous and
   the current states. If the current energy is lower than the previous one
   then the new proposal is accepted with probability 1.



.. py:function:: nonreversible_slice_sampling(slice: blackjax.types.Array, delta_energy: float, proposal, new_proposal)

   Slice sampling for non-reversible Metropolis-Hasting update.

   Performs a non-reversible update of a uniform [0, 1] value
   for Metropolis-Hastings accept/reject decisions :cite:p:`neal2020non`, in addition
   to the accept/reject step of a current state and new proposal.



