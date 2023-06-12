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

   blackjax.mcmc.proposal.proposal_generator
   blackjax.mcmc.proposal.proposal_from_energy_diff
   blackjax.mcmc.proposal.asymmetric_proposal_generator
   blackjax.mcmc.proposal.static_binomial_sampling
   blackjax.mcmc.proposal.progressive_uniform_sampling
   blackjax.mcmc.proposal.progressive_biased_sampling
   blackjax.mcmc.proposal.nonreversible_slice_sampling



Attributes
~~~~~~~~~~

.. autoapisummary::

   blackjax.mcmc.proposal.TrajectoryState


.. py:data:: TrajectoryState

   

.. py:class:: Proposal




   Proposal for the next chain step.

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

      


.. py:function:: proposal_generator(energy: Callable, divergence_threshold: float) -> Tuple[Callable, Callable]

   :param energy: A function that computes the energy associated to a given state
   :param divergence_threshold: max value allowed for the difference in energies not to be considered a divergence

   :returns: * *Two functions, one to generate an initial proposal when no step has been taken,*
             * *another to generate proposals after each step.*


.. py:function:: proposal_from_energy_diff(initial_energy: float, new_energy: float, divergence_threshold: float, state: TrajectoryState) -> Tuple[Proposal, bool]

   Computes a new proposal from the energy difference between two states.
   It also verifies whether this difference is a divergence, if the
   energy diff is above divergence_threshold.
   :param initial_energy: the energy from the initial state
   :param new_energy: the energy at the proposed state
   :param divergence_threshold: max value allowed for the difference in energies not to be considered a divergence
   :param state: the proposed state

   :rtype: A proposal and a flag for divergence


.. py:function:: asymmetric_proposal_generator(transition_energy_fn: Callable, divergence_threshold: float, proposal_factory: Callable = proposal_from_energy_diff) -> Tuple[Callable, Callable]

   A proposal generator that takes into account the transition between
   two states to compute a new proposal. In particular, both states are
   used to compute the energies to consider in weighting the proposal,
   to account for asymmetries.
    ----------
   transition_energy_fn
       A function that computes the energy of a transition from an initial state
       to a new state, given some optional keyword arguments.
   divergence_threshold
       The maximum value allowed for the difference in energies not to be considered a divergence.
   proposal_factory
       A function that builds a proposal from the transition energies.

   :returns: * *Two functions, one to generate an initial proposal when no step has been taken,*
             * *another to generate proposals after each step.*


.. py:function:: static_binomial_sampling(rng_key, proposal, new_proposal)

   Accept or reject a proposal.

   In the static setting, the probability with which the new proposal is
   accepted is a function of the difference in energy between the previous and
   the current states. If the current energy is lower than the previous one
   then the new proposal is accepted with probability 1.



.. py:function:: progressive_uniform_sampling(rng_key, proposal, new_proposal)


.. py:function:: progressive_biased_sampling(rng_key, proposal, new_proposal)

   Baised proposal sampling :cite:p:`betancourt2017conceptual`.

   Unlike uniform sampling, biased sampling favors new proposals. It thus
   biases the transition away from the trajectory's initial state.



.. py:function:: nonreversible_slice_sampling(slice, proposal, new_proposal)

   Slice sampling for non-reversible Metropolis-Hasting update.

   Performs a non-reversible update of a uniform [0, 1] value
   for Metropolis-Hastings accept/reject decisions :cite:p:`neal2020non`, in addition
   to the accept/reject step of a current state and new proposal.



