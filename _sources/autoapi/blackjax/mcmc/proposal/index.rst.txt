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
   blackjax.mcmc.proposal.static_binomial_sampling
   blackjax.mcmc.proposal.progressive_uniform_sampling
   blackjax.mcmc.proposal.progressive_biased_sampling
   blackjax.mcmc.proposal.nonreversible_slice_sampling



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
       cumulated Metropolis-Hastings acceptance probabilty across entire trajectory.


   .. py:attribute:: state
      :type: blackjax.mcmc.integrators.IntegratorState

      

   .. py:attribute:: energy
      :type: float

      

   .. py:attribute:: weight
      :type: float

      

   .. py:attribute:: sum_log_p_accept
      :type: float

      


.. py:function:: proposal_generator(kinetic_energy: Callable, divergence_threshold: float) -> Tuple[Callable, Callable]


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



