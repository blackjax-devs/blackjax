blackjax.smc.waste_free
=======================

.. py:module:: blackjax.smc.waste_free


Functions
---------

.. autoapisummary::

   blackjax.smc.waste_free.update_waste_free
   blackjax.smc.waste_free.waste_free_smc


Module Contents
---------------

.. py:function:: update_waste_free(mcmc_init_fn, logposterior_fn, mcmc_step_fn, n_particles: int, p: int, num_resampled, num_mcmc_steps=None)

   Given M particles, mutates them using p-1 steps. Returns M*P-1 particles,
   consistent of the initial plus all the intermediate steps, thus implementing a
   waste-free update function
   See Algorithm 2: https://arxiv.org/abs/2011.02328


.. py:function:: waste_free_smc(n_particles, p)

