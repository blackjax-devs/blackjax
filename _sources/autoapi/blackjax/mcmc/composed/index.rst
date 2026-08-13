blackjax.mcmc.composed
======================

.. py:module:: blackjax.mcmc.composed

.. autoapi-nested-parse::

   The GIST (Gibbs self-tuning) sampler family: general seam plus instances.

   Single-folder package (mirrors :mod:`blackjax.adaptation.meta`) bundling the
   general kernel spine (:mod:`~blackjax.mcmc.composed._seam`) with its shipped
   instances -- :mod:`~blackjax.mcmc.composed.step_size` (self-tuning step size,
   autoStep-style, ``blackjax.gist_step_size``) and
   :mod:`~blackjax.mcmc.composed.trajectory_length` (self-tuning trajectory
   length, no-U-turn, ``blackjax.gist_trajectory_length``). A composed
   step-size x trajectory-length instance is planned as this package's next
   addition.

   References: [1] Bou-Rabee, Carpenter, Marsden, "GIST: Gibbs self-tuning for
   locally adaptive Hamiltonian Monte Carlo", arXiv:2404.15253; [2] Bou-Rabee,
   Carpenter, Kleppe, Marsden, "Incorporating Local Step-Size Adaptivity into
   the No-U-Turn Sampler using Gibbs Self Tuning", arXiv:2408.08259 -- the
   published sources of the composition machinery.



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/blackjax/mcmc/composed/step_size/index
   /autoapi/blackjax/mcmc/composed/trajectory_length/index


