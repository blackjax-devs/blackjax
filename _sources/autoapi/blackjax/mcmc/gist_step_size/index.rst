blackjax.mcmc.gist_step_size
============================

.. py:module:: blackjax.mcmc.gist_step_size

.. autoapi-nested-parse::

   GIST instance (a): self-tuning step size, autoStep-style.

   The tuning parameter is ``alpha = (a, b, j)``: ``(a, b)`` are soft
   acceptance-ratio thresholds freshly drawn every transition from the uniform
   distribution on the triangle ``Delta = {(a, b) in (0, 1)^2 : a < b}``, and
   ``j`` is the integer log2 step-size index (``step_size = initial_step_size *
   2**j``) selected *deterministically* given ``(theta, rho, a, b)`` by the
   doubling/halving selector :func:`step_size_selector` (section 2.1.2).

   Because ``p(a, b, j | theta, rho) = Uniform_Delta(a, b) . 1{j = mu(theta, rho,
   a, b)}`` is a Dirac measure in its ``j``-argument, the GIST tuning-density
   ratio (eq. 9) collapses to an indicator that the *reverse* selection ``j' =
   mu(theta', rho', a, b)`` (re-run at the proposal, with the *same* ``(a, b)``)
   matches the forward ``j`` -- the "reversibility check" of
   [autoMALA]/[AutoStep], derived directly from the GIST framework rather than
   bolted on as a separate correctness patch (section 2.1.3).

   .. rubric:: References

   .. [1] Bou-Rabee, Carpenter, Marsden, "GIST: Gibbs self-tuning for locally
      adaptive Hamiltonian Monte Carlo", arXiv:2404.15253, section 2 (mapping
      onto the general kernel).
   .. [2] Liu, Surjanovic, Biron-Lattes, Bouchard-Cote, Campbell, "AutoStep:
      Locally adaptive involutive MCMC", arXiv:2410.18929, Algorithm 2 (the
      symmetric selector, default here).
   .. [3] Biron-Lattes, Surjanovic, Syed, Campbell, Bouchard-Cote, "autoMALA:
      Locally adaptive Metropolis-adjusted Langevin algorithm",
      arXiv:2310.16782, Algorithm 2 (the asymmetric selector, ``criterion=
      "asymmetric"``, provided for cross-validation against the original paper
      only -- can get stuck near the mode/in the tails).



Attributes
----------

.. autoapisummary::

   blackjax.mcmc.gist_step_size.init


Classes
-------

.. autoapisummary::

   blackjax.mcmc.gist_step_size.StepSizeTuningParameter
   blackjax.mcmc.gist_step_size.GISTStepSizeInfo


Functions
---------

.. autoapisummary::

   blackjax.mcmc.gist_step_size.step_size_selector
   blackjax.mcmc.gist_step_size.build_kernel
   blackjax.mcmc.gist_step_size.as_top_level_api


Module Contents
---------------

.. py:data:: init

.. py:class:: StepSizeTuningParameter



   The GIST tuning parameter ``alpha = (a, b, j)``, section 2.1.1.

   a, b
       Soft acceptance-ratio thresholds, freshly drawn ~ Uniform on
       ``{(a, b) in (0, 1)^2 : a < b}`` every transition (the Gibbs refresh
       of the tuning parameter). The uniform density cancels in the
       acceptance ratio since ``g = identity`` carries ``(a, b)`` through
       the involution unchanged (section 2.1.3) -- any consistent
       (deterministic-given-key) way of drawing them would do, but the
       uniform-on-the-triangle draw is what the paper's own ``p(alpha |
       theta, rho)`` factorization assumes.
   step_index
       ``j``, the integer log2 step-size index:
       ``step_size = initial_step_size * 2**j``.


   .. py:attribute:: a
      :type:  blackjax.types.Array


   .. py:attribute:: b
      :type:  blackjax.types.Array


   .. py:attribute:: step_index
      :type:  blackjax.types.Array


.. py:class:: GISTStepSizeInfo



   Additional information for a ``gist_step_size`` transition.

   momentum, tuning_parameter, is_accepted, is_divergent, acceptance_rate,
   energy, num_integration_steps
       Same as :class:`~blackjax.mcmc.gist.GISTInfo` (this instance's Info
       extends it with flat, named fields rather than nesting -- matches
       the ``NUTSInfo``-extends-``HMCInfo``-fields precedent, not literal
       NamedTuple inheritance).
   step_index, reverse_step_index
       ``j`` (selected forward) and ``j'`` (re-selected at the proposal,
       the "reversibility check", section 2.1.3). ``is_accepted`` is False
       whenever ``reverse_step_index != step_index``, in addition to the
       ordinary energy-based rejection path -- both are folded into
       ``is_accepted``; these two fields let you tell them apart.
   search_exhausted
       True if the doubling/halving search (forward OR reverse) hit
       ``max_search_steps`` without the selection criterion terminating.
       When True, the transition was forced to reject regardless of the
       energy term (section 2.1.2).
   step_size
       The step size ``epsilon = initial_step_size * 2**step_index``
       actually used to build the proposal.


   .. py:attribute:: momentum
      :type:  blackjax.types.Array


   .. py:attribute:: tuning_parameter
      :type:  StepSizeTuningParameter


   .. py:attribute:: is_accepted
      :type:  blackjax.types.Array


   .. py:attribute:: is_divergent
      :type:  blackjax.types.Array


   .. py:attribute:: acceptance_rate
      :type:  blackjax.types.Array


   .. py:attribute:: energy
      :type:  float


   .. py:attribute:: num_integration_steps
      :type:  blackjax.types.Array


   .. py:attribute:: step_index
      :type:  blackjax.types.Array


   .. py:attribute:: reverse_step_index
      :type:  blackjax.types.Array


   .. py:attribute:: search_exhausted
      :type:  blackjax.types.Array


   .. py:attribute:: step_size
      :type:  blackjax.types.Array


.. py:function:: step_size_selector(integrator: Callable, num_integration_steps: int, initial_step_size: float, max_search_steps: int = 10, criterion: str = 'symmetric') -> Callable

   Build the ``mu(state, a, b, logdensity_fn, metric) -> (step_index,
   search_exhausted)`` selector, section 2.1.2.

   (``logdensity_fn``/``metric`` extend the paper's ``mu(state, a, b)``
   shorthand: evaluating ``F(alpha)`` fundamentally requires both, they are
   not a free design choice.)

   :param integrator: Symplectic integrator used for the ``num_integration_steps``-step
                      trial trajectories.
   :param num_integration_steps: ``L``, the fixed number of leapfrog steps per trial trajectory.
   :param initial_step_size: ``epsilon_init``, the fixed base step size the doubling/halving
                             search starts from and reports its selection relative to.
   :param max_search_steps: Cap on doubling/halving iterations.
   :param criterion: ``"symmetric"`` ([AutoStep] Algorithm 2, default -- proven
                     irreducible and aperiodic) or ``"asymmetric"`` ([autoMALA]'s
                     original criterion, provided for cross-validation against the
                     original paper only).

   :rtype: The selector ``mu``.


.. py:function:: build_kernel(integrator: Callable = integrators.velocity_verlet, divergence_threshold: float = 1000, criterion: str = 'symmetric', max_search_steps: int = 10) -> Callable

   Build a ``gist_step_size`` kernel.

   :param integrator: The symplectic integrator to use to integrate the Hamiltonian
                      dynamics.
   :param divergence_threshold: Value of the difference in energy above which we consider that the
                                transition is divergent.
   :param criterion: ``"symmetric"`` (default) or ``"asymmetric"``, see
                     :func:`step_size_selector`.
   :param max_search_steps: Cap on doubling/halving iterations (both the forward selection and
                            the reversibility-check re-selection).

   :returns: * *A kernel that takes a rng_key and a Pytree that contains the current*
             * *state of the chain and that returns a new state of the chain along with*
             * *information about the transition.*


.. py:function:: as_top_level_api(logdensity_fn: Callable, inverse_mass_matrix: blackjax.mcmc.metrics.MetricTypes, initial_step_size: float, num_integration_steps: int = 1, *, criterion: str = 'symmetric', max_search_steps: int = 10, divergence_threshold: float = 1000, integrator: Callable = integrators.velocity_verlet) -> blackjax.base.SamplingAlgorithm

   ``blackjax.gist_step_size`` -- GIST self-tuning step size (autoStep-style).

   .. rubric:: Examples

   A new ``gist_step_size`` kernel can be initialized and used with the
   following code:

   .. code::

       gist_step_size = blackjax.gist_step_size(
           logdensity_fn, inverse_mass_matrix, initial_step_size=0.1
       )
       state = gist_step_size.init(position)
       new_state, info = gist_step_size.step(rng_key, state)

   :param logdensity_fn: The log-density function we wish to draw samples from.
   :param inverse_mass_matrix: The value to use for the inverse mass matrix when drawing a value
                               for the momentum and computing the kinetic energy.
   :param initial_step_size: ``epsilon_init``, section 2.1.1; the fixed base step size the
                             doubling/halving search starts from and reports its selection
                             relative to (``epsilon = initial_step_size * 2**j``). NOT re-tuned
                             by this kernel -- round-based re-tuning of ``initial_step_size``
                             ([AutoStep]/[autoMALA] Algorithm 3) is out of scope; it belongs in
                             ``blackjax/adaptation/``.
   :param num_integration_steps: ``L``, the (fixed) number of leapfrog steps per proposal at the
                                 selected step size. Default 1 reproduces the MALA-equivalent
                                 single-leapfrog-step case both source papers analyze.
   :param criterion: ``"symmetric"`` (default, [AutoStep] Algorithm 2 -- proven
                     irreducible and aperiodic) or ``"asymmetric"`` ([autoMALA]'s
                     original criterion -- provided for cross-validation against the
                     original paper only; can get stuck near the mode/in the tails, see
                     the module docstring).
   :param max_search_steps: Cap on doubling/halving iterations (both the forward selection and
                            the reversibility-check re-selection). On exhaustion the transition
                            is forced to reject; see ``GISTStepSizeInfo.search_exhausted``.
   :param divergence_threshold: The absolute value of the difference in energy between two states
                                above which we say that the transition is divergent.
   :param integrator: (algorithm parameter) The symplectic integrator to use to integrate
                      the trajectory.

   :rtype: A ``SamplingAlgorithm``.


