blackjax.mcmc.slice
===================

.. py:module:: blackjax.mcmc.slice

.. autoapi-nested-parse::

   Public API for the Slice sampling kernel.



Classes
-------

.. autoapisummary::

   blackjax.mcmc.slice.SliceState
   blackjax.mcmc.slice.SliceInfo


Functions
---------

.. autoapisummary::

   blackjax.mcmc.slice.init
   blackjax.mcmc.slice.build_kernel
   blackjax.mcmc.slice.as_top_level_api


Module Contents
---------------

.. py:class:: SliceState



   State of the Slice sampling chain.

   position
       Current position of the chain.
   logdensity
       Current value of the log-density.



   .. py:attribute:: position
      :type:  blackjax.types.ArrayTree


   .. py:attribute:: logdensity
      :type:  float


.. py:class:: SliceInfo



   Additional information on the Slice sampling transition.

   bracket_widths
       Per-dimension realized bracket widths produced by the doubling
       procedure on this step.  Useful for diagnosing proposal efficiency
       (very wide brackets relative to the posterior suggest the initial
       width is too small; very narrow ones suggest it may be too large).



   .. py:attribute:: bracket_widths
      :type:  blackjax.types.ArrayTree


.. py:function:: init(position: blackjax.types.ArrayLikeTree, logdensity_fn: Callable) -> SliceState

   Create an initial state from a position and log-density function.

   :param position: Initial position of the chain.
   :param logdensity_fn: Log-probability density function of the target distribution.

   :rtype: The initial state of the Slice sampling chain.


.. py:function:: build_kernel(n_doublings: int = 10, initial_widths: float | blackjax.types.Array = 1.0) -> Callable

   Build a Slice sampling kernel.

   Implementation according to [1]. Doubling implementation inspired
   by TensorFlow Probability's implementation.

   :param n_doublings: Maximum number of slice interval doublings.
   :param initial_widths: Fixed bracket width(s) used as the starting interval for the doubling
                          procedure.  Accepts either a scalar (applied uniformly to every
                          coordinate, default: 1.0) or a 1-D array of length equal to the
                          total flattened position dimension ``D``, giving a per-coordinate
                          width — mirroring TFP's per-dimension ``step_size``.  The value 1.0
                          is a reasonable default for posterior scales in the range 0.1–10:
                          the doubling procedure rapidly expands the bracket when the width is
                          too small, so correctness is insensitive to the exact value.  Pass
                          a smaller value (or per-dimension array) for very narrow posteriors,
                          or a larger one for very diffuse ones, to improve efficiency.

   :returns: * *A kernel that takes a rng_key and a Pytree that contains the current state*
             * *of the chain and returns a new state of the chain along with information*
             * *about the transition.*

   .. rubric:: References

   .. [1] Radford M. Neal, "Slice sampling", The Annals of Statistics,
      Ann. Statist. 31(3), 705-767, (June 2003).


.. py:function:: as_top_level_api(logdensity_fn: Callable, *, n_doublings: int = 10, initial_widths: float | blackjax.types.Array = 1.0) -> blackjax.base.SamplingAlgorithm

   Implements the user interface for the Slice sampling kernel.

   .. rubric:: Examples

   A new Slice sampling kernel can be initialized and used with the following
   code:

   .. code::

       slice_sampling = blackjax.slice_sampling(logdensity_fn, n_doublings=10)
       state = slice_sampling.init(position)
       new_state, info = slice_sampling.step(rng_key, state)

   We can JIT-compile the step function for better performance:

   .. code::

       step = jax.jit(slice_sampling.step)
       new_state, info = step(rng_key, state)

   :param logdensity_fn: The log-density function of the distribution we wish to sample from.
   :param n_doublings: Maximum number of slice interval doublings (default: 10).
   :param initial_widths: Fixed bracket width(s) used as the starting interval for the doubling
                          procedure.  A scalar applies to all coordinates; a 1-D array of
                          length ``D`` (total flattened position dimension) gives per-coordinate
                          widths, mirroring TFP's per-dimension ``step_size`` (default: 1.0).

   :rtype: A ``SamplingAlgorithm``.


