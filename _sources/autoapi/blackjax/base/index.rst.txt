:orphan:

:py:mod:`blackjax.base`
=======================

.. py:module:: blackjax.base


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   blackjax.base.InitFn
   blackjax.base.UpdateFn
   blackjax.base.MCMCSamplingAlgorithm
   blackjax.base.VIAlgorithm
   blackjax.base.RunFn
   blackjax.base.AdaptationAlgorithm




Attributes
~~~~~~~~~~

.. autoapisummary::

   blackjax.base.Position
   blackjax.base.State
   blackjax.base.Info


.. py:data:: Position

   

.. py:data:: State

   

.. py:data:: Info

   

.. py:class:: InitFn



   A `Callable` used to initialize the kernel state.

   Sampling algorithms often need to carry over some informations between
   steps, often to avoid computing the same quantity twice. Therefore the
   kernels do not operate on the chain positions themselves, but on states that
   contain this position and other information.

   The `InitFn` returns the state corresponding to a chain position. This state
   can then be passed to the `update` function of the `SamplingAlgorithm`.



.. py:class:: UpdateFn



   A transition kernel used as the `update` of a `SamplingAlgorithms`.

   Kernels are pure functions and are idempotent. They necessarily take a
   random state `rng_key` and the current kernel state (which contains the
   current position) as parameters, return a new state and some information
   about the transtion.

   Update functions is a simplified yet universal interface with every sampling
   algorithm. In essence, what all these algorithms do is take a rng state, a
   chain state (possibly a batch of data) and return a new state and some
   information about the transition.



.. py:class:: MCMCSamplingAlgorithm



   A pair of functions that represents a MCMC sampling algorithm.

   Blackjax sampling algorithms are implemented as a pair of pure functions: a
   kernel, that takes a new samples starting from the current state, and an
   initialization function that creates a kernel state from a chain position.

   As they represent Markov kernels, the kernel functions are pure functions
   and do not have internal state. To save computation time they also operate
   on states which contain the chain state and additional information that
   needs to be carried over for the next step.

   init:
       A pure function which when called with the initial position and the
       target density probability function will return the kernel's initial
       state.

   step:
       A pure function that takes a rng key, a state and possibly some
       parameters and returns a new state and some information about the
       transition.


   .. py:attribute:: init
      :type: InitFn

      

   .. py:attribute:: step
      :type: UpdateFn

      


.. py:class:: VIAlgorithm



   A pair of functions that represents a Variational Inference algorithm.

   Blackjax variational inference algorithms are implemented as a pair of pure
   functions: an approximator, which takes a target probability density (and
   potentially a guide), and a sampling function that uses the approximation to
   draw samples.

   approximate
       A pure function, which when called with an initial position (and
       potentially a guide function) returns a state that allows to build
       an approximation to the target probability density function.
   sample
       A pure function which returns samples from the approximation computed
       by `approximate`.


   .. py:attribute:: init
      :type: Callable

      

   .. py:attribute:: step
      :type: Callable

      

   .. py:attribute:: sample
      :type: Callable

      


.. py:class:: RunFn



   A `Callable` used to run the adaptation procedure.


.. py:class:: AdaptationAlgorithm



   A function that implements an adaptation algorithm.

   .. py:attribute:: run
      :type: RunFn

      


