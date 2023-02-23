:py:mod:`blackjax.mcmc.elliptical_slice`
========================================

.. py:module:: blackjax.mcmc.elliptical_slice

.. autoapi-nested-parse::

   Public API for the Elliptical Slice sampling Kernel



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   blackjax.mcmc.elliptical_slice.EllipSliceState
   blackjax.mcmc.elliptical_slice.EllipSliceInfo



Functions
~~~~~~~~~

.. autoapisummary::

   blackjax.mcmc.elliptical_slice.init
   blackjax.mcmc.elliptical_slice.kernel



.. py:class:: EllipSliceState



   State of the Elliptical Slice sampling algorithm.

   position
       Current position of the chain.
   logdensity
       Current value of the logdensity (evaluated at current position).


   .. py:attribute:: position
      :type: blackjax.types.PyTree

      

   .. py:attribute:: logdensity
      :type: blackjax.types.PyTree

      


.. py:class:: EllipSliceInfo



   Additional information on the Elliptical Slice sampling chain.

   This additional information can be used for debugging or computing
   diagnostics.

   momentum
       The latent momentum variable returned at the end of the transition.
   theta
       A value between [-2\pi, 2\pi] identifying points in the ellipsis drawn
       from the positon and mommentum variables. This value indicates the theta
       value of the accepted proposal.
   subiter
       Number of sub iterations needed to accept a proposal. The more subiterations
       needed the less efficient the algorithm will be, and the more dependent the
       new value is likely to be to the previous value.


   .. py:attribute:: momentum
      :type: blackjax.types.PyTree

      

   .. py:attribute:: theta
      :type: float

      

   .. py:attribute:: subiter
      :type: int

      


.. py:function:: init(position: blackjax.types.PyTree, logdensity_fn: Callable)


.. py:function:: kernel(cov_matrix: blackjax.types.Array, mean: blackjax.types.Array)

   Build an Elliptical Slice sampling kernel :cite:p:`murray2010elliptical`.

   :param cov_matrix: The value of the covariance matrix of the gaussian prior distribution from
                      the posterior we wish to sample.

   :returns: * *A kernel that takes a rng_key and a Pytree that contains the current state*
             * *of the chain and that returns a new state of the chain along with*
             * *information about the transition.*


