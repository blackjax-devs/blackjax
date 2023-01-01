MCMC
====

.. currentmodule:: blackjax

.. autosummary::
  :nosignatures:

  nuts
  hmc
  ghmc
  mala
  mgrad_gaussian
  orbital_hmc
  elliptical_slice
  rmh


At the highest level of Blackjax's API we find the sampling and adaptation algorithms. Sampling algorithms are made of an ``init_fn`` function, which turns a position into a sampling state, and a ``step_fn`` function, which transforms a state into a new state.

We initialize an algorithm using the log-probability function we wish to sample from, and values for the algorithms' parameters. Most common algorithms are available in the ``blackjax`` namespace directly:

.. code::

   import blackjax

   algorithm = blackjax.algorithm(logdensity_fn, **parameters)


One can then initialize the sampling state and take a new sample starting from a given position in the parameter space:

.. code::

   import jax

   rng_key = jax.random.PRNGKey(0)

   state = algorithm.init(position)
   new_state, info = algorithm.step(rng_key, state)


Under the hood, kernels have a signature of the form ``kernel(rng_key, state, logdensity_fn, **parameter)`` and this high-level interface provides convenient wrappers around these functions. It is possible to access the base kernel doing:

.. code::

   kernel = blackjax.algorithm.kernel(**algorithm_parameters)


The ``algorithm_parameters`` are different from the kernel ``parameters`` above. They characterize the structure of the kernel, and can be for instance the choice of integrator or metric for algorithms in the HMC family.

NUTS
~~~~

.. autoclass:: blackjax.nuts

HMC
~~~

.. autoclass:: blackjax.hmc

Generalized HMC
~~~~~~~~~~~~~~~

.. autoclass:: blackjax.ghmc

MALA
~~~~

.. autoclass:: blackjax.mala

Marginal gradient sampler for latent gaussian model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: blackjax.mgrad_gaussian

Periodic Orbital
~~~~~~~~~~~~~~~~

.. autoclass:: blackjax.orbital_hmc

Elliptical Slice Sampler
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: blackjax.elliptical_slice

RMH
~~~

.. autoclass:: blackjax.rmh
