Sampling
========

.. currentmodule:: blackjax

.. autosummary::
  :nosignatures:

  hmc
  nuts
  mala
  orbital_hmc
  rmh
  tempered_smc
  adaptive_tempered_smc


At the highest level of Blackjax's API we find the sampling and adaptation algorithms. Sampling algorithms are made of an ``init_fn`` function, which turns a position into a sampling state, and a ``step_fn`` function, which transforms a state into a new state.

We initialize an algorithm using the log-probability function we wish to sample from, and values for the algorithms' parameters. Most common algorithms are available in the ``blackjax`` namespace directly:

.. code::

   import blackjax

   algorithm = blackjax.algorithm(logprob_fn, **parameters)


One can then initialize the sampling state and take a new sample starting from a given position in the parameter space:

.. code::

   import jax

   rng_key = jax.random.PRNGKey(0)

   state = algorithm.init(position)
   new_state, info = algorithm.step(rng_key, state)


Under the hood, kernels have a signature of the form ``kernel(rng_key, state, logprob_fn, **parameter)`` and this high-level interface provides convenient wrappers around these functions. It is possible to access the base kernel doing:

.. code::

   kernel = blackjax.algorithm.kernel(**algorithm_parameters)


The ``algorithm_parameters`` are different from the kernel ``parameters`` above. They characterize the structure of the kernel, and can be for instance the choice of integrator or metric for algorithms in the HMC family.

HMC
~~~

.. autoclass:: blackjax.hmc

MALA
~~~~

.. autoclass:: blackjax.mala

NUTS
~~~~

.. autoclass:: blackjax.nuts

Periodic Orbital
~~~~~~~~~~~~~~~~

.. autoclass:: blackjax.orbital_hmc

RMH
~~~

.. autoclass:: blackjax.rmh

Tempered SMC
~~~~~~~~~~~~

.. autoclass:: blackjax.tempered_smc

.. autoclass:: blackjax.adaptive_tempered_smc
