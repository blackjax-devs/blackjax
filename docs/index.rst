Welcome to Blackjax
===================

Blackjax is a library of samplers for `JAX <https://github.com/google/jax>`_ that works on CPU as well as GPU. It is designed with two categories of users in mind:

- People who just need state-of-the-art samplers that are fast, robust and well tested;
- Researchers who can use the library's building blocks to design new algorithms.

It integrates really well with PPLs as long as they can provide a (potentially unnormalized) log-probability density function compatible with JAX. And while you're here:

.. code-block:: python

   import jax
   import jax.numpy as jnp
   import jax.scipy.stats as stats
   import numpy as np

   import blackjax

   observed = np.random.normal(10, 20, size=1_000)
   def logprob_fn(x):
       logpdf = stats.norm.logpdf(observed, x["loc"], x["scale"])
       return jnp.sum(logpdf)

   # Build the kernel
   step_size = 1e-3
   inverse_mass_matrix = jnp.array([1., 1.])
   nuts = blackjax.nuts(logprob_fn, step_size, inverse_mass_matrix)

   # Initialize the state
   initial_position = {"loc": 1., "scale": 2.}
   state = nuts.init(initial_position)

   # Iterate
   rng_key = jax.random.PRNGKey(0)
   step = jax.jit(nuts.step)
   for _ in range(1_000):
      _, rng_key = jax.random.split(rng_key)
      state, _ = step(rng_key, state)

Installation
============

Blackjax is written in pure Python but depends on XLA via JAX. Since the JAX
installation depends on your CUDA version BlackJAX does not list JAX as a
dependency. If you simply want to use JAX on CPU, install it with:

.. code-block:: bash

   pip install jax jaxlib

Follow `these instructions <https://github.com/google/jax#installation>`_ to
install JAX with the relevant hardware acceleration support.

Then install BlackJAX

.. code-block:: bash

    pip install blackjax

.. toctree::
   :maxdepth: 2
   :caption: HOW TO

   howto_use_ppl
   Sample with multiple chains?<examples/howto_sample_multiple_chains.md>

.. toctree::
   :maxdepth: 1
   :caption: Blackjax by example

   examples

.. toctree::
   :maxdepth: 2
   :caption: API Documentation

   sampling
   adaptation
   diagnostics


Index
=====

* :ref:`genindex`
