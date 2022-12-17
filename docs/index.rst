Welcome to Blackjax
===================

*The documentation corresponds to the current state of the `main` branch. There may be differences with the latest released version.*

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
   def logdensity_fn(x):
       logpdf = stats.norm.logpdf(observed, x["loc"], x["scale"])
       return jnp.sum(logpdf)

   # Build the kernel
   step_size = 1e-3
   inverse_mass_matrix = jnp.array([1., 1.])
   nuts = blackjax.nuts(logdensity_fn, step_size, inverse_mass_matrix)

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

The latest release of Blackjax can be installed from PyPi using `pip`:

.. code-block:: bash

   pip install blackjax

The current development branch can be installed from GitHub using `pip` as well:

.. code-block:: bash

   pip install git+https://github.com/blackjax-devs/blackjax


GPU-specific instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

BlackJAX is written in pure Python but depends on XLA via JAX. By default, the
version of JAX that will be installed along with BlackJAX will make your code
run on CPU only. **If you want to use BlackJAX on GPU/TPU** we recommend you follow
[these instructions](https://github.com/google/jax#installation) to install JAX
with the relevant hardware acceleration support.


.. code-block:: bash

    pip install blackjax

.. toctree::
   :maxdepth: 2
   :caption: HOW TO

   howto_use_ppl
   Sample with multiple chains?<examples/howto_sample_multiple_chains.md>
   Use custom gradients?<examples/howto_custom_gradients.md>
   Use non-JAX log-prob functions?<examples/howto_other_frameworks.md>

.. toctree::
   :maxdepth: 1
   :caption: Blackjax by example

   examples

.. toctree::
   :maxdepth: 2
   :caption: API Documentation

   mcmc
   sgmcmc
   smc
   vi
   adaptation
   diagnostics


Index
=====

* :ref:`genindex`
