# Developer Guidelines

## Style
In its broadest sense, an algorithm that belongs in the blackjax library should approximate integrals on a probability space. An introduction to probability theory is outside the scope of this document, but the Monte Carlo method is ever-present and important to understand. In simple terms, we want to approximate an integral with a sum. To do this, generate samples with probabilities defined by a density (continuous variable) or measure (discrete variable) function. The idea is to sample more from areas with higher probability but also from areas with low probability, just at a lower rate. You can also approximate the target density directly, using an approximation that is easier to handle, then do inference, i.e. solve integrals, with the approximation directly and use importance sampling to correct its bias.

In the following section, we’ll explain blackjax’s design of different algorithms for Monte Carlo integration. Keep in mind some basic principles:

Leverage JAX's unique strengths: functional programming and composable function-transformation approach.
Write small and general functions, compose them to create complex methods, reuse the same building blocks for similar algorithms.
Consider compatibility with the broader JAX ecosystem (Flax, Optax, GPJax).
Write code that is easy to read and understand.
Write code that is well documented, describe in detail the inner mechanism of the algorithm and its use.

## Core implementation
There are three types of sampling algorithms blackjax currently supports: Markov Chain Monte Carlo (MCMC), Sequential Monte Carlo (SMC), and Stochastic Gradient MCMC (SGMCMC); and one type of approximate inference algorithm: Variational Inference (VI). Additionally, blackjax supports adaptation algorithms that efficiently tune the hyperparameters of sampling algorithms, usually aimed at reducing autocorrelation between sequential samples.

Basic components are functions, which do specific tasks but are generally applicable, used to build all inference algorithms. When implementing a new inference algorithm, you should first break it down to its basic components, then find and use all that are already implemented *before* writing your own. A recurrent example is the Metropolis-Hastings step, a basic component used by many MCMC algorithms to keep the target distribution invariant. In blackjax, this common accept/reject step done with two functions: first the Hastings ratio is calculated by creating a proposal using `mcmc.proposal.proposal_generator`, then the proposal is accepted or rejected using `mcmc.proposal.static_binomial_sampling`.

Because JAX operates on pure functions, inference algorithms always return a NamedTuple containing the necessary variables to generate the next sample. Arguably, abstracting the handling of these variables is the whole point of blackjax, so it must be done in a way that abstracts the uninteresting bookkeeping from the end user but allows her to access important variables at each step. The algorithms should also return a NamedTuple with important information of each iteration.

The user-facing interface of a **sampling algorithm** should work like this:
```python
import blackjax
sampling_algorithm = blackjax.sampling_algorithm(logdensity_fn, *args, **kwargs)
state = sampling_algorithm.init(initial_position)
new_state, info = sampling_algorithm.step(rng_key, state)
```
Achieve this by building from the basic skeleton of a sampling algorithm (here)[https://github.com/blackjax-devs/blackjax/tree/main/docs/developer/sampling_algorithm.py]. Only the `sampling_algorithm` class and the `init` and `build_kernel` functions need to be in the final version of your algorithm, the rest might become useful but are not necessary.

The user-facing interface of an **approximate inference algorithm** should work like this:
```python
import blackjax
approx_inf_algorithm = blackjax.approx_inf_algorithm(logdensity_fn, optimizer, *args, **kwargs)
state = approx_inf_algorithm.init(initial_position)
new_state, info = approx_inf_algorithm.step(rng_key, state)
#user is able to build the approximate distribution using the state, or generate samples:
position_samples = approx_inf_algorithm.sample(rng_key, state, num_samples)
```
Achieve this by building from the basic skeleton of an approximate inference algorithm (here)[https://github.com/blackjax-devs/blackjax/tree/main/docs/developer/approximate_inf_algorithm.py]. Only the `approx_inf_algorithm` class and the `init`, `step` and `sample` functions need to be in the final version of your algorithm, the rest might become useful but are not necessary.

Well documented code is essential for a useful library. Start by decomposing your algorithm into basic components, finding those that are already implemented, then implement your own and build the high-level API from basic components.
