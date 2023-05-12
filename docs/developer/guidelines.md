# Developer Guidelines

In the broadest sense, an algorithm that belongs in the BlackJAX library should provide the tools to approximate integrals on a probability space. An introduction to probability theory is outside the scope of this document, but the Monte Carlo method is ever-present and important to understand. In simple terms, we want to approximate an integral with a sum. To do this, generate samples with [relative likelihood](https://en.wikipedia.org/wiki/Relative_likelihood) given by a target probability density function (known up to a normalization constant). The idea is to sample more from areas with higher likelihood but also from areas with low likelihood, just at a lower rate. You can also approximate the target density directly, using a density that is tractable and easy to sample from, then do inference with the approximation instead of the target, potentially using [importance sampling](https://en.wikipedia.org/wiki/Importance_sampling) to correct the approximation error.

In the following section, we’ll explain BlackJAX’s design of different algorithms for Monte Carlo integration. Keep in mind some basic principles:

- Leverage JAX's unique strengths: functional programming and composable function-transformation approach.
- Write small and general functions, compose them to create complex methods, reuse the same building blocks for similar algorithms.
- Consider compatibility with the broader JAX ecosystem (Flax, Optax, GPJax).
- Write code that is easy to read and understand.
- Write code that is well documented, describe in detail the inner mechanism of the algorithm and its use.

## Core implementation
There are three types of sampling algorithms BlackJAX currently supports: Markov Chain Monte Carlo (MCMC), Sequential Monte Carlo (SMC), and Stochastic Gradient MCMC (SGMCMC); and one type of approximate inference algorithm: Variational Inference (VI). Additionally, BlackJAX supports adaptation algorithms that efficiently tune the hyperparameters of sampling algorithms, usually aimed at reducing autocorrelation between sequential samples.

Basic components are functions which do specific tasks but are generally applicable, used to build all inference algorithms. When implementing a new inference algorithm you should first break it down to its basic components then find and use all that are already implemented *before* writing your own. A recurrent example is the [Metropolis-Hastings](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm) step, a basic component used by many MCMC algorithms to keep the target distribution invariant. In BlackJAX there are two basic components that do a specific (but simpler) and a general version of this accept/reject step:

- Metropolis step: if the proposal transition kernel is symmetric, i.e. if the probability of going from the initial to the proposed position is always equal to the probability of going from the proposed to the initial position, the acceptance probability is calculated by creating a proposal using `mcmc.proposal.proposal_generator`, then the proposal is accepted or rejected using `mcmc.proposal.static_binomial_sampling`.
- Metropolis-Hastings step: for the more general case of an asymmetric proposal transition kernel, the acceptance probability is calculated by creating a proposal using `mcmc.proposal.asymmetric_proposal_generator`, then the proposal is accepted or rejected using `mcmc.proposal.static_binomial_sampling`.

When implementing an algorithm you could choose to replace the classic, reversible Metropolis-Hastings step with Neal's [non-reversible slice sampling](https://arxiv.org/abs/2001.11950) step by simply replacing `mcmc.proposal.static_binomial_sampling` with `mcmc.proposal.nonreversible_slice_sampling` on either of the previous implementations. Just make sure to carry over to the next iteration an updated slice, instead of passing a pseudo-random number generating key, for the slice sampling step!

The previous example illustrates the power of basic components, useful not only to avoid rewriting the same methods for each new algorithm but also useful to personalize and test new algorithms which replace some steps of common efficient algorithms. Like how `blackjax.mcmc.ghmc` is `blackjax.mcmc.hmc` with a persistent momentum and a non-reversible slice sampling step instead of the Metropolis-Hastings step.

Because JAX operates on pure functions, inference algorithms always return a `typing.NamedTuple` containing the necessary variables to generate the next sample. Arguably, abstracting the handling of these variables is the whole point of BlackJAX, so it must be done in a way that abstracts the uninteresting bookkeeping from the end user but allows her to access important variables at each step. The algorithms should also return a `typing.NamedTuple` with important information about each iteration.

The user-facing interface of a **sampling algorithm** should work like this:
```python
sampling_algorithm = blackjax.sampling_algorithm(logdensity_fn, *args, **kwargs)
state = sampling_algorithm.init(initial_position)
new_state, info = sampling_algorithm.step(rng_key, state)
```
Achieve this by building from the basic skeleton of a sampling algorithm [here](https://github.com/blackjax-devs/blackjax/tree/main/docs/developer/sampling_algorithm.py). Only the `sampling_algorithm` class and the `init` and `build_kernel` functions need to be in the final version of your algorithm, the rest might become useful but are not necessary.

The user-facing interface of an **approximate inference algorithm** should work like this:
```python
approx_inf_algorithm = blackjax.approx_inf_algorithm(logdensity_fn, optimizer, *args, **kwargs)
state = approx_inf_algorithm.init(initial_position)
new_state, info = approx_inf_algorithm.step(rng_key, state)
#user is able to build the approximate distribution using the state, or generate samples:
position_samples = approx_inf_algorithm.sample(rng_key, state, num_samples)
```
Achieve this by building from the basic skeleton of an approximate inference algorithm [here](https://github.com/blackjax-devs/blackjax/tree/main/docs/developer/approximate_inf_algorithm.py). Only the `approx_inf_algorithm` class and the `init`, `step` and `sample` functions need to be in the final version of your algorithm, the rest might become useful but are not necessary.
