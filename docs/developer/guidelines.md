# Developer Guidelines

In the following section, we’ll explain BlackJAX’s design of different algorithms for Monte Carlo integration. Keep in mind some basic principles:

- Leverage JAX's unique strengths: functional programming and composable function-transformation approach.
- Write small and general functions, compose them to create complex methods, and reuse the same building blocks for similar algorithms.
- Consider compatibility with the broader JAX ecosystem (Flax, Optax, GPJax).
- Write code that is easy to read and understand.
- Write well-documented code describing in detail the inner mechanism of the algorithm and its use.

## Core implementation
BlackJAX supports sampling algorithms such as Markov Chain Monte Carlo (MCMC), Sequential Monte Carlo (SMC), Stochastic Gradient MCMC (SGMCMC), and approximate inference algorithms such as Variational Inference (VI). In all cases, BlackJAX takes a Markovian approach, whereby its current state contains all the information to obtain the next iteration of an algorithm. This naturally results in a functionally pure structure, where no side-effects are allowed, simplifying parallelisation. Additionally, BlackJAX supports adaptation algorithms that efficiently tune the hyperparameters of sampling algorithms, usually aimed at reducing autocorrelation between sequential samples.

The user-facing interface of a **sampling algorithm** is made up of an initializer and an iterator:
```python
# Generic sampling algorithm:
sampling_algorithm = blackjax.sampling_algorithm(logdensity_fn, *args, **kwargs)
state = sampling_algorithm.init(initial_position)
new_state, info = sampling_algorithm.step(rng_key, state)
```
Build from the basic skeleton of a sampling algorithm [here](https://github.com/blackjax-devs/blackjax/tree/main/docs/developer/sampling_algorithm.py). Only the `sampling_algorithm` class and the `init` and `build_kernel` functions need to be in the final version of your algorithm; the rest might be useful but are not necessary.

The user-facing interface of an **approximate inference algorithm** is made up of an initializer, iterator, and sampler:
```python
# Generic approximate inference algorithm:
approx_inf_algorithm = blackjax.approx_inf_algorithm(logdensity_fn, optimizer, *args, **kwargs)
state = approx_inf_algorithm.init(initial_position)
new_state, info = approx_inf_algorithm.step(rng_key, state)
position_samples = approx_inf_algorithm.sample(rng_key, state, num_samples)
```
Build from the basic skeleton of an approximate inference algorithm [here](https://github.com/blackjax-devs/blackjax/tree/main/docs/developer/approximate_inf_algorithm.py). Only the `approx_inf_algorithm` class and the `init`, `step` and `sample` functions need to be in the final version of your algorithm; the rest might be useful but are not necessary.

## Basic components
All inference algorithms are composed of basic components which provide the lowest level of algorithm abstraction and are available to the user. When implementing a new inference algorithm, you should first break it down to its basic components, then find and use all already implemented *before* writing your own. For example, the [Metropolis-Hastings](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm) step, a basic component used by many MCMC algorithms to keep the target distribution invariant. In BlackJAX, two basic components do a specific (but simpler) and a general version of this accept/reject step:

- Metropolis step: if the proposal transition kernel is symmetric, i.e. if the probability of going from the initial to the proposed position is always equal to the probability of going from the proposed to the initial position, the acceptance probability is calculated using `mcmc.proposal.safe_energy_diff`, then the proposal is accepted or rejected using `mcmc.proposal.static_binomial_sampling`. For instance, see `mcmc.hmc.hmc_proposal`.
- Metropolis-Hastings step: for the more general case of an asymmetric proposal transition kernel, the acceptance probability is calculated by creating a proposal using `mcmc.proposal.compute_asymmetric_acceptance_ratio`, then the proposal is accepted or rejected using `mcmc.proposal.static_binomial_sampling`. For instance, see `mcmc.mala.build_kernel`.

When implementing an algorithm you could choose to replace the reversible binomial sampling step with Neal's [non-reversible slice sampling](https://arxiv.org/abs/2001.11950) step by simply replacing `mcmc.proposal.static_binomial_sampling` with `mcmc.proposal.nonreversible_slice_sampling` on either of the previous implementations. Make sure to carry over to the next iteration an updated slice for the slice sampling step, instead of passing a pseudo-random number generating key!

The previous example illustrates the practicality of basic components: they avoid rewriting the same methods and allow to easily test new algorithms that customize established algorithms, like how `blackjax.mcmc.ghmc` is `blackjax.mcmc.hmc` only with a persistent momentum and a non-reversible slice sampling step instead of the static binomial sampling step.

Because JAX operates on pure functions, inference algorithms always return a `typing.NamedTuple` containing the necessary variables to generate the next sample. Arguably, abstracting the handling of these variables is the whole point of BlackJAX, so you must do it in a way that abstracts the uninteresting bookkeeping and allows access to important variables at each step. The algorithms should also return a `typing.NamedTuple` with important information about each iteration.
