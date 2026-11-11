---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Independent rejection sampling

Rejection sampling produces independent target draws when a proposal density
dominates the target by a known bound. Unlike MCMC, it does not require a chain
state or warmup. The target may be unnormalised, but the proposal density must
match the normalised distribution used by the proposal sampler.

Consider the Beta(2, 1) density $f(x)=2x$ on $[0,1]$. A uniform proposal has
$q(x)=1$ on the same interval, and $f(x)\leq 2q(x)$. The acceptance probability is
therefore $f(x)/(2q(x))=x$, with an average of two proposals per accepted draw.

```{code-cell} ipython3
import jax
import jax.numpy as jnp

from blackjax import rejection_sampling

def logdensity(x):
    return jnp.where((x >= 0) & (x < 1), jnp.log(2 * x), -jnp.inf)

def proposal_logdensity(x):
    return jnp.where((x >= 0) & (x < 1), 0.0, -jnp.inf)

kernel = rejection_sampling.build_kernel(
    logdensity_fn=logdensity,
    proposal_sampler=lambda key: jax.random.uniform(key),
    proposal_logdensity_fn=proposal_logdensity,
    log_bound=jnp.log(2.0),
    max_steps=128,
)

keys = jax.random.split(jax.random.key(0), 1000)
samples, info = jax.jit(jax.vmap(kernel))(keys)
assert jnp.all(info.is_bound_valid)
assert jnp.all(info.is_accepted)
print("Sample mean:", samples.mean())  # Population mean: 2/3
print("Proposals per draw:", info.num_proposals.mean())  # Expected value: 2
```

`jax.vmap` batches independent runs with separate random keys. Each run stops
when a proposal is accepted, an invalid ratio is encountered, or `max_steps` is
reached. The supplied density and proposal functions must be JAX-compatible,
including in uncompiled calls, because the loop body is traced.

Always inspect the returned information. If `is_accepted` is false, the returned
value is only the last proposal, not a target draw. An exhausted proposal budget
can be retried with a fresh key or a larger budget. An invalid envelope requires
correcting the bound, proposal support, or density functions before sampling.
The sampler checks the evaluated ratios, but cannot prove that the bound holds
everywhere. A bound estimated from observed proposals is not sufficient.

A loose envelope can make rejection sampling impractical, especially in high
dimensions. This utility does not select the proposal or estimate its bound.
