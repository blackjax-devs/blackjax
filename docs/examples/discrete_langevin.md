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

# Metropolis-adjusted Langevin sampling on binary variables

`blackjax.dmala` implements the binary Metropolis-adjusted discrete Langevin
algorithm from [Zhang, Liu and Liu (2022)](https://proceedings.mlr.press/v162/zhang22t.html).
All coordinates can be proposed in a single transition. The forward and reverse
proposal probabilities enter the Metropolis correction.

The state consists of zero/one values, represented as floating-point arrays to
allow gradients of a differentiable extension of the log probability. The
extension guides proposals; its values on the binary states define the target.
This API supports binary arrays and PyTrees, not categorical or constrained
combinatorial spaces. It implements the adjusted algorithm, not DULA.

For example, sample a two-bit interacting model. Its four states can also be
enumerated to obtain the exact target mean:

```{code-cell} ipython3
import jax
import jax.numpy as jnp

import blackjax

def logdensity(x):
    return 0.4 * x[0] - 0.7 * x[1] + 0.8 * x[0] * x[1]

algorithm = blackjax.dmala(logdensity, step_size=0.7)
initial_state = algorithm.init(jnp.array([0, 0]))

def step(state, key):
    new_state, info = algorithm.step(key, state)
    return new_state, new_state.position

final_state, samples = jax.lax.scan(
    step, initial_state, jax.random.split(jax.random.key(0), 5000)
)
assert jnp.all((samples == 0) | (samples == 1))

states = jnp.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
probabilities = jax.nn.softmax(jax.vmap(logdensity)(states))
print("Exact target mean:", probabilities @ states)
print("Sample mean:", samples[1000:].mean(axis=0))
```

Use a positive finite scalar `step_size`. It is the paper's parameter alpha;
for a bit at x, the flip log odds are
`gradient * (1 - 2*x) / 2 - 1 / (2*alpha)`.
Smaller values discourage simultaneous flips; very small values can make the
chain mix slowly. Samples form a Markov chain and are correlated. Inspect
acceptance and convergence diagnostics rather than treating the iterations as
independent draws. Kernels are not compiled automatically; use `jax.jit` for
individual transitions or `jax.lax.scan` for a compiled loop.
