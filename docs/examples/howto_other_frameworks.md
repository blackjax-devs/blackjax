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

# Use a logdensity function that is not compatible with JAX's primitives


We obviously recommend to use Blackjax with log-probability functions that are compatible with JAX's primitives. These can be built manually or with Aesara, Numpyro, Oryx, PyMC, TensorFlow-Probability.

Nevertheless, you may have a good reason to use a function that is incompatible with JAX's primitives, whether it is for performance reasons or for compatiblity with an already-implemented model. Who are we to judge?

In this example we will show you how this can be done using JAX's experimental `host_callback` API, and hint at a faster solution.

```{code-cell} ipython3
:tags: [remove-output]

import jax
from datetime import date
rng_key = jax.random.key(int(date.today().strftime("%Y%m%d")))
```

## Aesara model compiled to Numba

The following example builds a logdensity function with [Aesara](https://github.com/aesara-devs/aesara), compiles it with [Numba](https://numba.pydata.org/) and uses Blackjax to sample from the posterior distribution of the model.

```{code-cell} ipython3
import aesara.tensor as at
import numpy as np

srng = at.random.RandomStream(0)

loc = np.array([-2, 0, 3.2, 2.5])
scale = np.array([1.2, 1, 5, 2.8])
weights = np.array([0.2, 0.3, 0.1, 0.4])

N_rv = srng.normal(loc, scale, name="N")
I_rv = srng.categorical(weights, name="I")
Y_rv = N_rv[I_rv]
```

We can sample from the prior predictive distribution to make sure the model is correctly implemented:

```{code-cell} ipython3
import aesara

sampling_fn = aesara.function((), Y_rv)
print(sampling_fn())
print(sampling_fn())
```

We do not care about the posterior distribution of the indicator variable `I_rv` so we marginalize it out, and subsequently build the logdensity's graph:

```{code-cell} ipython3
from aeppl import joint_logprob

y_vv = Y_rv.clone()
i_vv = I_rv.clone()

logdensity = []
for i in range(4):
    i_vv = at.as_tensor(i, dtype="int64")
    component_logdensity, _ = joint_logprob(realized={Y_rv: y_vv, I_rv: i_vv})
    logdensity.append(component_logdensity)
logdensity = at.stack(logdensity, axis=0)

total_logdensity = at.logsumexp(at.log(weights) + logdensity)
```

We are now ready to compile the logdensity to Numba:

```{code-cell} ipython3
logdensity_fn = aesara.function((y_vv,), total_logdensity, mode="NUMBA")
logdensity_fn(1.)
```

As is we cannot use these functions within jit-compiled functions written with JAX, or apply `jax.grad` to get the function's gradients:

```{code-cell} ipython3
try:
    jax.jit(logdensity_fn)(1.)
except Exception:
    print("JAX raised an exception while jit-compiling!")

try:
    jax.grad(logdensity_fn)(1.)
except Exception:
    print("JAX raised an exception while differentiating!")
```

Indeed, a function written with Numba is incompatible with JAX's primitives. Luckily Aesara can build the model's gradient graph and compile it to Numba as well:

```{code-cell} ipython3
total_logdensity_grad = at.grad(total_logdensity, y_vv)
logdensity_grad_fn = aesara.function((y_vv,), total_logdensity_grad, mode="NUMBA")
logdensity_grad_fn(1.)
```

## Use `jax.experimental.host_callback` to call Numba functions

In order to be able to call `logdensity_fn` within JAX, we need to define a function that will call it via JAX's `host_callback`. Yet, this wrapper function is not differentiable with JAX, and so we will also need to define this functions' `custom_vjp`, and use `host_callback` to call the gradient-computing function as well:

```{code-cell} ipython3
import jax.experimental.host_callback as hcb

@jax.custom_vjp
def numba_logpdf(arg):
    return hcb.call(lambda x: logdensity_fn(x).item(), arg, result_shape=arg)

def call_grad(arg):
    return hcb.call(lambda x: logdensity_grad_fn(x).item(), arg, result_shape=arg)

def vjp_fwd(arg):
    return numba_logpdf(arg), call_grad(arg)

def vjp_bwd(grad_x, y_bar):
    return (grad_x * y_bar,)

numba_logpdf.defvjp(vjp_fwd, vjp_bwd)
```

And we can now call the function from a jitted function and apply `jax.grad` without JAX complaining:

```{code-cell} ipython3
:tags: [remove-stderr]

jax.jit(numba_logpdf)(1.), jax.grad(numba_logpdf)(1.)
```

And use Blackjax's NUTS sampler to sample from the model's posterior distribution:

```{code-cell} ipython3
import blackjax

inverse_mass_matrix = np.ones(1)
step_size=1e-3
nuts = blackjax.nuts(numba_logpdf, step_size, inverse_mass_matrix)
init = nuts.init(0.)

rng_key, init_key = jax.random.split(rng_key)
state, info = nuts.step(init_key, init)

for _ in range(10):
    rng_key, nuts_key = jax.random.split(rng_key)
    state, _ = nuts.step(nuts_key, state)

print(state)
```

If you run this on your machine you will notice that this runs quite slowly compared to a pure-JAX equivalent, that's because `host_callback` implied a lot of back-and-forth with Python. To see this let's compare execution times between *pure Numba on the one hand*:

```{code-cell} ipython3
%%time
for _ in range(100_000):
    logdensity_fn(100)
```

And *JAX on the other hand, with 100 times less iterations*:

```{code-cell} ipython3
%%time
for _ in range(1_000):
    numba_logpdf(100.)
```

That's a **lot** of overhead!

So while the implementation is simple considering what we're trying to achieve, it is only recommended for workloads where most of the time is spent evaluating the logdensity and its gradient, and where this overhead becomes irrelevant.


## Use custom XLA calls to call Numba functions faster


To avoid this kind overhead we can use an XLA custom call to execute Numba functions so there is no callback to Python in loops. Writing a function that performs such custom calls given a Numba function is a bit out of scope for this tutorial, but you can get inspiration from [jax-triton](https://github.com/jax-ml/jax-triton/blob/main/jax_triton/triton_call.py) to implement a custom call to a Numba function. You will also need to register a custom vjp, but you already know how to do that.
