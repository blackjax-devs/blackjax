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

We obviously recommend using Blackjax with log-probability functions that are compatible with
JAX's primitives. These can be built manually or with Numpyro, Oryx, PyMC, TensorFlow-Probability.

Nevertheless, you may have a good reason to use a function that is incompatible with JAX's
primitives, whether it is for performance reasons or for compatibility with an already-implemented
model. Who are we to judge?

In this example we will show you how this can be done using `jax.pure_callback`, and hint at a
faster solution.

```{admonition} Before you start
You will need [PyTensor](https://pytensor.readthedocs.io/) and [Numba](https://numba.pydata.org/)
to run this example: ``pip install pytensor numba``
```

```{code-cell} ipython3
import jax
import jax.numpy as jnp
import numpy as np
from datetime import date
rng_key = jax.random.key(int(date.today().strftime("%Y%m%d")))
```

## A PyTensor model compiled to Numba

The following example builds a Gaussian mixture logdensity with
[PyTensor](https://pytensor.readthedocs.io/), compiles it to
[Numba](https://numba.pydata.org/) and uses Blackjax to sample from the posterior.

```{code-cell} ipython3
import numpy as np
import pytensor
import pytensor.tensor as pt
from pytensor.gradient import grad as pt_grad

loc    = np.array([-2.0, 0.0,  3.2, 2.5])
scale  = np.array([ 1.2, 1.0,  5.0, 2.8])
weights = np.array([0.2, 0.3,  0.1, 0.4])
```

We build the log-density graph symbolically in PyTensor, then compile both the
log-density and its gradient to Numba:

```{code-cell} ipython3
y_var = pt.scalar('y')

component_lp = (
    -0.5 * ((y_var - loc) / scale) ** 2
    - pt.log(scale)
    - 0.5 * pt.log(2 * np.pi)
)
total_lp   = pt.logsumexp(pt.log(weights) + component_lp, axis=0)
total_grad = pt_grad(total_lp, y_var)

logdensity_fn      = pytensor.function([y_var], total_lp,   mode="NUMBA")
logdensity_grad_fn = pytensor.function([y_var], total_grad, mode="NUMBA")

print(logdensity_fn(1.0))
print(logdensity_grad_fn(1.0))
```

As is we cannot use these Numba-compiled functions inside JIT-compiled JAX code:

```{code-cell} ipython3
import jax.numpy as jnp

try:
    jax.jit(logdensity_fn)(1.)
except Exception:
    print("JAX raised an exception while jit-compiling!")

try:
    jax.grad(logdensity_fn)(1.)
except Exception:
    print("JAX raised an exception while differentiating!")
```

## Use `jax.pure_callback` to call Numba functions from JAX

`jax.pure_callback` lets us call any Python (or Numba-compiled) function from inside
JIT-compiled JAX code. The callback is treated as a pure function — same inputs always
produce the same outputs.

Since JAX cannot automatically differentiate through the callback, we register a custom VJP
that calls the Numba-compiled gradient via a second `pure_callback`:

```{code-cell} ipython3
_result_shape = jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32)


@jax.custom_vjp
def numba_logpdf(y):
    return jax.pure_callback(
        lambda x: np.float32(logdensity_fn(float(x))),
        _result_shape,
        y,
    )


def _numba_logpdf_fwd(y):
    return numba_logpdf(y), y  # residuals = y


def _numba_logpdf_bwd(y, g):
    grad = jax.pure_callback(
        lambda x: np.float32(logdensity_grad_fn(float(x))),
        _result_shape,
        y,
    )
    return (g * grad,)


numba_logpdf.defvjp(_numba_logpdf_fwd, _numba_logpdf_bwd)
```

We can now call the function from JIT-compiled code and differentiate it:

```{code-cell} ipython3
:tags: [remove-stderr]

print(jax.jit(numba_logpdf)(1.0))
print(jax.grad(numba_logpdf)(1.0))
```

And use Blackjax's NUTS sampler to sample from the model's posterior distribution:

```{code-cell} ipython3
import blackjax

inverse_mass_matrix = np.ones(1)
step_size = 1e-3
nuts = blackjax.nuts(numba_logpdf, step_size, inverse_mass_matrix)
init = nuts.init(0.)

rng_key, init_key = jax.random.split(rng_key)
state, info = nuts.step(init_key, init)

for _ in range(10):
    rng_key, nuts_key = jax.random.split(rng_key)
    state, _ = nuts.step(nuts_key, state)

print(state)
```

If you run this on your machine you will notice that this runs quite slowly compared to a
pure-JAX equivalent. That's because `pure_callback` implies a round-trip from XLA back to
Python (and Numba) on every call. Let's see the overhead:

```{code-cell} ipython3
%%time
for _ in range(100_000):
    logdensity_fn(100.0)
```

```{code-cell} ipython3
%%time
for _ in range(1_000):
    numba_logpdf(100.)
```

That's a **lot** of overhead!

So while the implementation is simple considering what we're trying to achieve, it is only
recommended for workloads where most of the time is spent evaluating the logdensity and its
gradient, and where this overhead becomes irrelevant.


## Use custom XLA calls for better performance

To avoid this overhead you can use an XLA custom call to execute Numba functions without
any callback to Python in loops. Writing such a custom call is a bit out of scope for this
tutorial, but you can get inspiration from
[jax-triton](https://github.com/jax-ml/jax-triton/blob/main/jax_triton/triton_call.py).
You will also need to register a custom VJP, but you already know how to do that.
