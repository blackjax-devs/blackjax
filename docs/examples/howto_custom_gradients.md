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

# Use custom gradients


JAX provides a convenient `jax.grad` function to evaluate the gradient of any function build with JAX primitives. Which is why Blackjax uses `jax.grad` internally whenever it needs to evaluate the gradient. This should be enough for most applications, but sometimes you may need to provide your own gradients to blackjax for several reasons:

- You have a convenient closed-form expression for the gradient that is evaluated faster than the gradient that JAX produces;
- The forward-mode differentiation is faster than the backward-mode;
- The log-density function you are using is non differentiable by JAX, which is the case of [many optimizers](https://github.com/google/jaxopt).

Do not despair! Blackjax covers these use cases using JAX's [custom derivative dispatch](https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html). In the following we will consider a very academic example, but which should be enough to understand the mechanics of how to set custom gradients with JAX.

## Functions defined as the minimum of another function

Functions can be defined as the minimum of another one, $f(x) = min_{y} g(x,y)$. Computing their gradients may be tedious, especially if the minimisation happens numerically rather than in closed form. We show how automatic derivatives can be modified on such examples, resulting in better overall efficiency and stability.

Our example is taken from the theory of [convex conjugates](https://en.wikipedia.org/wiki/Convex_conjugate), used for example in optimal transport. Let's consider the following function:

$$
\begin{equation*}
g(x, y) = h(y) - \langle x, y\rangle,\qquad h(x) = \frac{1}{p}|x|^p,\qquad p > 1.
\end{equation*}
$$

And define the function $f$ as $f(x) = -min_y g(x, y)$ which we can be implemented as:

```{code-cell} ipython3
:tags: [remove-output]

import jax

from datetime import date
rng_key = jax.random.key(int(date.today().strftime("%Y%m%d")))
```

```{code-cell} ipython3
import jax.numpy as jnp
from jax.scipy.optimize import minimize

def h(x, p):
    out = jnp.abs(x) ** p
    return out / p

def f(x, p):
    """Returns the minimum value of g and where it is achieved.
    """
    def g(y):
        return jnp.sum(h(y, p) - x * y)

    res = minimize(g, jnp.zeros((1,)), method="BFGS")
    return -res.fun, res.x[0]
```

Note the we also return the value of $y$ where the minimum of $g$ is achieved (this will be useful later).


### Trying to differentate the function with `jax.grad`

The gradient of the function $f$ is undefined for JAX, which cannot differentiate through `while` loops used in BFGS, and trying to compute it directly raises an error:

```{code-cell} ipython3
# We only want the gradient with respect to `x`
try:
    jax.grad(f, has_aux=True)(0.5, 3)
except Exception as e:
    print(e)
```

### Deriving the gradient mathematically

In order to avoid this, we can leverage the mathematical structure of $f(x) = -\min_y h(y) - \langle x, y\rangle$. Indeed, asumming that the minimum is unique and achieved at $y(x)$ we have

```{math}
\begin{equation*}
    \frac{df}{dx} = -\bigg[\frac{dh}{dy} \frac{dy}{dx} - \frac{dy}{dx} x - y\bigg]
\end{equation*}
```

The first order optimality criterion

```{math}
\begin{equation*}
    \frac{dh}{dy} - x = 0,
\end{equation*}
```

ensures that

```{math}
\begin{equation*}
    \frac{df}{dx} = y(x).
\end{equation*}
```

In other words, the value of the derivative at $x$ is the value $y(x)$ at which the minimum of the function $g$ is achieved.


### Telling JAX to use a custom gradient

We can thus now tell JAX to compute the derivative of the function using the argmin using `jax.custom_vjp`

```{code-cell} ipython3
from functools import partial


@partial(jax.custom_jvp, nondiff_argnums=(1,))
def f_with_gradient(x, p):
    # We only return the value of f
    return f(x, p)[0]

@f_with_gradient.defjvp
def f_jac_vec_prod(p, primals, tangents):
    x, = primals
    x_dot, = tangents

    # We use the fact that the gradient of f is
    # the argmin.
    f_out, argmin = f(x, p)

    return f_out, argmin * x_dot
```

Which now outputs a value:

```{code-cell} ipython3
jax.grad(f_with_gradient)(0.31415, 3)
```

### Making sure the result is correct

The form of the function $g$ was specifically chosen because we have a closed-form expression for $f$ which is differentiable and will allow us to check the value of the previously defined gradient:

$$
\begin{align*}
f(x) &=\frac{1}{q}|x|^q\\
\frac{1}{q} + \frac{1}{p} &= 1\\
\end{align*}
$$

Which is obviously differentiable. We implement it:

```{code-cell} ipython3
def true_f(x, p):
    q = 1 / (1 - 1 / p)
    out = jnp.abs(x) ** q
    return out / q

print(jax.grad(true_f)(0.31415, 3))
```

And compare the gradient of this function with the custom gradient defined above:

```{code-cell} ipython3
:tags: [hide-input]

print(f"Gradient of closed-form f: {jax.grad(true_f)(0.31415, 3)}")
print(f"Custom gradient based on argmin: {jax.grad(f_with_gradient)(0.31415, 3)}")
```

They give close enough values! In other words, it suffices to know that the value of the gradient is the argmin to define a custom gradient function that gives good results.

+++

### Using the function with Blackjax


Let us now demonstrate that we can use `f_with_gradients` with Blackjax. We define a toy log-density function and use a gradient-based sampler:

```{code-cell} ipython3
import blackjax


def logdensity_fn(y):
    logdensity = jax.scipy.stats.norm.logpdf(y)
    x = f_with_gradient(y, 3)
    logdensity += jax.scipy.stats.norm.logpdf(x)
    return logdensity

hmc = blackjax.hmc(logdensity_fn,1e-2, jnp.ones(1), 20)
state = hmc.init(1.)

rng_key, step_key = jax.random.split(rng_key)
new_state, info = hmc.step(step_key, state)
```

```{code-cell} ipython3
state, new_state
```
