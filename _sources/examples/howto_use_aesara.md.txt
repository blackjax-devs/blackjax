---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.0
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Use with Aesara models

Blackjax accepts any log-probability function as long as it is compatible with `jax.jit`, `jax.grad` (for gradient-based samplers) and `jax.vmap`. In this example we will show ho we can use [Aesara](https://github.com/aesara-devs/aesara) as a modeling language and Blackjax as an inference library.

``` {admonition} Before you start
You will need [Aesara](https://github.com/aesara-devs/aesara) and [AePPL](https://github.com/aesara-devs/aeppl) to run this example. Please follow the installation instructions on their respective repository.
```

We will implement the following Binomial response model for the rat tumor dataset:

``` {math}
\begin{align*}
Y &\sim \operatorname{Binomial}(N, \theta)\\
\theta &\sim \operatorname{Beta}(\alpha, \beta)\\
\alpha, \beta &\sim \frac{1}{(\alpha + \beta)^{2.5}}
\end{align*}
```

```{code-cell} ipython3
:tags: [hide-cell]
# index of array is type of tumor and value shows number of total people tested.
group_size = [20, 20, 20, 20, 20, 20, 20, 19, 19, 19, 19, 18, 18, 17, 20, 20, 20, 20, 19, 19, 18, 18, 25, 24, 23, 20, 20, 20, 20, 20, 20, 10, 49, 19, 46, 27, 17, 49, 47, 20, 20, 13, 48, 50, 20, 20, 20, 20, 20, 20, 20, 48, 19, 19, 19, 22, 46, 49, 20, 20, 23, 19, 22, 20, 20, 20, 52, 46, 47, 24, 14]

# index of array is type of tumor and value shows number of positve people.
n_of_positives = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 5, 2, 5, 3, 2, 7, 7, 3, 3, 2, 9, 10, 4, 4, 4, 4, 4, 4, 4, 10, 4, 4, 4, 5, 11, 12, 5, 5, 6, 5, 6, 6, 6, 6, 16, 15, 15, 9, 4]

n_rat_tumors = len(group_size)
```

Let us know implement the model in Aesara/AePPL. We start with implementing the generative model in two part, the improper prior on `a` and `b` and then the response model:

```{code-cell} ipython3
import aesara
import aesara.tensor as at

from aeppl import joint_logprob

# improper prior on `a` and `b`.
a_vv = at.scalar('a')
b_vv = at.scalar('b')
logprior = -2.5 * at.log(a_vv + b_vv)

# response model
srng = at.random.RandomStream(0)
theta_rv = srng.beta(a_vv, b_vv, size=(n_rat_tumors,))
Y_rv = srng.binomial(group_size, theta_rv)
```

We can then easily compile a function that samples from the prior predictive distribution, i.e. returns values of `Y_rv` based on the variables' prior distribution. Let us make this function depend on the values of `a_vv` and `b_vv`:

```{code-cell} ipython3
prior_predictive_fn = aesara.function((a_vv, b_vv), Y_rv)
print(prior_predictive_fn(.5, .5))
print(prior_predictive_fn(.1, .3))
```

We can naively compile the log-probability density function of the model using AePPL's `joint_logprob`. This function takes the generative model's graph, and returns a graph that compute the model's logprob where the random variables `Y_rv` and `theta_rv` are replaced with the value variables `theta_vv` and `Y_vv` that we provide.

```{code-cell} ipython3
theta_vv = theta_rv.clone()
Y_vv = Y_rv.clone()

loglikelihood = joint_logprob({Y_rv: Y_vv, theta_rv: theta_vv})
logprob = logprior + loglikelihood
```

I said "naively", because the Beta distribution generates samples between 0 and 1 and gradient-based algorithms like NUTS work better on unbounded intervals. We can tell AePPL to apply a log-odds transformation to the Beta-distributed variable, and subsequently sample in the transformed space:

```{code-cell} ipython3
from aeppl.transforms import TransformValuesRewrite, LogOddsTransform

transforms_op = TransformValuesRewrite(
     {theta_vv: LogOddsTransform()}
)
loglikelihood = joint_logprob({Y_rv: Y_vv, theta_rv: theta_vv}, extra_rewrites=transforms_op)
logprob = logprior + loglikelihood
```


```{note}
NUTS is not the best sampler for this example, since the Beta distribution is the conjugate distribution of the Binomial. Marginalizing would lead to a faster sampler with less variance. [AeMCMC](https://github.com/aesara-devs/aemcmc) (in alpha state) makes this kind of transformation automatically on Aesara models.
```

You can alway debug the `logprob` graph by printing it:

```{code-cell} ipython3
aesara.dprint(logprob)
```

To sample with Blackjax we will need to use Aesara's JAX backend; `logprob_jax` defined below is a function that uses JAX operators, can be passed as an argument to `jax.jit` and `jax.grad`:

```{code-cell} ipython3
:tags: [remove-stderr]
logprob_fn = aesara.function((a_vv, b_vv, theta_vv, Y_vv), logprob, mode="JAX")
logprob_jax = logprob_fn.vm.jit_fn
```

JAX-compiled functions currently returns a tuple with a single element, but JAX can only differentiate scalar values and will complain. In addition, we would like to work with dictionaries for the values of the variables in Blackjax, and finally the value of `Y_vv` is fixed. So let's wrap the compiled function in a function that has the desired behavior:

```{code-cell}
def logprob_fn(position):
    flat_position = tuple(position.values())
    return logprob_jax(*flat_position, n_of_positives)[0]
```

We first need to define an initial position from which we are going to start sampling:

```{code-cell} ipython3
import jax

def init_param_fn(seed):
    """
    initialize a, b & thetas
    """
    key1, key2 = jax.random.split(seed)
    return {
        "a": jax.random.uniform(key1, (), "float64", minval=0, maxval=3),
        "b": jax.random.uniform(key2, (), "float64", minval=0, maxval=3),
        "thetas": jax.random.uniform(seed, (n_rat_tumors,), "float64", minval=0, maxval=1),
    }

rng_key = jax.random.PRNGKey(0)
init_position = init_param_fn(rng_key)
```

And finally sample using Blackjax:

```{code-cell} ipython3
:tags: [hide-cell]

def inference_loop(
    rng_key, kernel, initial_states, num_samples
):
    @jax.jit
    def one_step(states, rng_key):
        states, infos = kernel(rng_key, states)
        return states, (states, infos)

    keys = jax.random.split(rng_key, num_samples)
    _, (states, infos) = jax.lax.scan(one_step, initial_states, keys)

    return (states, infos)
```

```{code-cell} ipython3
import blackjax

n_adapt = 3000
n_samples = 1000

adapt = blackjax.window_adaptation(blackjax.nuts, logprob_fn, n_adapt)
state, kernel, _ = adapt.run(rng_key, init_position)

states, infos = inference_loop(
    rng_key, kernel, state, n_samples
)
```
