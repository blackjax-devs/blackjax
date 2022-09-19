---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Use BlackJAX with Aesara

Blackjax accepts any log-probability function as long as it is compatible with `jax.jit` and `jax.grad` (for gradient-based samplers). In this example we will show ho we can use [Aesara](https://github.com/aesara-devs/aesara) as a modeling language and Blackjax as an inference library.

This example relies on [Aesara](https://github.com/aesara-devs/aesara) and [AePPL](https://github.com/aesara-devs/aeppl). Please follow the installation instructions on their respective repository.

We will implement the following binomial response model with a beta prior:

$$
\begin{align*}
Y &\sim \operatorname{Binomial}(N, \theta)\\
\theta &\sim \operatorname{Beta}(\alpha, \beta)\\
\alpha, \beta &\sim \frac{1}{(\alpha + \beta)^{2.5}}
\end{align*}
$$

for the rat tumor dataset:

```{code-cell} ipython3
# index of array is type of tumor and value shows number of total people tested.
group_size = [20, 20, 20, 20, 20, 20, 20, 19, 19, 19, 19, 18, 18, 17, 20, 20, 20, 20, 19, 19, 18, 18, 25, 24, 23, 20, 20, 20, 20, 20, 20, 10, 49, 19, 46, 27, 17, 49, 47, 20, 20, 13, 48, 50, 20, 20, 20, 20, 20, 20, 20, 48, 19, 19, 19, 22, 46, 49, 20, 20, 23, 19, 22, 20, 20, 20, 52, 46, 47, 24, 14]

# index of array is type of tumor and value shows number of positve people.
n_of_positives = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 5, 2, 5, 3, 2, 7, 7, 3, 3, 2, 9, 10, 4, 4, 4, 4, 4, 4, 4, 10, 4, 4, 4, 5, 11, 12, 5, 5, 6, 5, 6, 6, 6, 6, 16, 15, 15, 9, 4]

n_rat_tumors = len(group_size)
```

We can build the graph of the logprob function using Aesara and AePPL:

```{code-cell} ipython3
import aesara
import aesara.tensor as at

from aeppl import joint_logprob

# Define the improper prior on `a` and `b`.
a_vv = at.scalar('a')
b_vv = at.scalar('b')
logprior = -2.5 * at.log(a_vv + b_vv)

srng = at.random.RandomStream(0)
theta_rv = srng.beta(a_vv, b_vv, size=(n_rat_tumors,))
Y_rv = srng.binomial(group_size, theta_rv)

# These are the value variables AePPL is going to replace the random variables
# with in the logprob graph.
theta_vv = theta_rv.clone()
Y_vv = Y_rv.clone()

loglikelihood = joint_logprob({Y_rv: Y_vv, theta_rv: theta_vv})
logprob = logprior + loglikelihood
```

We probably shouldn't be using NUTS (why?) for this example, but if we are going to use it we should use it well. The beta distribution generates samples between 0 and 1 and gradient-based algorithms like NUTS do not like these intervals much. So we apply a log-odds transformation to the beta-distributed variable and sample in the transformed space. AePPL can do the transfomation for us:

```{code-cell} ipython3
from aeppl.transforms import TransformValuesRewrite, LogOddsTransform

transforms_op = TransformValuesRewrite(
     {theta_vv: LogOddsTransform()}
)
loglikelihood = joint_logprob({Y_rv: Y_vv, theta_rv: theta_vv}, extra_rewrites=transforms_op)
logprob = logprior + loglikelihood
```

Let us now compile the logprob /graph/ to a /function/ that computes the log-probability:

```{code-cell} ipython3
logprob_fn = aesara.function((a_vv, b_vv, theta_vv, Y_vv), logprob)
```

This compiles the logprob function using Aesara's C backend. To sample with Blackjax we will need to use Aesara's JAX backend; it is still work in progress so the code will look complicated. All you need to know is that `jax_fn` is a function that uses JAX operators, can be passed as an argument to `jax.jit` and `jax.grad`:

```{code-cell} ipython3
from aesara.link.jax.dispatch import jax_funcify
from aesara.graph.fg import FunctionGraph
from aeppl.rewriting import logprob_rewrites_db
from aesara.compile import mode
from aesara.raise_op import CheckAndRaise

@jax_funcify.register(CheckAndRaise)
def jax_funcify_Assert(op, **kwargs):
    # Jax does not allow assert whose values aren't known during JIT compilation
    # within it's JIT-ed code. Hence we need to make a simple pass through
    # version of the Assert Op.
    # https://github.com/google/jax/issues/2273#issuecomment-589098722
    def assert_fn(value, *inps):
        return value

    return assert_fn

fgraph = FunctionGraph(inputs=(a_vv, b_vv, theta_vv, Y_vv), outputs=(logprob,))
mode.JAX.optimizer.rewrite(fgraph)
jax_fn = jax_funcify(fgraph)
```

Let us now inialize the parameter values:

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
init_position = tuple(init_param_fn(rng_key).values())

def logprob(position):
    return jax_fn(*position, n_of_positives)[0]

logprob(init_position)
```

And finally sample using Blackjax:

```{code-cell} ipython3
import blackjax

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

n_adapt = 3000
n_samples = 1000

adapt = blackjax.window_adaptation(blackjax.nuts, logprob, n_adapt, initial_step_size=1., target_acceptance_rate=0.8)
state, kernel, _ = adapt.run(rng_key, init_position)
states, infos = inference_loop(
    rng_key, kernel, state, n_samples
)
```
