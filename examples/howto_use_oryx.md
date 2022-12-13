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
mystnb:
  execution_timeout: 400
---

# Use with Oryx models

Oryx is a probabilistic programming library written in JAX, it is thus natively compatible with Blackjax. In this notebook we will show how we can use Oryx as a modeling language together with Blackjax as an inference library.

We reproduce the [example in Oryx's documentation](https://www.tensorflow.org/probability/oryx/notebooks/probabilistic_programming#case_study_bayesian_neural_network) and train a Bayesian Neural Network (BNN) on the iris dataset:

```{code-cell} python
from sklearn import datasets

iris = datasets.load_iris()
features, labels = iris['data'], iris['target']
num_features = features.shape[-1]
num_classes = len(iris.target_names)
```

```{code-cell} python
:tags: [hide-input]

print(f"Number of features: {num_features}")
print(f"Number of classes: {num_classes}")
print(f"Number of data points: {features.shape[0]}")
```

Oryx's approach, like Aesara's, is to implement probabilistic models as generative models and then apply transformations to get the log-probability density function. We begin with implementing a dense layer with normal prior probability on the weights and use the function `random_variable` to define random variables:

```{code-cell} python
import jax
from oryx.core.ppl import random_variable


from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions


def dense(dim_out, activation=jax.nn.relu):

    def forward(key, x):
        dim_in = x.shape[-1]
        w_key, b_key = jax.random.split(key)
        w = random_variable(
            tfd.Sample(tfd.Normal(0., 1.), sample_shape=(dim_out, dim_in)),
            name='w'
        )(w_key)
        b = random_variable(
            tfd.Sample(tfd.Normal(0., 1.), sample_shape=(dim_out,)),
            name='b'
        )(b_key)

        return activation(jnp.dot(w, x) + b)

    return forward
```

We now use this layer to build a multi-layer perceptron. The `nest` function is used to create "scope tags" that allows in this context to re-use our `dense` layer multiple times without name collision in the dictionary that will contain the parameters:

```{code-cell} python
from oryx.core.ppl import nest


def mlp(hidden_sizes, num_classes):
    num_hidden = len(hidden_sizes)

    def forward(key, x):
        keys = jax.random.split(key, num_hidden + 1)
        for i, (subkey, hidden_size) in enumerate(zip(keys[:-1], hidden_sizes)):
            x = nest(dense(hidden_size), scope=f'layer_{i + 1}')(subkey, x)
        logits = nest(dense(num_classes, activation=lambda x: x),
                        scope=f'layer_{num_hidden + 1}')(keys[-1], x)
        return logits

    return forward
```

Finally, we model the labels as categorical random variables:

```{code-cell} python
import functools

def predict(mlp):
    def forward(key, xs):
        mlp_key, label_key = jax.random.split(key)
        logits = jax.vmap(functools.partial(mlp, mlp_key))(xs)
        return random_variable(
            tfd.Independent(tfd.Categorical(logits=logits), 1), name='y')(label_key)

    return forward
```


We can now build the BNN and sample an initial position for the inference algorithm using `joint_sample`:

```{code-cell} python
import jax.numpy as jnp
from oryx.core.ppl import joint_sample


bnn = mlp([200, 200], num_classes)
initial_weights = joint_sample(bnn)(jax.random.PRNGKey(0), jnp.ones(num_features))

print(initial_weights.keys())
```

```{code-cell} python
:tags: [hide-input]

num_parameters = sum([layer.size for layer in jax.tree_util.tree_flatten(initial_weights)[0]])
print(f"Number of parameters in the model: {num_parameters}")
```

To sample from this model we will need to obtain its joint distribution log-probability using `joint_log_prob`:

```{code-cell} python
from oryx.core.ppl import joint_log_prob

def logprob_fn(weights):
  return joint_log_prob(predict(bnn))(dict(weights, y=labels), features)
```

We can now run the window adaptation to get good values for the parameters of the NUTS algorithm:

```{code-cell} python
%%time
import blackjax

rng_key = jax.random.PRNGKey(0)
adapt = blackjax.window_adaptation(blackjax.nuts, logprob_fn)
last_state, kernel, _ = adapt.run(rng_key, initial_weights, 100)
```

and sample from the model's posterior distribution:

```{code-cell} python
:tags: [hide-cell]

def inference_loop(rng_key, kernel, initial_state, num_samples):
    def one_step(state, rng_key):
        state, info = kernel(rng_key, state)
        return state, (state, info)

    keys = jax.random.split(rng_key, num_samples)
    _, (states, infos) = jax.lax.scan(one_step, initial_state, keys)

    return states, infos
```

```{code-cell} python
%%time

states, infos = inference_loop(rng_key, kernel, last_state, 100)
```

We can now use our samples to take an estimate of the accuracy that is averaged over the posterior distribution. We use `intervene` to "inject" the posterior values of the weights instead of sampling from the prior distribution:

```{code-cell} python
from oryx.core.ppl import intervene

posterior_weights = states.position

output_logits = jax.vmap(
    lambda weights: jax.vmap(lambda x: intervene(bnn, **weights)(
        jax.random.PRNGKey(0), x)
    )(features)
)(posterior_weights)

output_probs = jax.nn.softmax(output_logits)
```

```{code-cell} python
:tags: [hide-input]

print('Average sample accuracy:', (
    output_probs.argmax(axis=-1) == labels[None]).mean())

print('BMA accuracy:', (
    output_probs.mean(axis=0).argmax(axis=-1) == labels[None]).mean())
```
