---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
file_format: mystnb
mystnb:
  execution_timeout: 200
  merge_streams: true
---

# MNIST Digit Recognition With a 3-Layer Perceptron

This example is inspired form [this notebook](https://github.com/jeremiecoullon/SGMCMCJax/blob/master/docs/nbs/BNN.ipynb) in the SGMCMCJax repository. We try to use a 3-layer neural network to recognise the digits in the MNIST dataset.

```{code-cell} python
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import flax.linen as nn
import distrax
import numpy as np
from functools import partial
```

## Data Preparation

We download the MNIST data using `tensorflow-datasets`:

```{code-cell} python
:tags: [remove-stdout, remove-stderr]

import tensorflow_datasets as tfds

mnist_data, _ = tfds.load(
    name="mnist", batch_size=-1, with_info=True, as_supervised=True
)
mnist_data = tfds.as_numpy(mnist_data)
data_train, data_test = mnist_data["train"], mnist_data["test"]
```

Now we need to apply several transformations to the dataset before splitting it into a test and a test set:
- The images come into 28x28 pixels matrices; we reshape them into a vector;
- The images are arrays of RGB codes between 0 and 255. We normalize them by the maximum value to get a range between 0 and 1;
- We hot-encode category numbers.

```{code-cell} python
:tags: [remove-stdout, remove-stderr]

def one_hot_encode(x, k, dtype=np.float32):
    "Create a one-hot encoding of x of size k."
    return np.array(x[:, None] == np.arange(k), dtype)


def prepare_data(dataset: tuple, num_categories=10):
    X, y = dataset
    y = one_hot_encode(y, num_categories)

    num_examples = X.shape[0]
    num_pixels = 28 * 28
    X = X.reshape(num_examples, num_pixels)
    X = X / 255.0

    return jnp.array(X), jnp.array(y), num_examples


def batch_data(rng_key, data, batch_size, data_size):
    """Return an iterator over batches of data."""
    while True:
        _, rng_key = jax.random.split(rng_key)
        idx = jax.random.choice(
            key=rng_key, a=jnp.arange(data_size), shape=(batch_size,)
        )
        minibatch = tuple(elem[idx] for elem in data)
        yield minibatch


X_train, y_train, N_train = prepare_data(data_train)
X_test, y_test, N_test = prepare_data(data_test)
```

## Model: 3-layer Perceptron

We will use a very simple (bayesian) neural network in this example: A MLP with gaussian priors on the weights. We first need a function that computes the model's logposterior density given the data and the current values of the parameters. If we note $X$ the array that represents an image and $y$ the array such that $y_i = 0$  if the image is in category $i$, $y_i=1$ otherwise, the model can be written as:

```{math}
\begin{align*}
  \boldsymbol{p} &= \operatorname{NN}(X)\\
  \boldsymbol{y} &\sim \operatorname{Categorical}(\boldsymbol{p})
\end{align*}
```

```{code-cell} python
class NN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=100)(x)
        x = nn.softmax(x)
        x = nn.Dense(features=10)(x)
        return nn.log_softmax(x)

model = NN()


def logprior_fn(params):
    """Compute the value of the log-prior density function."""
    leaves, _ = jax.tree_util.tree_flatten(params)
    flat_params = jnp.concatenate([jnp.ravel(a) for a in leaves])
    return jnp.sum(distrax.Normal(0.0, 1.0).log_prob(flat_params))


def loglikelihood_fn(params, data):
    """Categorical log-likelihood"""
    X, y = data
    return jnp.sum(y * model.apply(params, X))


@jax.jit
def compute_accuracy(params, X, y):
    """Compute the accuracy of the model.

    To make predictions we take the number that corresponds to the highest probability value.
    """
    target_class = jnp.argmax(y, axis=1)
    predicted_class = jnp.argmax(model.apply(params, X), axis=1)
    return jnp.mean(predicted_class == target_class)
```

## Sample From the Posterior Distribution of the Perceptron's Weights

Now we need to get initial values for the parameters, and we simply sample from their prior distribution:

+++

We now sample from the model's posteriors using SGLD. We discard the first 1000 samples until the sampler has reached the typical set, and then take 2000 samples. We record the model's accuracy with the current values every 100 steps.

```{code-cell} python
from fastprogress.fastprogress import progress_bar

import blackjax
import blackjax.sgmcmc.gradients as gradients


data_size = len(y_train)
batch_size = 512
step_size = 5e-5

num_warmup = (data_size // batch_size) * 20
num_samples = 1000

# Batch the data
rng_key = jax.random.PRNGKey(1)
batches = batch_data(rng_key, (X_train, y_train), batch_size, data_size)

# Set the initial state
state = jax.jit(model.init)(rng_key, jnp.ones(X_train.shape[-1]))

# Build the SGLD kernel with a constant learning rate
grad_fn = gradients.estimator(logprior_fn, loglikelihood_fn, data_size)
sgld = blackjax.sgld(grad_fn)

# Sample from the posterior
accuracies = []
samples = []
steps = []
for step in progress_bar(range(num_samples + num_warmup)):
    _, rng_key = jax.random.split(rng_key)
    batch = next(batches)
    state = jax.jit(sgld)(rng_key, state, batch, step_size)
    if step % 100 == 0:
        accuracy = compute_accuracy(state, X_test, y_test)
        accuracies.append(accuracy)
        steps.append(step)
    if step > num_warmup:
        samples.append(state)
```

Let us plot the accuracy at different points in the sampling process:

```{code-cell} python
:tags: [hide-input]

import matplotlib.pylab as plt

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
ax.plot(steps, accuracies)
ax.set_xlabel("Number of sampling steps")
ax.set_ylabel("Prediction accuracy")
ax.set_xlim([0, num_warmup + num_samples])
ax.set_ylim([0, 1])
ax.set_yticks([0.1, 0.3, 0.5, 0.7, 0.9])
plt.title("Sample from 3-layer MLP posterior (MNIST dataset) with SgLD")
plt.plot();
```

### Sampling with SGHMC

We can also use SGHMC to samples from this model

```{code-cell} python
# Build the SGHMC kernel with a constant learning rate
step_size = 9e-6
grad_fn = gradients.estimator(logprior_fn, loglikelihood_fn, data_size)
sghmc = blackjax.sghmc(grad_fn)

# Batch the data
state = jax.jit(model.init)(rng_key, jnp.ones(X_train.shape[-1]))

# Sample from the posterior
sghmc_accuracies = []
samples = []
steps = []
for step in progress_bar(range(num_samples + num_warmup)):
    _, rng_key = jax.random.split(rng_key)
    minibatch = next(batches)
    state = jax.jit(sghmc)(rng_key, state, minibatch, step_size)
    if step % 100 == 0:
        sghmc_accuracy = compute_accuracy(state, X_test, y_test)
        sghmc_accuracies.append(sghmc_accuracy)
        steps.append(step)
    if step > num_warmup:
        samples.append(state)
```

```{code-cell} python
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
ld_plot, = ax.plot(steps, accuracies)
hmc_plot, = ax.plot(steps, sghmc_accuracies)
ax.set_xlabel("Number of sampling steps")
ax.set_ylabel("Prediction accuracy")
ax.set_xlim([0, num_warmup + num_samples])
ax.set_ylim([0, 1])
ax.set_yticks([0.1, 0.3, 0.5, 0.7, 0.9])
plt.title("Sample from 3-layer MLP posterior (MNIST dataset)")
ax.legend((ld_plot, hmc_plot), ('SGLD', 'SGHMC'), loc='lower right', shadow=True)
plt.plot();
```

```{code-cell} python
:tags: [hide-input]

print(f"The average accuracy for SGLD in the sampling phase is {100 * np.mean(accuracies[10:]):.2f}%")
print(f"The average accuracy for SGHMC in the sampling phase is {100 * np.mean(sghmc_accuracies[10:]):.2f}%")
```

Which is not a bad accuracy at all for such a simple model! Remember though that we draw samples from the posterior distribution of the digit probabilities; we can thus use this information to filter out examples for which the model is "unsure" of its prediction.

Here we will say that the model is unsure of its prediction for a given image if the digit that is most often predicted for this image is predicted less tham 95% of the time.

```{code-cell} python
predicted_class = jnp.exp(
    jnp.stack([jax.vmap(model.apply, in_axes=(None, 0))(s, X_test) for s in samples])
)
```

```{code-cell} python
num_test_samples = len(y_test)
max_predicted = [np.argmax(predicted_class[:, i, :], axis=1) for i in range(num_test_samples)]
freq_max_predicted = np.array(
    [
        (max_predicted[i] == np.argmax(np.bincount(max_predicted[i]))).sum() / num_samples
        for i in range(num_test_samples)
    ]
)
certain_mask = freq_max_predicted > 0.95
```

Let's plot a few examples where the model was very uncertain:

```{code-cell} python
most_uncertain_idx = np.argsort(freq_max_predicted)

for i in range(10):
    print(np.bincount(max_predicted[most_uncertain_idx[i]]) / 2000)
    fig = plt.figure()
    plt.imshow(X_test[most_uncertain_idx[i]].reshape(28, 28), cmap="gray")
    plt.show()
```

And now compute the average accuracy over all the samples without these uncertain predictions:

```{code-cell} python
:tags: [hide-input]

avg_accuracy = np.mean(
    [compute_accuracy(s, X_test[certain_mask], y_test[certain_mask]) for s in samples]
)
print(
    f"The average accuracy removing the samples for which the model is uncertain is {100*avg_accuracy:.3f}%"
)
```
