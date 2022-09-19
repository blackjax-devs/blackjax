---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.0
kernelspec:
  display_name: blackjax
  language: python
  name: python3
---
# MNIST Digit Recognition With a 3-Layer Perceptron

This example is inspired form [this notebook](https://github.com/jeremiecoullon/SGMCMCJax/blob/master/docs/nbs/BNN.ipynb) in the SGMCMCJax repository. We try to use a 3-layer neural network to recognise the digits in the MNIST dataset.

```{code-cell} ipython3
import jax
import jax.nn as nn
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np
```

## Data Preparation

We download the MNIST data using `tensorflow-datasets`:

```{code-cell} ipython3
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

```{code-cell} ipython3
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
X_test, y_test, N_test = prepare_data(data_train)
```

## Model: 3-layer Perceptron

We will use a very simple (bayesian) neural network in this example: A MLP with gaussian priors on the weights. We first need a function that computes the model's logposterior density given the data and the current values of the parameters. If we note $X$ the array that represents an image and $y$ the array such that $y_i = 0$  if the image is in category $i$, $y_i=1$ otherwise, the model can be written as:

\begin{align*}
  \boldsymbol{p} &= \operatorname{NN}(X)\\
  \boldsymbol{y} &\sim \operatorname{Categorical}(\boldsymbol{p})
\end{align*}

```{code-cell} ipython3
def predict_fn(parameters, X):
    """Returns the probability for the image represented by X
    to be in each category given the MLP's weights vakues.

    """
    activations = X
    for W, b in parameters[:-1]:
        outputs = jnp.dot(W, activations) + b
        activations = nn.softmax(outputs)

    final_W, final_b = parameters[-1]
    logits = jnp.dot(final_W, activations) + final_b
    return nn.log_softmax(logits)


def logprior_fn(parameters):
    """Compute the value of the log-prior density function."""
    logprob = 0.0
    for W, b in parameters:
        logprob += jnp.sum(stats.norm.logpdf(W))
        logprob += jnp.sum(stats.norm.logpdf(b))
    return logprob


def loglikelihood_fn(parameters, data):
    """Categorical log-likelihood"""
    X, y = data
    return jnp.sum(y * predict_fn(parameters, X))


def compute_accuracy(parameters, X, y):
    """Compute the accuracy of the model.

    To make predictions we take the number that corresponds to the highest probability value.
    """
    target_class = jnp.argmax(y, axis=1)
    predicted_class = jnp.argmax(
        jax.vmap(predict_fn, in_axes=(None, 0))(parameters, X), axis=1
    )
    return jnp.mean(predicted_class == target_class)
```

## Sample From the Posterior Distribution of the Perceptron's Weights

Now we need to get initial values for the parameters, and we simply sample from their prior distribution:

```{code-cell} ipython3
def init_parameters(rng_key, sizes):
    """

    Parameter
    ----------
    rng_key
        PRNGKey used by JAX to generate pseudo-random numbers
    sizes
        List of size for the subsequent layers. The first size must correspond
        to the size of the input data and the last one to the number of
        categories.

    """
    num_layers = len(sizes)
    keys = jax.random.split(rng_key, num_layers)
    return [
        init_layer(rng_key, m, n) for rng_key, m, n in zip(keys, sizes[:-1], sizes[1:])
    ]


def init_layer(rng_key, m, n, scale=1e-2):
    """Initialize the weights for a single layer."""
    key_W, key_b = jax.random.split(rng_key)
    return (scale * jax.random.normal(key_W, (n, m))), scale * jax.random.normal(
        key_b, (n,)
    )
```

We now sample from the model's posteriors. We discard the first 1000 samples until the sampler has reached the typical set, and then take 2000 samples. We record the model's accuracy with the current values every 100 steps.

```{code-cell} ipython3
%%time

import blackjax
from blackjax.sgmcmc.gradients import grad_estimator

data_size = len(y_train)
batch_size = int(0.01 * data_size)
layer_sizes = [784, 100, 10]
step_size = 5e-5
num_warmup = 1000
num_samples = 2000

# Batch the data
rng_key = jax.random.PRNGKey(1)
batches = batch_data(rng_key, (X_train, y_train), batch_size, data_size)

# Build the SGLD kernel with a constant learning rate
grad_fn = grad_estimator(logprior_fn, loglikelihood_fn, data_size)
sgld = blackjax.sgld(grad_fn, step_size)

# Set the initial state
init_positions = init_parameters(rng_key, layer_sizes)
state = sgld.init(init_positions, next(batches))

# Sample from the posterior
accuracies = []
samples = []
steps = []
for step in range(num_samples + num_warmup):
    _, rng_key = jax.random.split(rng_key)
    batch = next(batches)
    state = sgld.step(rng_key, state, batch)
    if step % 100 == 0:
        accuracy = compute_accuracy(state.position, X_test, y_test)
        accuracies.append(accuracy)
        steps.append(step)
    if step > num_warmup:
        samples.append(state.position)
```

Let us plot the accuracy at different points in the sampling process:

```{code-cell} ipython3
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
plt.plot()
```

```{code-cell} ipython3
print(f"The average accuracy in the sampling phase is {np.mean(accuracies[10:]):.2f}")
```

Which is not a bad accuracy at all for such a simple model and after only 1000 steps! Remember though that we draw samples from the posterior distribution of the digit probabilities; we can thus use this information to filter out examples for which the model is "unsure" of its prediction.

Here we will say that the model is unsure of its prediction for a given image if the digit that is most often predicted for this image is predicted less tham 95% of the time.

```{code-cell} ipython3
predicted_class = np.exp(
    np.stack([jax.vmap(predict_fn, in_axes=(None, 0))(s, X_test) for s in samples])
)
```

```{code-cell} ipython3
max_predicted = [np.argmax(predicted_class[:, i, :], axis=1) for i in range(60000)]
freq_max_predicted = np.array(
    [
        (max_predicted[i] == np.argmax(np.bincount(max_predicted[i]))).sum() / 2000
        for i in range(60000)
    ]
)
certain_mask = freq_max_predicted > 0.95
```

Let's plot a few examples where the model was very uncertain:

```{code-cell} ipython3
most_uncertain_idx = np.argsort(freq_max_predicted)

for i in range(10):
    print(np.bincount(max_predicted[most_uncertain_idx[i]]) / 2000)
    fig = plt.figure()
    plt.imshow(X_test[most_uncertain_idx[i]].reshape(28, 28), cmap="gray")
    plt.show()
```

And now compute the average accuracy over all the samples without these uncertain predictions:

```{code-cell} ipython3
avg_accuracy = np.mean(
    [compute_accuracy(s, X_test[certain_mask], y_test[certain_mask]) for s in samples]
)
```

```{code-cell} ipython3
print(
    f"The average accuracy removing the samples for which the model is uncertain is {avg_accuracy:.3f}"
)
```
