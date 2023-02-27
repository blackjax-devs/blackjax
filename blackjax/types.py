from typing import Any, Iterable, Mapping, Union

import jax
import jax.numpy as jnp
import numpy as np

#: JAX or Numpy array
Array = Union[np.ndarray, jnp.ndarray]

#: JAX PyTrees
PyTree = Union[Array, Iterable["PyTree"], Mapping[Any, "PyTree"]]

#: JAX PRNGKey
PRNGKey = jax.random.PRNGKeyArray
