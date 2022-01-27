from typing import Any, Iterable, Mapping, Union

import jax._src.prng as prng
import jax.numpy as jnp
import numpy as np

#: JAX or Numpy array
Array = Union[np.ndarray, jnp.ndarray]

#: JAX PyTrees
PyTree = Union[Array, Iterable[Array], Mapping[Any, Array]]
# It is not currently tested but we also support recursive PyTrees.
# Once recursive typing is fully supported (https://github.com/python/mypy/issues/731), we can uncomment the line below.
# PyTree = Union[Array, Iterable["PyTree"], Mapping[Any, "PyTree"]]

#: JAX PRNGKey
PRNGKey = prng.PRNGKeyArray
