from typing import Any, Iterable, Mapping, Union

import jax
from chex import Array

#: JAX PyTrees
PyTree = Union[Array, Iterable["PyTree"], Mapping[Any, "PyTree"]]

#: JAX PRNGKey
PRNGKey = jax.random.PRNGKeyArray
