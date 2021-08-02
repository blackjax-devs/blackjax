from typing import Any, Iterable, Mapping, Union

import jax.numpy as jnp
import numpy as np

Array = Union[np.ndarray, jnp.ndarray]
PyTree = Union[Array, Iterable[Array], Mapping[Any, Array]]
# It is not currently tested but we also support recursive PyTrees.
# Once recursive typing is fully supported (https://github.com/python/mypy/issues/731), we can uncomment the line below.
# PyTree = Union[Array, Iterable["PyTree"], Mapping[Any, "PyTree"]]

PRNGKey = Array
