from typing import Dict, List, Tuple, Union

import jax.numpy as jnp
import numpy as np

Array = Union[np.ndarray, jnp.ndarray]
PyTree = Union[Dict, List, Tuple, Array]

PRNGKey = jnp.ndarray
