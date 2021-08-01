from typing import Dict, List, Tuple, Union

import jax.numpy as jnp

Array = Union[np.ndarray, jnp.DeviceArray]
PyTree = Union[Dict, List, Tuple, Array]
