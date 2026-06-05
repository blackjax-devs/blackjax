import jax
import chex
import pytest

class TestChex(chex.TestCase):
    @chex.assert_max_traces(n=2)
    def test_one(self):
        def f(x): return x + 1
        j = jax.jit(f)
        j(1)
        j(2)

