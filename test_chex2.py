import jax
import chex
import pytest

class TestChex(chex.TestCase):
    @chex.assert_max_traces(n=1)
    @jax.jit
    def method_to_trace(self, x):
        return x + 1

    def test_one(self):
        self.method_to_trace(1)
        self.method_to_trace(2)
        self.method_to_trace(3)
