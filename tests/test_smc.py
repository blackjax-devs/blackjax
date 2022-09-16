"""Test the generic SMC sampler"""
import chex
import jax
from absl.testing import absltest


class SMCTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.PRNGKey(42)


if __name__ == "__main__":
    absltest.main()
