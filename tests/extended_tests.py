import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import jax.numpy as jnp
from vector_ops import transpose_vector


def test_transpose_vector():
    """Test vector addition."""
    x = jnp.array([1, 2, 3])
    result = transpose_vector(x)
    expected = jnp.array([[1],[2],[3]])
    assert jnp.allclose(result, expected), f"Expected {expected}, got {result}"
    print("✓ test_add_vectors passed!")
