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

def run_all_tests():
    """Run all tests and report results."""
    test = test_transpose_vector
    failed = 0
    try:
        test()
    except AssertionError as e:
        print(f"✗ {test.__name__} failed: {e}")
        failed += 1
    except Exception as e:
        print(f"✗ {test.__name__} error: {e}")
        failed += 1
    
    print(f"\nTests run: {1}, Failed: {failed}")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)