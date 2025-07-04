import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import jax.numpy as jnp
from vector_ops import add_vectors, subtract_vectors, multiply_vectors, dot_vectors


def test_add_vectors():
    """Test vector addition."""
    x = jnp.array([1, 2, 3])
    y = jnp.array([4, 5, 6])
    result = add_vectors(x, y)
    expected = jnp.array([5, 7, 9])
    assert jnp.allclose(result, expected), f"Expected {expected}, got {result}"
    print("✓ test_add_vectors passed!")


def test_subtract_vectors():
    """Test vector subtraction."""
    x = jnp.array([5, 7, 9])
    y = jnp.array([1, 2, 3])
    result = subtract_vectors(x, y)
    expected = jnp.array([4, 5, 6])
    assert jnp.allclose(result, expected), f"Expected {expected}, got {result}"
    print("✓ test_subtract_vectors passed!")


def test_multiply_vectors():
    """Test vector multiplication."""
    x = jnp.array([2, 3, 4])
    y = jnp.array([5, 6, 7])
    result = multiply_vectors(x, y)
    expected = jnp.array([10, 18, 28])
    assert jnp.allclose(result, expected), f"Expected {expected}, got {result}"
    print("✓ test_multiply_vectors passed!")


def test_dot_vectors():
    """Test vector dot product."""
    x = jnp.array([1, 2, 3])
    y = jnp.array([4, 5, 6])
    result = dot_vectors(x, y)
    expected = 32  # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    assert jnp.allclose(result, expected), f"Expected {expected}, got {result}"
    print("✓ test_dot_vectors passed!")


def run_all_tests():
    """Run all tests and report results."""
    tests = [test_add_vectors, test_subtract_vectors, test_multiply_vectors, test_dot_vectors]
    failed = 0
    
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} error: {e}")
            failed += 1
    
    print(f"\nTests run: {len(tests)}, Failed: {failed}")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)