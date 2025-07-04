import jax.numpy as jnp


def add_vectors(x, y):
    """Add two JAX arrays element-wise."""
    return x + y


def subtract_vectors(x, y):
    """Subtract two JAX arrays element-wise."""
    # Intentional bug: using + instead of -
    return x + y  # BUG: This should be x - y


def multiply_vectors(x, y):
    """Multiply two JAX arrays element-wise."""
    return x * y