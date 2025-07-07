import jax.numpy as jnp


def add_vectors(x, y):
    """Add two JAX arrays element-wise."""
    return x + y


def subtract_vectors(x, y):
    """Subtract two JAX arrays element-wise."""
    # Intentional bug: using + instead of -
    return x - y  # BUG: This should be x - y


def multiply_vectors(x, y):
    """Multiply two JAX arrays element-wise."""
    return x * y


def transpose_vector(x):
    """Transpose a JAX array (vector or matrix).
    
    For 1D arrays, converts them to column vectors (shape (n, 1)).
    For 2D+ arrays, performs standard matrix transpose.

    Args:
        x: A JAX array to transpose

    Returns:
        The transposed array. For 1D input, returns a column vector.
    """
    if x.ndim == 1:
        # Convert 1D array to column vector
        return x.reshape(-1, 1)
    else:
        # Standard transpose for 2D+ arrays
        return jnp.transpose(x)

print(jnp.array([1,2,3]).shape)
print(transpose_vector(jnp.array([1,2,3])).shape)
print(transpose_vector(jnp.array([1,2,3])))