import numpy as np

def scalar_or_vector_to_vector(scalar_or_vector, length):
    """Convert a scalar or a vector to a vector of specified length.

    Args:
        scalar_or_vector: A scalar or a vector.
        length: The length of the vector.

    Returns:
        A vector of specified length.
    """
    if isinstance(scalar_or_vector, (int, float)):
        return np.ones(length) * scalar_or_vector
    else:
        assert len(scalar_or_vector) == length, "The length of the vector is not equal to the specified length!"
        return scalar_or_vector