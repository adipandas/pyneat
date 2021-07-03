import numpy as np


def sigmoid(z, a_min=-60.0, a_max=60.0, scale_factor=2.5):
    """
    Sigmoid Activation function.

    Args:
        z (float or numpy.ndarray): Input.
        a_min (float): Lower limit of activation input.
        a_max (float): Upper limit of activation input.
        scale_factor (float): scaling factor for the input ``z``.

    Returns:
        float or numpy.ndarray: Output of activation

    Notes:
        * Before calculating the output using in this activation following steps are done in this function:
            - Scaling of input value: Input ``z`` is scaled to ``scale_factor*z``.
            - The value of ``scale_factor*z`` is clipped and kept in the range ``[a_min, a_max]``.
    """
    z = np.clip(scale_factor * z, a_min, a_max)
    return 1.0 / (1.0 + np.exp(-z))


def tanh(z, a_min=-60.0, a_max=60.0, scale_factor=2.5):
    """
    Tanh activation.

    Args:
        z (float or numpy.ndarray): Input.
        a_min (float): Lower limit of activation input.
        a_max (float): Upper limit of activation input.
        scale_factor (float): scaling factor for the input ``z``.

    Returns:
        float or numpy.ndarray: Output of activation

    Notes:
        * Before calculating the output using in this activation following steps are done in this function:
            - Scaling of input value: Input ``z`` is scaled to ``scale_factor*z``.
            - The value of ``scale_factor*z`` is clipped and kept in the range ``[a_min, a_max]``.
    """
    z = np.clip(scale_factor * z, a_min, a_max)
    return np.tanh(z)


def relu(z):
    """
    Rectilinear Unit.

    Args:
        z (float or numpy.ndarray): Input.

    Returns:
        float or numpy.ndarray: Output of activation.

    """
    return np.maximum(z, 0)


def abs(z):
    """
    Absolute value.

    Args:
        z (float or numpy.ndarray): Input.

    Returns:
        float or numpy.ndarray: Output of activation.

    """
    return np.abs(z)


# All possible values of activation functions
ACTIVATIONS = [sigmoid, tanh, relu, abs]
