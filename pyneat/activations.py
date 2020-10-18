import math
import numpy as np


def sigmoid(z):
    """

    Parameters
    ----------
    z : input

    Returns
    -------
    value of sigmoid function for input z

    """
    z = max(-60.0, min(60.0, 2.5 * z))
    return 1.0 / (1.0 + math.exp(-z))


def tanh(z):
    """
    Hyperbolic Tangent function

    Parameters
    ----------
    z : input

    Returns
    -------
    value of hyperbolic tangent

    """
    z = max(-60.0, min(60.0, 2.5 * z))
    return np.tanh(z)


def relu(z):
    """
    Rectilinear Unit

    Parameters
    ----------
    z : input to rectilinear unit

    Returns
    -------
    value of relu function

    """
    return np.maximum(z, 0)


def abs(z):
    """
    Absolue value

    Parameters
    ----------
    z : input

    Returns
    -------
    absolute value of input

    """
    return np.abs(z)


# All possible values of activation functions
ACTIVATIONS = [sigmoid, tanh, relu, abs]
