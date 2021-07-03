import random
import numpy as np

from pyneat.config import BIAS_INIT_SCALE, NODE_DIST_COEFF, BIAS_MUTATE_RATE, BIAS_MUTATE_SCALE, BIAS_REINIT_RATE, ACTIVATION_MUTATE_RATE

from pyneat.activations import sigmoid, ACTIVATIONS


class Node(object):
    """
    Class to create a node/neuron in a network.

    Args:
        key (int): unique id of the node.

    Notes:
        * Neuron and Node are the same thing. These two terms are used interchangablely.
    """
    def __init__(self, key):
        self.key = key
        """int: Unique ID of the node.
        """

        self.bias = np.random.normal(0, BIAS_INIT_SCALE)
        """float: Bias of the neuron."""

        self.response = 1.0
        """float: Default is ``1.0``.
        """

        self.activation = sigmoid
        """callable: Activation function of the node. Default is ``sigmoid``.
        """

        self.aggregation = np.sum
        """callable: Input aggregation function for the node. Default is ``np.sum``.
        """

    def dist(self, other):
        """
        Calculate distance between two nodes.

        Args:
            other (Node): other node w.r.t. which distance is calculated.

        Returns:
            float: Distance between `self` (this) node and other node.

        Notes:
            * Calculation of distance between two nodes (a.k.a. neurons) involves:
                - Magnitude of difference between ``bias`` values of each neuron.
                - ``activation`` attribute comparison of two neurons.
                - ``aggregation`` attribute comparison of two neurons.
        """

        d = abs(self.bias - other.bias)
        if self.activation != other.activation:
            d += 1.0
        if self.aggregation != other.aggregation:
            d += 1.0
        return NODE_DIST_COEFF * d

    def mutate_(self):
        """
        Mutation of node
        """

        # mutate bias value of node
        r = random.random()
        if r < BIAS_MUTATE_RATE:
            self.bias = np.clip(self.bias + np.random.normal(0, BIAS_MUTATE_SCALE), -30, 30)

        # re-initialize bias value of node
        elif r < BIAS_MUTATE_RATE + BIAS_REINIT_RATE:
            self.bias = np.random.normal(0, BIAS_INIT_SCALE)

        # mutation of activation function of node
        if random.random() < ACTIVATION_MUTATE_RATE:
            self.activation = random.choice(ACTIVATIONS)
