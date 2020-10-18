import random
import numpy as np

from gym_developmental.control_policies.pyneat.config import BIAS_INIT_SCALE, NODE_DIST_COEFF, BIAS_MUTATE_RATE, \
    BIAS_MUTATE_SCALE, BIAS_REINIT_RATE, ACTIVATION_MUTATE_RATE

from gym_developmental.control_policies.pyneat.activations import sigmoid, ACTIVATIONS


class Node(object):
    """
    Class to create a node/neuron in a network
    """
    def __init__(self, key):
        """

        Parameters
        ----------
        key : unique id of the node
        """

        self.bias = np.random.normal(0, BIAS_INIT_SCALE)        # bias of the network
        self.response = 1.0
        self.activation = sigmoid                               # activation function of the node
        self.aggregation = np.sum                               # agregation function of the node
        self.key = key                                          # unique id/key of the node

    def dist(self, other):
        """
        calculate distance between two nodes

        Parameters
        ----------
        other : other node wrt which distance is to be calculated

        Returns
        -------
        distance between `self`(this) node and other node.
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

        Returns
        -------

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
