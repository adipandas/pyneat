import random
import numpy as np
from gym_developmental.control_policies.pyneat.config import WEIGHT_INIT_SCALE, EDGE_DIST_COEFF, WEIGHT_MUTATE_RATE, \
    WEIGHT_MUTATE_SCALE, WEIGHT_REINIT_RATE, ACTIVE_MUTATE_RATE


class Edge(object):
    """
    Class to create edge connecting two nodes (two neurons) in the network.
    Edge is unidirectional going from neuron 1 to neuron 2.
    """
    def __init__(self, u, v, weight=None):
        """

        Parameters
        ----------
        u : key of neuron 1
        v : key of neuron 2
        weight : weight of the connection
        """

        self.weight = np.random.normal(0, WEIGHT_INIT_SCALE) if weight is None else weight
        self.uv = (u, v)
        self.active = True          # flag indicating if the edge is active (true) or not in network.

    def dist(self, other):
        """
        Distance between this edge and other edge

        Parameters
        ----------
        other : other edge

        Returns
        -------
        distance between two edges

        """

        d = (abs(self.weight - other.weight)) + float(self.active != other.active)
        d = d * EDGE_DIST_COEFF
        return d

    def mutate_(self):
        """
        Mutation of the edge

        Returns
        -------

        """
        r = random.random()

        # perturb the weight of the edge
        if r < WEIGHT_MUTATE_RATE:
            self.weight = np.clip(self.weight + np.random.normal(0, WEIGHT_MUTATE_SCALE), -30, 30)

        # re-initialize the weight of the edge
        elif r < WEIGHT_MUTATE_RATE + WEIGHT_REINIT_RATE:
            self.weight = np.random.normal(0, WEIGHT_INIT_SCALE)

        # Activate/Deactivate edge
        if random.random() < ACTIVE_MUTATE_RATE:
            self.active = random.random() < 0.5
