import random
import numpy as np

from pyneat.config import WEIGHT_INIT_SCALE, EDGE_DIST_COEFF, WEIGHT_MUTATE_RATE, WEIGHT_MUTATE_SCALE, WEIGHT_REINIT_RATE, ACTIVE_MUTATE_RATE


class Edge(object):
    """
    Class to create **directed** edge connecting two nodes (a.k.a. neurons) in the network. Edge is ``unidirectional`` going from ``neuron 1`` to ``neuron 2``.

    Args:
        u (int): key of neuron 1
        v (int): key of neuron 2
        weight (float): Weight of the connection. Default is ``None``.

    """
    def __init__(self, u, v, weight=None):
        self.weight = np.random.normal(0, WEIGHT_INIT_SCALE) if weight is None else weight
        self.uv = (u, v)
        self.active = True          # flag indicating if the edge is active (True) or not in network.

    def dist(self, other):
        """
        Distance between this edge and other edge.

        Args:
            other (Edge): other edge

        Returns:
            float: distance between two edges
        """

        d = (abs(self.weight - other.weight)) + float(self.active != other.active)
        d = d * EDGE_DIST_COEFF
        return d

    def mutate_(self):
        """
        Mutation of the edge.

        Notes:
            - Edge can mutate in following manner:
                * edge weight can be perturbed.
                * edge weight can reinitialize.
                * edge can activate or deactivate.
        """
        r = random.random()

        if r < WEIGHT_MUTATE_RATE:  # perturb the weight of the edge
            perturbed_wt: float = self.weight + np.random.normal(0.0, WEIGHT_MUTATE_SCALE)
            self.weight = np.clip(perturbed_wt, -30., 30.)
        elif r < (WEIGHT_MUTATE_RATE + WEIGHT_REINIT_RATE):  # re-initialize the weight of the edge
            self.weight = np.random.normal(0, WEIGHT_INIT_SCALE)

        if random.random() < ACTIVE_MUTATE_RATE:    # Activate/Deactivate edge
            self.active = random.random() < 0.5
