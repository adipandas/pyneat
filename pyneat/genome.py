from copy import deepcopy
import random
from itertools import count

from pyneat.config import NODE_DISJOINT_COEFF, EDGE_DISJOINT_COEFF, MIN_NODE_COUNT, NODE_ADD_PROB, NODE_DEL_PROB, EDGE_ADD_PROB, EDGE_DEL_PROB
from pyneat.node import Node
from pyneat.edge import Edge
from pyneat.crossover import crossover
from pyneat.utils import creates_cycle


class Genome(object):
    """
    Genome is a direct encoding of neural network.

    Args:
        key (int): Unique id for the genome.
        input_size (int): Input size of the genome.
        output_size (int): Output size of the genome.

    Notes:
        - Genome is a representation of neural network in NEAT.
        - Nodes in each genome represent a neuron.
        - Edge in each genome represent a connection between two neurons (nodes).
    """

    node_count = count(MIN_NODE_COUNT)                 # counter to keep track of node-keys in genomes

    def __init__(self, key, input_size, output_size):

        self.key = key
        """int: Unique ID for each genome.
        """

        self.nodes = dict()
        """dict[int, Node]: Dictionary of nodes in the genome (a.k.a. network).
        """

        self.edges = dict()
        """dict[int, Edge]: Dictionary of edges in the genome (a.k.a. network).
        """

        self.input_keys = [-1 * i for i in range(1, input_size + 1)]
        """list: Input node (a.k.a. neuron) keys initialized as -ve integers
        """

        self.output_keys = [i for i in range(output_size)]
        """list: Output neuron keys.
        """

        # Initialize output nodes.
        for k in self.output_keys:
            self.nodes[k] = Node(k)

        # Initialize input to output connections.
        for u in self.input_keys:
            for v in self.output_keys:
                self.edges[(u, v)] = Edge(u, v)

    def dist(self, other):
        """
        Distance between this and other genome.

        Args:
            other (Genome): Genome from which the distance of this genome is calculated.

        Returns:
            float: distance between this and other genome

        Notes:
            - Genome is a neural network.
            - Distance between two genomes is sum of distances between their edges and nodes.
        """
        d = self._nodes_dist(other) + self._edges_dist(other)
        return d

    def _nodes_dist(self, other):
        """
        Distance between nodes/neurons of two genomes, i.e., this genome and other genome.

        Parameters:
            other (Genome): Genome from which the distance is calculated.

        Returns:
            float: distance between nodes of two genomes.

        Notes:
            - Genome is a neural network.
        """

        disjoint_nodes = 0                      # measure of how much disjoint two genomes are (disjoint-ness score)
        d = 0.0

        if self.nodes or other.nodes:           # check if any nodes/neurons are present in both genomes

            for k2 in other.nodes:              # each key in other genome's nodes
                if k2 not in self.nodes:        # if same key is not present in this genome's nodes
                    disjoint_nodes += 1

            for k1, n1 in self.nodes.items():   # for each node in this genome's nodes
                n2 = other.nodes.get(k1)
                if n2 is None:                  # if other genome's nodes does NOT contain the node
                    disjoint_nodes += 1
                else:                           # if other genome and this genome has the node with same key
                    d += n1.dist(n2)            # distance between nodes with same key from both genomes

            max_nodes = max(len(self.nodes), len(other.nodes))      # max number of nodes by comparing two genomes

            d = (d + (NODE_DISJOINT_COEFF * disjoint_nodes)) / max_nodes  # distance between nodes of two genomes

        return d

    def _edges_dist(self, other):
        """
        Distance between **edges** of this genome and the other genome.

        Parameters:
            other (Genome): Genome w.r.t. which distance is calculated between the edges.

        Returns:
            float: Distance w.r.t. edges between this and other genome.

        Notes:
            - Genome is a neural network.
        """

        d = 0.0
        disjoint_edges = 0                          # measure how disjoint two genomes are w.r.t. egdes

        if self.edges or other.edges:               # if edges are present in atleast one of the two genomes

            for k2 in other.edges:                  # for each egde in other genome
                if k2 not in self.edges:            # if corresponding egde is NOT present in this genome
                    disjoint_edges += 1             # the two genomes are disjoint

            for k1, e1 in self.edges.items():       # for each edge in this genome
                e2 = other.edges.get(k1)            # get corresponding edge in other genome

                if e2 is None:                      # if corresponding edge is NOT present in other genome
                    disjoint_edges += 1             # the two genomes are disjoint
                else:                               # if the edges with same keys exist in both genomes
                    d += e1.dist(e2)                # distance between these egdes is calculated

            max_edges = max(len(self.edges), len(other.edges))          # max number of edges by comparing both genomes

            # overall distance between this and other genome w.r.t. edges
            d = (d + (EDGE_DISJOINT_COEFF * disjoint_edges)) / max_edges

        return d

    def crossover_edges(self, other, child):
        """
        Cross over between **edges** in this genome and other genome to update a child genome

        Args:
            other (Genome): other genome
            child (Genome): child genome

        Returns:
            Genome: updated child genome after the cross-over of edges

        Notes:
            - Genome is a neural network.
        """
        for key, edge_p1 in self.edges.items():         # for each egde in edges of this (self) parent genome
            edge_p2 = other.edges.get(key)              # get the edge with same key from other parent genome
            if edge_p2 is None:                         # if other parent genome does NOT contain edge with same keys
                child.edges[key] = deepcopy(edge_p1)    # copy edge from this (self) parent genome to child genome
            else:                                       # if edge is present in both parents
                child.edges[key] = crossover(edge_p1, edge_p2, Edge(key[0], key[1]), attrs=['weight', 'active'])         # follow crossover logic
        return child

    def crossover_nodes(self, other, child):
        """
        Crossover between **nodes** in this genome and other genome to update a child genome. In node crossover, if a node is present in this genome (parent 1 or p1) as well as other genome (parent 2 or p2)
        there is addition of a new node in child genome.

        Args:
            other (Genome): other genome
            child (Genome): child genome

        Returns:
            Genome: updated child genome after the cross-over of two genomes w.r.t nodes

        Notes:
            - Genome is a neural network.
        """

        for key, node_p1 in self.nodes.items():             # for each node in this (parent 1 or p1) genome
            node_p2 = other.nodes.get(key)                  # get node with same key from other (parent 2 or p2) genome
            if node_p2 is None:                             # if parent-2 does NOT have the node with same key
                child.nodes[key] = deepcopy(node_p1)        # copy the node from this genome (p1) to child genome as is
            else:                                           # if node with same key present in both parents p1 and p2
                child.nodes[key] = crossover(node_p1, node_p2, Node(next(Genome.node_count)), attrs=['bias', 'response', 'activation', 'aggregation'])  # crossover logic: add new node in child

        return child

    def mutate_(self):
        """
        Mutate this genome.

        Notes:
            Genome mutation may include one or more of the following:
                * add node to genome
                * delete node from genome
                * add new edge to genome
                * delete edge from genome
                * mutate node-properties of nodes in this genome
                * mutate edge-properties of edges in this genome

        """
        self._mutate_add_node_()
        self._mutate_del_node_()
        self._mutate_add_edge_()
        self._mutate_del_edge_()
        self._mutate_node_properties()
        self._mutate_edge_properties()

    def _mutate_add_node_(self):
        """
        Genome mutation by addition of node.
        """
        if random.random() < NODE_ADD_PROB:
            if len(self.edges) == 0:                                        # if no edge in the genome do nothing
                return
            edge_to_split = random.choice(list(self.edges.values()))        # choose edge to split and add node
            edge_to_split.active = False                                    # deactivate edge to split

            new_node = Node(next(Genome.node_count))                        # create a new node
            self.nodes[new_node.key] = new_node                             # add the new node to the genome
            edge_u_to_new = Edge(edge_to_split.uv[0], new_node.key, weight=1.0)     # create edge to new node from node `u`
            self.edges[edge_u_to_new.uv] = edge_u_to_new                    # add edge to new node from node `u`

            edge_new_to_v = Edge(new_node.key, edge_to_split.uv[1], weight=edge_to_split.weight)   # create edge from new node to node `v`
            self.edges[edge_new_to_v.uv] = edge_new_to_v                    # add edge from new node to node `v`

    def _mutate_del_node_(self):
        """
        Genome mutation by deletion of node.
        """
        if random.random() < NODE_DEL_PROB:

            available_nodes = [k for k in self.nodes.keys() if k not in self.output_keys]   # can delete any node except output nodes

            if available_nodes:                                # if there is atleaset one available node
                del_key = random.choice(available_nodes)       # randomly choose the node to delete (node key)

                edges_to_delete = set()                        # set of edges to delete

                for k, v in self.edges.items():               # for each edge in this genome
                    if del_key in k:                          # check if `del_key` is part of this edge
                        edges_to_delete.add(k)                # delete edges which have `del_key` node in them

                for key in edges_to_delete:                   # for each edge in list of edges_to_delete
                    del self.edges[key]                       # delete edge

                del self.nodes[del_key]                       # delete node

    def _mutate_add_edge_(self):
        """
        Mutate genome by addition of edge
        """

        if random.random() < EDGE_ADD_PROB:
            possible_outputs = list(self.nodes.keys())                          # list of possible out-nodes for edge
            out_node = random.choice(possible_outputs)                          # choose a single node as output

            possible_inputs = possible_outputs + self.input_keys                # list of possible in-nodes for edge
            in_node = random.choice(possible_inputs)                            # choose single in_node for edge

            key = (in_node, out_node)                                           # the edge key to add in genome

            stop = False

            if key in self.edges:                                               # if edge already exists
                stop = True                                                     # stop edge formation

            # if both potential nodes (u, v) forming an edge fall in set of to output-neurons then stop edge formation
            if in_node in self.output_keys and out_node in self.output_keys:
                stop = True

            if creates_cycle(self.edges.keys(), in_node, out_node):             # if new edge creates a cycle in genome
                stop = True                                                     # stop edge formation

            if not stop:                                                        # if edge formation is NOT stopped
                self.edges[key] = Edge(in_node, out_node)                       # form an edge in the network

    def _mutate_del_edge_(self):
        """
        Genome mutation by deleting a randomly selected edge in the network.

        Notes:
            * This mutation happens with some probability for deletion defined in the configuration file hyperparameters ``EDGE_DEL_PROB``.
        """
        if random.random() < EDGE_DEL_PROB:
            if len(self.edges) > 0:                           # if there is atleast one edge in genome (neural network)
                key = random.choice(list(self.edges.keys()))  # choose one of the edges randomly
                del self.edges[key]                           # delete the chosen edge from genome

    def _mutate_edge_properties(self):
        """
        Genome mutation by mutating properties of each edge.
        """
        for edge in self.edges.values():            # for each edge in network
            edge.mutate_()                          # mutate edge

    def _mutate_node_properties(self):
        """
        Genome mutation by mutating properties of each node.
        """
        for node in self.nodes.values():            # for each node in network
            node.mutate_()                          # mutate node
