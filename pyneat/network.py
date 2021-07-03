from typing import Callable, Union, Iterable


class EvalNode(object):
    """
    Class to help in node (a.k.a. neuron) evaluation in phenotype.

    Args:
        node_id (int): id of node.
        activation (Callable): activation function of node
        aggregation (Callable): aggregation function of node
        bias (float): bias of node
        incoming_connections (list[Tuple[int, int]]): incoming connections to node
    """

    def __init__(self, node_id, activation: Callable[[float], float], aggregation: Callable[[Union[list, tuple]], float], bias, incoming_connections):

        self.node_id = node_id
        self.activation = activation
        self.aggregation = aggregation
        self.bias = bias
        self.incoming_connections = incoming_connections


class NeuralNetwork(object):
    """
    Class to help in neural-network creation from genome. Something like ``genotype to phenotype`` module.

    Args:
        input_keys (list or tuple or Iterable): keys of input nodes
        output_keys (list or tuple or Iterable): keys of output nodes
        eval_nodes (list or tuple or Iterable): list of nodes to evaluate
    """

    def __init__(self, input_keys, output_keys, eval_nodes):
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.eval_nodes = eval_nodes

    @classmethod
    def make_network(cls, genome):
        """
        Make a network.

        Args:
            genome (pyneat.genome.Genome): genome used for creation of network-phenotype

        Returns:
            NeuralNetwork: neural network phenotype of genome
        """

        edges = [edge.uv for edge in genome.edges.values() if edge.active]                                # active edges in genome

        required_nodes = cls._required_nodes(edges, genome.input_keys, genome.output_keys)                # nodes required for network creation

        layers = cls._make_layers(required_nodes, edges, genome.input_keys)     # create network layers

        eval_nodes = cls._make_eval_nodes(layers, edges, genome)                # create evaluation nodes

        net = cls(genome.input_keys, genome.output_keys, eval_nodes)            # create network

        return net

    @staticmethod
    def _required_nodes(edges, input_keys, output_keys):
        """
        Identify nodes that are required for output by working backwards from output nodes.

        Args:
            edges (list or tuple or Iterable): list of active edges in the network phenotype
            input_keys (list or tuple or Iterable):  list of input keys
            output_keys (list or tuple or Iterable): list of output keys

        Returns:
            set[int]: set of nodes required for network creation and that are between input and output nodes.

        """

        required = set(output_keys)         # all output nodes are required and these form output layer of network
        seen = set(output_keys)

        while True:

            # go in reverse order begin from second last layer and go towards first
            layer = set(u for (u, v) in edges if v in seen and u not in seen)     # get nodes connected to output nodes

            if not layer:                                                         # if the layer is empty
                break                                                             # break

            layer_nodes = set(u for u in layer if u not in input_keys)            # nodes which are not inputs

            if not layer_nodes:                                                   # if no nodes in layer_nodes
                break                                                             # break

            required = required.union(layer_nodes)                                # add nodes as required nodes

            seen = seen.union(layer)                                              # seen nodes in this iteration

        return required

    @staticmethod
    def _make_layers(required_nodes, edges, input_keys):
        """
        Make layers for the neural network.

        Args:
            required_nodes (set[int]): IDs of required nodes
            edges (list[Tuple[int, int]] or Iterable[Tuple[int, int]]): edges in the network
            input_keys (list[int] or Iterable[int]): IDs of input nodes.

        Returns:
            list[set[int]]: Layers in the network. Each list element contains set of nodes IDs corresponding to that layer of neural network.

        Notes:
            * Nodes represent neurons in the network.
        """

        layers = []
        seen = set(input_keys)
        while True:
            candidates = set(v for (u, v) in edges if u in seen and v not in seen)  # candidate nodes for next layer that connect a seen node to an unseen node.

            # Keep only required nodes whose entire input set is contained in seen.
            layer = set()
            for w in candidates:                         # for node in set of candidates
                if w in required_nodes and all(u in seen for (u, v) in edges if v == w):  # if node is requried and its input is in seen
                    layer.add(w)

            if not layer:                               # if layer is empty
                break                                   # break the loop

            layers.append(layer)                        # add layer [i.e.set of nurons/nodes] to list of layers
            seen = seen.union(layer)                    # add the neurons from layer to the set of seen neurons

        return layers

    @staticmethod
    def _make_eval_nodes(layers, edges, genome):
        """
        Make the evaluation nodes from nodes and edges information from genome. The evaluation nodes are phenotype or realization of neurons in the neural network.

        Args:
            layers list[set[int]]: Layers in neural network.
            edges (list[Tuple[int, int]] or Iterable[Tuple[int, int]]): Edges in neural network connecting nodes.
            genome (pyneat.genome.Genome): Genome or Genotype of neural network.

        Returns:
            list[EvalNode]: list of EvalNodes or phenotype-neurons to evaluate during each iteration.

        Notes:
            * Nodes represent neurons in the network.
        """

        eval_nodes = []

        for layer in layers:                                                                            # for each layer of nodes in network
            for node in layer:                                                                          # for each node in layer
                incoming_conns = [(u, genome.edges[u, v].weight) for u, v in edges if v == node]        # incomming edges to node
                eval_node = EvalNode(node_id=node, activation=genome.nodes[node].activation, aggregation=genome.nodes[node].aggregation, bias=genome.nodes[node].bias, incoming_connections=incoming_conns)
                eval_nodes.append(eval_node)
        return eval_nodes

    def forward(self, x):
        """
        Forward pass of the neural network.

        Args:
            x (list[float or int] or tuple[float or int] or Iterable[float or int] or numpy.ndarray): Input to the Neural network.

        Returns:
            list[float]: Neural network output for the given input vector.
        """

        values = {k: 0.0 for k in self.input_keys + self.output_keys}           # initailize ``values`` dict with input and output neuron values.

        for k, v in zip(self.input_keys, x):                                          # populate input neuron ``values`` with given input
            values[k] = v

        for eval_node in self.eval_nodes:                                             # for each neuron in network
            node_inputs = []
            for i, w in eval_node.incoming_connections:                               # for in_node_id, edge_wt in incoming connection to node
                node_inputs.append((values[i] * w))                                   # collect input to this node from in_node_id
            agg = eval_node.aggregation(node_inputs)                                  # aggregation of all in_node_ids to this node
            values[eval_node.node_id] = eval_node.activation(eval_node.bias + agg)    # activation of this neuron/node

        outputs = [values[n] for n in self.output_keys]             # values of neurons
        return outputs

    def __call__(self, x):
        return self.forward(x)
