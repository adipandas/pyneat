
class EvalNode(object):
    """
    Class to help in node/neuron evaluation in phenotype
    """

    def __init__(self, node_id, activation, aggregation, bias, incoming_connections):
        """

        Parameters
        ----------
        node_id : id of node
        activation : activation function of node
        aggregation : aggregation function of node
        bias : bias of node
        incoming_connections : incoming connections to node
        """

        self.node_id = node_id
        self.activation = activation
        self.aggregation = aggregation
        self.bias = bias
        self.incoming_connections = incoming_connections


class Network(object):
    """
    Class to help in neural-network creation from genome.
    Something like genotype to phenotype module.
    """

    def __init__(self, input_keys, output_keys, eval_nodes):
        """

        Parameters
        ----------
        input_keys : keys of input nodes
        output_keys : keys of output nodes
        eval_nodes : list of nodes to evaluate
        """
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.eval_nodes = eval_nodes

    @classmethod
    def make_network(cls, genome):
        """

        Parameters
        ----------
        genome : genome used for creation of network-phenotype

        Returns
        -------
        neural network phenotype of genome
        """

        edges = [edge.uv for edge in genome.edges.values() if edge.active]       # active edges in genome

        required_nodes = cls._required_nodes(edges,
                                             genome.input_keys,
                                             genome.output_keys)                 # nodes required for network creation

        layers = cls._make_layers(required_nodes,
                                  edges,
                                  genome.input_keys)                            # create network layers

        eval_nodes = cls._make_eval_nodes(layers, edges, genome)                # create evaluation nodes

        net = cls(genome.input_keys, genome.output_keys, eval_nodes)            # create network

        return net

    @staticmethod
    def _required_nodes(edges, input_keys, output_keys):
        """
        Identify nodes that are required for output by working backwards from output nodes.

        Parameters
        ----------
        edges : list of active edges in the network phenotype
        input_keys :  list of input keys
        output_keys : list of output keys

        Returns
        -------
        set of nodes required for network creation

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
        Make layers for the network

        Parameters
        ----------
        required_nodes : required nodes
        edges : edges in the network
        input_keys : input keys

        Returns
        -------
        layers in the network. The returned value is a list containing set of nodes/neurons in each layer of network.

        """

        layers = []
        seen = set(input_keys)
        while True:

            # Find candidate nodes for the next layer that connect a seen node to an unseen node.
            candidates = set(v for (u, v) in edges if u in seen and v not in seen)

            # Keep only required nodes whose entire input set is contained in seen.
            layer = set()
            for w in candidates:                                                         # for node in candidate
                # if node is requried and its input is in seen
                if w in required_nodes and all(u in seen for (u, v) in edges if v == w):
                    layer.add(w)

            if not layer:                               # if layer is empty
                break                                   # break the loop

            layers.append(layer)                        # add layer [i.e.set of nurons/nodes] to list of layers
            seen = seen.union(layer)                    # add the neurons from layer to the set of seen neurons

        return layers

    @staticmethod
    def _make_eval_nodes(layers, edges, genome):
        """

        Parameters
        ----------
        layers : layers in neural network
        edges : edges in neural network connecting nodes/neurons
        genome : genome or genotype of neural network.

        Returns
        -------
        list of EvalNodes or phenotype-neurons

        """

        eval_nodes = []

        for layer in layers:                                        # for each layer of nodes in network
            for node in layer:                                      # for each node in layer

                # incomming edges to node
                incoming_conns = [(u, genome.edges[u, v].weight) for u, v in edges if v == node]

                eval_node = EvalNode(node_id=node,
                                     activation=genome.nodes[node].activation,
                                     aggregation=genome.nodes[node].aggregation,
                                     bias=genome.nodes[node].bias,
                                     incoming_connections=incoming_conns)

                eval_nodes.append(eval_node)
        return eval_nodes

    def forward(self, x):
        """
        forward pass in the network

        Parameters
        ----------
        x : input

        Returns
        -------
        output-vector (list) of neural network for the given input vector

        """

        values = {k: 0.0 for k in self.input_keys + self.output_keys}

        for k, v in zip(self.input_keys, x):
            values[k] = v

        for eval_node in self.eval_nodes:                       # for each neuron in network
            node_inputs = []

            for i, w in eval_node.incoming_connections:         # for in_node_id, edge_wt in incoming connection to node
                node_inputs.append((values[i] * w))             # collect input to this node from in_node_id

            agg = eval_node.aggregation(node_inputs)            # aggregation of all in_node_ids to this node

            values[eval_node.node_id] = eval_node.activation(eval_node.bias + agg)    # activation of this neuron/node

        outputs = [values[n] for n in self.output_keys]             # values of neurons
        return outputs
