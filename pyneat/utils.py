import random
from copy import deepcopy
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def crossover(obj1, obj2, obj_new, attrs):
    """
    crossover logic to do the crossover of two parent objects (viz. `obj1` and `obj2`)
    to update child object (i.e. obj_new) with some given attributes `attrs`.
    Crossover logic sets each attribute in list of attributes for the child object with values taken from either of the
    parent objects with probability of 50%.

    Parameters
    ----------
    obj1 : parent object 1
    obj2 : parent object 2
    obj_new : child object
    attrs : list of attributes to update in child object

    Returns
    -------
    updated child object after cross over

    """

    for attr in attrs:                                      # for each attribute in list of input attributes
        if random.random() > 0.5:                           # choose one of the parents (obj1) with 50% chance
            setattr(obj_new, attr, getattr(obj1, attr))     # set the attribute of child with parent obj1 value
        else:                                               # choose other parent (obj2) with 50% chance
            setattr(obj_new, attr, getattr(obj2, attr))     # set the attribute of child with parent obj2 value

    return obj_new


def creates_cycle(edges, u, v):
    """
    Check if the edge (u, v) form a cycle in genome (neural network).
    Check if there is a path from node `v` to node `u` in network
    Note: Genome is a DIRECTED graph.

    Parameters
    ----------
    edges : list of edges in the genome
    u : in-node for potential edge
    v : out-node for potential edge

    Returns
    -------
    bool if true, indicates the edge u, v forms a cycle in genome (neural network).

    """

    if u == v:                          # if there is an edge from node to itself. (v->u and u->v)
        return True                     # this edge creates a cycle

    graph = defaultdict(list)

    # create adjacency list of each node in directed graph (genome)
    for i, j in edges:
        graph[i].append(j)              # dict with [key(graph_node)-value(list of adjacent_nodes)]

    if v not in graph:                  # if edge `v` (destination node for edge (u, v)) is not in keys of `graph` dict
        return False                    # edge (u, v) does not form a cycle

    seen = set()
    queue = [v]

    while len(queue) > 0:
        curr = queue[0]
        queue = queue[1:]
        seen.add(curr)

        if curr == u:                       # u->...->v and v->...->u edge formation
            return True                     # forms a cycle

        for child in graph[curr]:
            if child not in seen:
                queue.append(child)
    return False


def visualize(candidate, stats):
    """
    Visualizing the phenotype/neural network

    Parameters
    ----------
    candidate : neural network candidate
    stats : dict containing the statistics of genome

    Returns
    -------

    """

    # -------- plot statistics and show ------------
    plt.plot(stats['max'], label='max')
    plt.plot(stats['mean'], label='mean')

    plt.legend()

    plt.show()
    # ------------------------------------------------

    # plot network structure

    V = candidate.nodes             # vertices/nodes in graph
    E = candidate.edges             # edges in graph

    def a_names(a):
        """
        Name of activation function

        Parameters
        ----------
        a : activation

        Returns
        -------
        string-name of activation function

        """
        if 'sigmoid' in a.__name__:
            return 'σ()'
        return a.__name__ + '()'

    def color(v):
        """

        Parameters
        ----------
        v : nodes/vertex in graph

        Returns
        -------
        color of vertex, green for input, red for output, blue for hidden

        """
        if v in candidate.input_keys:
            return "green"
        if v in candidate.output_keys:
            return "red"
        return "blue"

    v_label = {k: '%s' % (a_names(v.activation)) for k, v in V.items()}     # mapping from vertex_id to its label
    e_label = {k: np.round(v.weight, 3) for k, v in E.items()}              # mapping from edge to edge weight

    # create directional graph for neural network
    G = nx.DiGraph()
    G.add_nodes_from(V)
    G.add_edges_from(E)

    paths = nx.all_pairs_shortest_path_length(G)
    for v, ps in list(paths):
        ps = deepcopy(ps)
        if all([u not in ps for u in candidate.output_keys]):  # no path to the output layer
            G.remove_node(v)
            v_label_ = deepcopy(v_label)
            if v in v_label_:
                del v_label[v]
            e_label_ = deepcopy(e_label)
            for i, j in e_label_:
                if i == v or j == v:
                    del e_label[i, j]

    pos = nx.drawing.nx_agraph.pygraphviz_layout(G, prog='dot')

    pos = {k: (v[0], -v[1]) for k, v in pos.items()}

    nx.draw(G, pos=pos, node_color=[color(k) for k in G.nodes])
    nx.draw_networkx_labels(G, labels=v_label, pos=pos)
    nx.draw_networkx_edge_labels(G, edge_labels=e_label, pos=pos)

    plt.show()
