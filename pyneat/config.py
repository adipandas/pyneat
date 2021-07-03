"""
Configuration file for NEAT
"""

MIN_NODE_COUNT = 10                 # make sure this is larger than the output dimension
ELITISM = 2                         # elite population count
CUTOFF_PCT = 0.2                    # cutoff percentage for choosing population from past generation in species
MIN_FITNESS_RANGE = 1.0             # Minimum range of fitness, difference between max and min fitnesses of Genomes, Note: 1.0 is abitrarily chosen
MIN_SPECIES_SIZE = 2                # minimum size of species/partition population
COMPATIBILITY_THRESHOLD = 3.0       # max distance between two genomes to potentially be in same species
NODE_DIST_COEFF = 0.5               # scaling of distance between two nodes
NODE_DISJOINT_COEFF = 1.0           # scaling of disjoint-ness of two genomes w.r.t. nodes
EDGE_DIST_COEFF = 0.5               # scaling of distance between two edges
EDGE_DISJOINT_COEFF = 1.0           # scaling of disjoint-ness of two genomes w.r.t. edges
NODE_ADD_PROB = 0.3                 # probability of adding node during mutation in genome
NODE_DEL_PROB = 0.2                 # probability of deleting node during mutation in genome
EDGE_ADD_PROB = 0.3                 # probability of adding edge during mutation in genome
EDGE_DEL_PROB = 0.2                 # probability of deleting edge during mutation in genome
WEIGHT_MUTATE_RATE = 0.8            # probability of mutation of weight of edge in the network (genome)
WEIGHT_REINIT_RATE = 0.1            # probability of (re)initailization of weight of edge in the network (genome)
ACTIVE_MUTATE_RATE = 0.01           # probability to activate/deactivate an edge in a network (genome)
BIAS_MUTATE_RATE = 0.7              # probability of mutation of bias-weight of a node in a network (genome)
BIAS_REINIT_RATE = 0.1              # probability of (re)initailization of bias value of a node in a network (genome)
RESPONSE_MUTATE_RATE = 0.0
ACTIVATION_MUTATE_RATE = 0.20       # probability of mutation of activation function of a node in a network (genome)
WEIGHT_MUTATE_SCALE = 0.5           # std of weight mutation of an edge connecting two neurons in network (genome)
BIAS_MUTATE_SCALE = 0.5             # std of bias value for mutation of bias
WEIGHT_INIT_SCALE = 1.0             # std of weight of edge connecting two neurons in the network (genome)
BIAS_INIT_SCALE = 1.0               # std for bias value intialization
