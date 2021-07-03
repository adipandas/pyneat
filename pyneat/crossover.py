import random


def crossover(obj1, obj2, obj_new, attrs):
    """
    crossover logic to do the crossover of two parent objects (viz. `obj1` and `obj2`)
    to update child object (i.e. obj_new) with some given attributes `attrs`.
    Crossover logic sets each attribute in list of attributes for the child object with values taken from either of the
    parent objects with probability of 50%.

    Args:
        obj1 (pyneat.genome.Genome or pyneat.node.Node or pyneat.edge.Edge): parent object 1
        obj2 (pyneat.genome.Genome or pyneat.node.Node or pyneat.edge.Edge): parent object 2
        obj_new (pyneat.genome.Genome or pyneat.node.Node or pyneat.edge.Edge): child object
        attrs (list[str]): list of attributes to update in child object

    Returns:
        pyneat.genome.Genome or pyneat.node.Node or pyneat.edge.Edge: Updated child object after cross over
    """

    for attr in attrs:                                      # for each attribute in list of input attributes
        if random.random() > 0.5:                           # choose one of the parents (obj1) with 50% chance
            setattr(obj_new, attr, getattr(obj1, attr))     # set the attribute of child with parent obj1 value
        else:                                               # choose other parent (obj2) with 50% chance
            setattr(obj_new, attr, getattr(obj2, attr))     # set the attribute of child with parent obj2 value

    return obj_new
