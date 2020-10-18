from gym_developmental.control_policies.pyneat.genome import Genome


def new_child(p1, p2, f1, f2, child_id):
    """

    Parameters
    ----------
    p1 : Genome parent 1
    p2 : Genome parent 2
    f1 : fitness value of parent 1
    f2 : fitness value of parent 2
    child_id : int as unique id of child to be created in population

    Returns
    -------
    genome as a child of input parents p1 and p2.

    """

    child = Genome(child_id, len(p1.input_keys), len(p1.output_keys))
    if f1 < f2:
        p1, p2 = p2, p1
    child = p1.crossover_edges(p2, child)
    child = p1.crossover_nodes(p2, child)
    child.mutate_()
    return child
