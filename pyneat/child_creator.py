from pyneat.genome import Genome


def new_child(p1, p2, f1, f2, child_id):
    """
    Create a child genome given two parent genomes.

    Args:
        p1 (Genome): Genome parent 1.
        p2 (Genome): Genome parent 2.
        f1 (float): Fitness value of parent 1.
        f2 (float): Fitness value of parent 2.
        child_id (int): Unique id of child to be created in population.

    Returns:
        Genome: Child Genome formed by input parents ``p1`` and ``p2``.

    Notes:
        - Genome is a representation of neural network in NEAT.
    """

    child = Genome(child_id, len(p1.input_keys), len(p1.output_keys))

    if f1 < f2:         # parent with greater fitness is referred as ``p1``
        p1, p2 = p2, p1

    child = p1.crossover_edges(p2, child)
    child = p1.crossover_nodes(p2, child)
    child.mutate_()
    return child
