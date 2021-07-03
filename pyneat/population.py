from itertools import count
from pyneat.genome import Genome
from pyneat.partitions import Partitions


class Population(object):
    """
    Class to handle all the population of genomes.
    """

    global_count = count(1)

    def __init__(self):
        self.gid_to_genome = dict()
        """dict[int, Genome]: id to genome mapping"""

        self.gid_to_ancestors = dict()
        """dict[int, Genome]: id to genome ancestors mapping"""

    @classmethod
    def initial_population(cls, args):
        """
        Instantiate initial population.

        Args:
            args: Arguments to initialize the population. `args` has the following attributes:
                - population_size (int): initial size of population
                - input_size (int): input size of the policy to be evolved
                - output_size (int): output size of the policy to be evolved

        Returns:
            Population: Population created using the parameters passed in as `args`
        """
        pop = cls()                                                     # create population object

        for _ in range(args.population_size):                                                 # for each individual in population_size
            gid = next(Population.global_count)                                               # create new genome_id for each individual

            pop.gid_to_genome[gid] = Genome(gid, args.input_size, args.output_size)           # create genome/individual/neural-network
            pop.gid_to_ancestors[gid] = tuple()                                               # gid to ancestors mapping, initialize as empty

        return pop

    def partition(self, initial_partitions):
        """
        Partition population by similarity.

        Args:
            initial_partitions (Partitions): Set of Partitions under consideration

        Returns:
            Partitions: New partitions/species for next generation.

        Notes:
            Partition represents a species in populatoin.
        """

        unpartitioned = set(self.gid_to_genome.keys())                # set of unpartitioned/unspeciated genome
        new_partitions = Partitions()                                 # instantiate empty species/partitions

        # Find new representatives (retain the old partitions ids).
        for pid, p in initial_partitions.pid_to_partition.items():    # for each partition in initial_partitions

            new_rep = p.find_representative(unpartitioned, self)      # find representative of partition - genome

            new_partitions.new_partition(pid, members=[new_rep.key], representative=new_rep)      # find representative population

            unpartitioned.remove(new_rep.key)                         # remove the new representative genome

        # Add remaining members to partitions by finding the partition with the
        # most similar representative; if none exist then create a new partition.
        while unpartitioned:                                         # while all population not speciated
            gid = unpartitioned.pop()                                # genome_id of unspeciated genome from population
            g = self.gid_to_genome[gid]                              # get genome with genome_id
            pid = new_partitions.closest_representative(g)           # eval partition_id of species closest to genome
            if pid is None:                                          # if partition_id does not exist
                new_partitions.new_partition(pid, members=[gid], representative=g)       # create new partition for genome
            else:                                                    # if closest partition_id exists
                new_partitions.pid_to_partition[pid].members.append(gid)   # put genome in the closest partition_id

        return new_partitions

    def new_child(self, p1, p2, f1, f2):
        """
        Enables populating the new population object with a new child created using two input parents.

        Args:
            p1 (Genome): parent 1 genome
            p2 (Genome): parent 2 genome
            f1 (float): fitness of parent 1
            f2 (float): fitness of parent 2

        Returns:
            Genome: new child genome
        """
        child = Genome(next(Population.global_count), len(p1.input_keys), len(p1.output_keys))

        if f1 < f2:
            p1, p2 = p2, p1

        child = p1.crossover_edges(p2, child)
        child = p1.crossover_nodes(p2, child)

        child.mutate_()
        return child
