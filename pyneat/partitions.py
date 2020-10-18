import numpy as np
from itertools import count
from gym_developmental.control_policies.pyneat.config import COMPATIBILITY_THRESHOLD, \
    MIN_FITNESS_RANGE, MIN_SPECIES_SIZE


class Partition(object):
    """
    Class to be used as container of partitions. Group candidates according to some distance function.
    It is also referred as `Speciation` in NEAT.
    """

    def __init__(self, key, members=[], representative=None):
        """

        Parameters
        ----------
        key : unique key/id of partition
        members : members of partition/species. Default is empty list [].
        representative : representative of partition, instance of `Genome` class
        """
        self.key = key
        self.members = members
        self.representative = representative

    def find_representative(self, gids, population):
        """
        New representative is the closest candidate from `gids` to the current representative.

        Parameters
        ----------
        gids : set of genome ids which do not belong to any partition
        population : population containing genomes in which we have to find the representative, instance of Population

        Returns
        -------
        Closest candidate genome to the current representative.

        """

        candidates = []
        for gid in gids:                                                  # for each genome in population
            d = self.representative.dist(population.gid_to_genome[gid])   # eval distance(representative, genome)
            candidates.append((d, population.gid_to_genome[gid]))         # save calculated distance and genome_id

        _, new_rep_genome = min(candidates, key=lambda x: x[0])           # return new representative genome
        return new_rep_genome


class Partitions(object):
    """
    Class which is a container for various partitions (or various species). It contains set of species.
    """

    partition_count = count(1)

    def __init__(self):
        self.pid_to_partition = {}                              # dict to store mapping from partition_id to partition
        self.gid_to_pid = {}                                    # dict to store mapping from genome_id to partition

    def new_partition(self, pid, members, representative):
        """
        Create new partition in population.
        NOTE: partition == species

        Parameters
        ----------
        pid : partition id
        members : members/candidates from population to be included in the partition/species
        representative : representative of the new partition (genome representing the new species/partition)

        Returns
        -------

        """

        if pid is None:
            pid = next(Partitions.partition_count)                      # create a partition/species id

        p = Partition(pid, members, representative)                     # make species object
        self.pid_to_partition[pid] = p                                  # store species object with corresponding id

    def closest_representative(self, genome):
        """
        Evaluate closest representative of a species to input genome

        Parameters
        ----------
        genome : genome or neural network

        Returns
        -------
        id of species which is closest to input genome

        """

        candidates = []
        for pid, p in self.pid_to_partition.items():                    # for each species/partition in population
            d = p.representative.dist(genome)                           # distance of genome from species representative

            if d < COMPATIBILITY_THRESHOLD:                             # if distance less than threshold
                candidates.append((d, pid))                             # genome is candidate to this partition/species

        if candidates:                                                  # if list of potential candidates is NOT empty
            _, pid = min(candidates, key=lambda x: x[0])                # choose species/partition_id with min(distance)
        else:                                                           # if list of potential candidates is empty
            pid = None                                                  # return `None` as id
        return pid

    def adjust_fitnesses(self, fitnesses):
        """
        Adjust fitnesses is calculated for each species/partition. Mean fitness of each species is evaluated.
        This mean fitness is then normalized.
        The normalized fitness is returned as adjusted fitness.

        Parameters
        ----------
        fitnesses : dict of fitnesses of genomes in population.

        Returns
        -------
        dict of adjusted (normalized) fitnesses of each species/partition.

        """

        partition_adjusted_fitnesses = {}

        min_fitness = min(fitnesses.values())
        max_fitness = max(fitnesses.values())

        fitness_range = max(MIN_FITNESS_RANGE, max_fitness - min_fitness)

        for pid, partition in self.pid_to_partition.items():                # for each species/partition
            msf = np.mean([fitnesses[m] for m in partition.members])        # mean fitness of genomes in the species
            af = (msf - min_fitness) / fitness_range                        # normalize the mean fitness of partition
            partition_adjusted_fitnesses[pid] = af                          # put normalized fitness into adjusted dict

        return partition_adjusted_fitnesses

    def next_partition_sizes(self, partition_adjusted_fitnesses, pop_size):
        """
        Decide partition sizes for the next generation by fitness. Based on Neat-Python.

        Parameters
        ----------
        partition_adjusted_fitnesses : adjusted (normalized) fitness of each generation
        pop_size :  size of population.

        Returns
        -------
        dict of population size of each partition

        """

        # partition size for each species in previous generation
        previous_sizes = {pid: len(p.members) for pid, p in self.pid_to_partition.items()}

        af_sum = sum(partition_adjusted_fitnesses.values())     # sum of adjusted fitness of each partition

        sizes = {}

        # min_species_size = 2
        for pid in partition_adjusted_fitnesses:                # for each partition/species in current generation

            if af_sum > 0:                                      # if adjusted fitness sum is positive
                s = max(MIN_SPECIES_SIZE,
                        partition_adjusted_fitnesses[pid]/af_sum*pop_size)   # species size proportional to its fitness
            else:
                s = MIN_SPECIES_SIZE                            # to avoid extinction of species in next generation

            # difference between new and previous generation population of species
            d = (s - previous_sizes[pid]) * 0.5

            c = int(round(d))

            size = previous_sizes[pid]
            if abs(c) > 0:
                size += c
            elif d > 0:
                size += 1
            elif d < 0:
                size -= 1

            sizes[pid] = size

        normalizer = pop_size / sum(sizes.values())

        sizes = {pid: max(MIN_SPECIES_SIZE,
                          int(round(size * normalizer))) for pid, size in sizes.items()}

        return sizes
