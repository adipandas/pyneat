import numpy as np
from itertools import count
from pyneat.config import COMPATIBILITY_THRESHOLD, MIN_FITNESS_RANGE, MIN_SPECIES_SIZE
from pyneat.genome import Genome
from pyneat.population import Population


class Partition(object):
    """
    Class to be used as container of partitions. Group candidates according to some distance function. It is referred as **Speciation** in NEAT.

    Args:
        key (int): unique key/id of partition
        members (list): members of partition/species. Default is empty list [].
        representative (Genome): representative of partition, instance of `Genome` class
    """

    def __init__(self, key, members=[], representative=None):
        self.key = key
        self.members = members
        self.representative = representative

    def find_representative(self, gids, population):
        """
        New representative is the closest candidate from `gids` to the current representative.

        Args:
            gids (set or list or tuple): set of genome ids which do not belong to any partition
            population (Population): population containing genomes in which we have to find the representative, instance of Population

        Returns:
            Genome: Closest candidate genome to the current representative.
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

    Notes:
        * Partition is a species
    """

    partition_count = count(1)

    def __init__(self):
        self.pid_to_partition = dict()
        """dict[int, Partition] to store mapping from partition_id to partition"""

        self.gid_to_pid = {}
        """dict[int, int]: to store mapping from genome_id to partition
        """

    def new_partition(self, pid, members, representative):
        """
        Create new partition in population.

        Parameters:
            pid (id): partition id
            members (list[int]): members/candidates from population to be included in the partition/species
            representative (Genome): representative of the new partition (genome representing the new species/partition)

        Notes:
            * Partition is a species
        """
        if pid is None:
            pid = next(Partitions.partition_count)                      # create a partition/species id

        p = Partition(pid, members, representative)                     # make species object
        self.pid_to_partition[pid] = p                                  # store species object with corresponding id

    def closest_representative(self, genome):
        """
        Evaluate closest representative of a species to input genome

        Args:
            genome (Genome): genome or neural network.

        Returns:
            int: id of species which is closest to input genome

        Notes:
            * Partition is a species
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
        Adjust fitnesses is calculated for each partition (a.k.a. species). Mean fitness of each partition is evaluated. This mean fitness is then normalized. The normalized fitness is returned as adjusted fitness.

        Args:
            fitnesses (dict[int, float]): dictionary of fitnesses of genomes in population.

        Returns:
            dict[int, float]: of adjusted (normalized) fitnesses of each partition.

        Notes:
            * Partition is a species
        """

        partition_adjusted_fitnesses = dict()

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

        Args:
            partition_adjusted_fitnesses (dict[int, float]): adjusted (normalized) fitness of each generation's species
            pop_size (int):  size of population.

        Returns:
            dict[]: dict of population size of each partition

        """

        previous_sizes = {pid: len(p.members) for pid, p in self.pid_to_partition.items()}  # partition size for each species in previous generation

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

        sizes = {pid: max(MIN_SPECIES_SIZE, int(round(size * normalizer))) for pid, size in sizes.items()}
        return sizes
