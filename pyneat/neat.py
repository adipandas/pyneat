from collections import defaultdict
import math
import random
import numpy as np

from pyneat.config import ELITISM, CUTOFF_PCT
from pyneat.population import Population
from pyneat.partitions import Partitions
from pyneat.utils import visualize


def next_generation(fitnesses, population, partitions):
    """
    Create next generation using current fitness, population and partitions

    Args:
        fitnesses (dict[int, float]):  Fitness values of genomes in current population. ``Key`` is genome ID and ``Value`` is genome fitness
        population (Population): Current population (all genomes).
        partitions (Partitions): Current partitions (a.k.a. species)

    Returns:
        new_population (Population): Updated population for next generation.

    """

    partition_adjusted_fitnesses = partitions.adjust_fitnesses(fitnesses)       # adjusted fitness (normalized)

    sizes = partitions.next_partition_sizes(partition_adjusted_fitnesses, len(population.gid_to_genome))      # populatoin per species

    new_population = Population()                                               # new population object

    for pid, p in partitions.pid_to_partition.items():                          # for each species/partition
        size = sizes[pid]                                                       # get proposed size of species

        # Sort in order of descending fitness and remove low-fitness members.
        old_members = sorted(list(p.members), key=lambda x: fitnesses[x], reverse=True)         # sorted old fitnesses in desc order

        for gid in old_members[:ELITISM]:                                       # for elite in old species generation
            new_population.gid_to_genome[gid] = population.gid_to_genome[gid]   # add elite to next generation
            size -= 1                                                           # one space occupied in species quota

        cutoff = max(int(math.ceil(CUTOFF_PCT * len(old_members))), 2)
        old_members = old_members[:cutoff]

        # Generate new members.
        while size > 0:
            size -= 1

            # choose two members as parents from old population
            gid1, gid2 = random.choice(old_members), random.choice(old_members)

            parent1, parent2 = population.gid_to_genome[gid1], population.gid_to_genome[gid2]
            fitness1, fitness2 = fitnesses[gid1], fitnesses[gid2]
            child = new_population.new_child(parent1, parent2, fitness1, fitness2)      # create new child member

            new_population.gid_to_genome[child.key] = child                     # add new child to new population
            new_population.gid_to_ancestors[child.key] = (gid1, gid2)           # add parents to ancestry of child

    return new_population


def run(eval_population_fn, args):
    """
    Run NEAT.

    Args:
        eval_population_fn (Union[partial[Dict[int, float]], Callable[[Population, bool], Dict[int, float]]]): Fitness function to evaluate population. Input for this function is instance of Population to evaluate.
        args : arguments to initialize the population. `args` has the following attributes:
            * population_size (int): initial size of population.
            * input_size (int): input size of the policy to be evolved.
            * output_size (int): output size of the policy to be evolved.
            * stop_threshold (float): Fitness threshold to stop the generations.
            * max_generations (int): Maximum number of generations.
            * stop_criterion (Callable): Function that evaluates performance per generation by aggregation of fitnesses of whole population. This function can be something like ``numpy.mean``, ``numpy.max`` etc.

    """
    population = Population.initial_population(args)
    partitions = population.partition(initial_partitions=Partitions())

    stats = defaultdict(list)                                                       # object to log generation history

    gen = 0                                                                         # generation count
    while True:
        gen += 1                                                                    # increament generations

        fitnesses = eval_population_fn(population)                                  # evaluate fitness for current population

        # log population statistics
        mean_population_fitness = np.mean(list(fitnesses.values()))
        max_population_fitness = np.max(list(fitnesses.values()))

        stats['mean'].append(mean_population_fitness)          # mean fitness in current generation
        stats['max'].append(max_population_fitness)            # max fitness in current generation

        # check stop condition, i.e., if fitness > fitness_threshold or max number of generations reached
        if args.stop_criterion(list(fitnesses.values())) >= args.stop_threshold or gen > args.max_generations:
            gid, fitness = max(list(fitnesses.items()), key=lambda x: x[1])

            visualize(population.gid_to_genome[gid], stats)

            if args.task != 'xor':
                eval_pop = Population()
                eval_pop.gid_to_genome[gid] = population.gid_to_genome[gid]
                eval_population_fn(eval_pop, render=True)
            break

        print(f"generation:{gen:4d}\tmean_fitness:{float(np.mean(list(fitnesses.values()))):.3f}\tmax_fitness:{np.max(list(fitnesses.values())):.3f}\tpopulation:{len(population.gid_to_genome):4d}\tpartitions: {len(partitions.pid_to_partition):4d}")

        # Create next generation
        population = next_generation(fitnesses, population, partitions)
        partitions = population.partition(initial_partitions=partitions)
