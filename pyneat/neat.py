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
        fitnesses (): dict of fitness values of genomes in current population
        population : current population (all genomes)
        partitions : current partitions/species

    Returns:
        new_population (Population): Updated population for next generation.

    """

    partition_adjusted_fitnesses = partitions.adjust_fitnesses(fitnesses)       # adjusted fitness (normalized)

    sizes = partitions.next_partition_sizes(partition_adjusted_fitnesses,
                                            len(population.gid_to_genome))      # populatoin per species

    new_population = Population()                                               # new population object

    for pid, p in partitions.pid_to_partition.items():                          # for each species/partition
        size = sizes[pid]                                                       # get proposed size of species

        # Sort in order of descending fitness and remove low-fitness members.
        old_members = sorted(list(p.members),
                             key=lambda x: fitnesses[x],
                             reverse=True)                                      # sorted old fitnesses in desc order

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

            child = new_population.new_child(population.gid_to_genome[gid1],
                                             population.gid_to_genome[gid2],
                                             fitnesses[gid1],
                                             fitnesses[gid2])                   # create new child member

            new_population.gid_to_genome[child.key] = child                     # add new child to new population
            new_population.gid_to_ancestors[child.key] = (gid1, gid2)           # add parents to ancestry of child

    return new_population


def run(eval_population_fn, args):
    """
    run NEAT

    Args:
        eval_population_fn (partial[]): function to evaluate population. Input for this function is instance of Population class.
        args : arguments to initialize the population. `args` has the following attributes:
            * population_size: initial size of population
            * input_size: input size of the policy to be evolved
            * output_size: output size of the policy to be evolved
            * stop_threshold: threshold to stop the generations
            * max_generations: int value of maximum number of generations
            * stop_criterion: it can be a function that evaluates performance per generation by
            aggregation of fitnesses of whole population. This function can be something like numpy.mean, numpy.max etc.

    Returns
    -------

    """
    population = Population.initial_population(args)
    partitions = population.partition(initial_partitions=Partitions())

    stats = defaultdict(list)                                                       # object to log generation history

    gen = 0                                                                         # generation count

    while True:
        gen += 1                                                                    # increament generations

        # Evaluate fitness
        fitnesses = eval_population_fn(population)                                  # evaluate current population

        # log population statistics
        mean_population_fitness = np.mean(list(fitnesses.values()))
        max_population_fitness = np.max(list(fitnesses.values()))

        stats['mean'].append(mean_population_fitness)                       # max fitness in current generation
        stats['max'].append(max_population_fitness)                       # mean fitness in current generation

        # check stop condition, i.e., if fitness > fitness_threshold or max number of generations reached
        if args.stop_criterion(list(fitnesses.values())) >= args.stop_threshold or gen > args.max_generations:
            gid, fitness = max(list(fitnesses.items()), key=lambda x: x[1])

            visualize(population.gid_to_genome[gid], stats)

            if args.task != 'xor':
                eval_pop = Population()
                eval_pop.gid_to_genome[gid] = population.gid_to_genome[gid]
                eval_population_fn(eval_pop, render=True)
            break

        print("generation %d\t fitness\tmean %.3f\tmax %.3f\npopulation %d in %d partitions\n" %
              (gen, float(np.mean(list(fitnesses.values()))), np.max(list(fitnesses.values())),
               len(population.gid_to_genome), len(partitions.pid_to_partition)))

        # Create next generation
        population = next_generation(fitnesses, population, partitions)
        partitions = population.partition(initial_partitions=partitions)
