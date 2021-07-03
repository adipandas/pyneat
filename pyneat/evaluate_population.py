from functools import partial
import gym
import numpy as np
from pyneat.network import NeuralNetwork


def gym_eval_population(env, population, render=False):
    """
    Function to evaluate population with the given environment.

    Args:
        env (gym.Env): Gym environment.
        population (pyneat.population.Population): Population of agents to evaluate on gym environment.
        render (bool): If ``True``, render environment while evaluation.

    Returns:
        dict[int, float]: Dictionary containing fitness score (``value``) of each individual in the population (``key``).

    """
    fitnesses = {}

    for gid, g in population.gid_to_genome.items():
        net = NeuralNetwork.make_network(g)

        episode_reward = 0
        num_episodes_per_eval = 5
        for i in range(num_episodes_per_eval):
            done = False
            t = 0
            state = env.reset()
            while not done:

                action = net.forward(state)
                if 'LunarLander' in env.__repr__():
                    action = np.argmax(action)
                elif 'CartPole' in env.__repr__():
                    action = int(action[0] > 0.5)

                state, reward, done, _ = env.step(action)
                episode_reward += reward
                t += 1

                if render:
                    env.render()

        render = False
        fitnesses[gid] = episode_reward / num_episodes_per_eval
    return fitnesses


def create_gym_eval_population_fn(env: gym.Env):
    """
    Create a function for particular Evironment Evaluation.

    Args:
        env (gym.Env): Environment object for which the evaluation function is created.

    Returns:
        Union[partial[dict], Callable[[pyneat.population.Population, bool], Dict[int, float]]]: Evaluation function for the given gym environment.
    """
    eval_func = partial(gym_eval_population, env)
    return eval_func


def xor_eval_population(population):
    """
    Evaluation of population for solving the XOR problem.

    Args:
        population (pyneat.population.Population): Population of agents to evaluate.

    Returns:
        dict[int, float]: Dictionary with ``key-value`` pairs of genome (``key`` as genome id) in the population and corresponding fitnesses (``value``) of each.
    """
    xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
    xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]

    fitnesses = dict()
    for gid, g in population.gid_to_genome.items():
        net = NeuralNetwork.make_network(g)
        fitness = 4.0
        for xi, xo in zip(xor_inputs, xor_outputs):
            output = net.forward(xi)
            fitness -= (output[0] - xo[0]) ** 2
        fitnesses[gid] = fitness
    return fitnesses
