from functools import partial
import numpy as np
from gym_developmental.control_policies.pyneat.network import Network


def gym_eval_population(env, population, render=False):
    fitnesses = {}

    for gid, g in population.gid_to_genome.items():
        net = Network.make_network(g)

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


def xor_eval_population(population):
    xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
    xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]

    fitnesses = {}
    for gid, g in population.gid_to_genome.items():
        net = Network.make_network(g)
        fitness = 4.0
        for xi, xo in zip(xor_inputs, xor_outputs):
            output = net.forward(xi)
            fitness -= (output[0] - xo[0]) ** 2
        fitnesses[gid] = fitness
    return fitnesses


def create_gym_eval_population_fn(env):
    eval_func = partial(gym_eval_population, env)
    return eval_func
