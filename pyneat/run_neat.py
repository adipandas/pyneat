"""
Run with using following commands in terminal:
$ cd ./developmental-gym
$ python -m gym_developmental.control_policies.pyneat.run_neat --task=xor

"""

if __name__ == '__main__':
    import numpy as np
    import argparse
    import gym

    from gym_developmental.control_policies.pyneat.evaluate_population import xor_eval_population, \
        create_gym_eval_population_fn
    from gym_developmental.control_policies.pyneat.neat import run

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['xor', 'cartpole', 'lunar'], required=True)

    args = parser.parse_args()

    if args.task == 'xor':
        eval_func = xor_eval_population

        args.stop_criterion = np.max
        args.stop_threshold = 4.0 - 1e-3
        args.input_size = 2
        args.output_size = 1
        args.population_size = 250
        args.max_generations = 200

        run(eval_func, args)

    if args.task == 'cartpole':
        env = gym.make('CartPole-v1')
        env._max_episode_steps = 500

        eval_func = create_gym_eval_population_fn(env)

        args.stop_threshold = 200
        args.input_size = 4
        args.output_size = 1
        args.stop_criterion = np.mean
        args.max_generations = 200
        args.population_size = 250

        run(eval_func, args)

    if args.task == 'lunar':
        env = gym.make('LunarLander-v2')
        eval_func = create_gym_eval_population_fn(env)

        args.stop_threshold = 250
        args.input_size = 8
        args.output_size = 4
        args.stop_criterion = np.max
        args.max_generations = 200
        args.population_size = 500

        run(eval_func, args)
