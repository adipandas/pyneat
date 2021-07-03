[neat_inverted_pendulum]: media/inverted_pendulum_all.gif "Sample of gazebo sim"
[neat_lunar_lander]: media/lunar_lander_all.gif "Sample of rviz"

# NEAT: NeuroEvolution of Augmenting Topologies

| Trained Policies |
:------:
| ![neat_inverted_pendulum][neat_inverted_pendulum] |
| ![neat_lunar_lander][neat_lunar_lander] |

## Installation

```bash
pip install numpy scipy matplotlib
conda install pygraphviz
pip install networkx
```

Installation of OpenAI Gym: [[link](https://github.com/openai/gym)]

## How to use?

This is a minimal implementation of NEAT. I haven't really used any sort of parallel computing tricks like multiprocessing over here. The implementation should be fairly easy to understand.

```bash
python run_neat.py --help

python run_neat.py --task=lunar   # Run NEAT to learn lunar-lander policy
```

For any customization:
* Different tasks: Edit ``run_neat.py``  
* Hyperparameters: Edit ``pyneat/config.py``

## References:
* Stanley, Kenneth O., and Risto Miikkulainen. "Efficient evolution of neural network topologies." Proceedings of the 2002 Congress on Evolutionary Computation. CEC'02 (Cat. No. 02TH8600). Vol. 2. IEEE, 2002.
* neat-python: https://github.com/CodeReclaimers/neat-python

You can use the following implementation for additional/advanced functionalities.
* [https://github.com/google/brain-tokyo-workshop/tree/master/WANNRelease/prettyNEAT](https://github.com/google/brain-tokyo-workshop/tree/master/WANNRelease/prettyNEAT) 
