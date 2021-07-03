
[inverted_pendulum_plot]:  "Inverted Pendulum PLOT"
[inverted_pendulum_neural_network]:  "Inverted Pendulum Neural Network"
[lunar_lander_gif]: media/lunar_lander.gif "Lunar lander GIF"
[lunar_lander_plot]: media/lunar_lander_NEAT.png "Lunar lander PLOT"
[lunar_lander_neural_network]: media/lunar_lander_network_NEAT.png "Lunar lander Neural Network"

# NEAT: NeuroEvolution of Augmenting Topologies

 Environments | Plot (reward vs. iterations) | Network Architecture
:---: | :---: | :---:
<img src="media/inverted_pendulum.gif" width="250" height="200"/> | <img src="media/inverted_pendulum_NEAT.png" width="250" height="200"/> | <img src="media/inverted_pendulum_network_NEAT.png" width="250" height="200"/>
<img src="media/lunar_lander.gif" width="250" height="200"/> | <img src="media/lunar_lander_NEAT.png" width="250" height="200"/> | <img src="media/lunar_lander_network_NEAT.png" width="250" height="200"/>

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
