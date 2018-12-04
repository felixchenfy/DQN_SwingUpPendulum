
# Feiyu Chen  
# Project: Swing up a 1-link pendulum using Reinforcement Learning
This is the final project of EECS 395/495: Optimization in Deep Learning and Machine Learning.

## Introction
* This is a game to swing up a pendulum and keep it inverted.
* User or AI player could input 3 types of torque (zero, positive, negative) by pressing "j" or "k".
* The GUI of the game is made by using "tkinter" library. The motion of link is solved from Eular-Lagrange Equation.
* The AI of the game uses a reinforcement learning algorithm called Deep Q-network. It's an modified version of the original Q-learning algorithm by using **Neural Network** to build up the mapping from states to actions. Two extra techniques called **Experience Replay** and **Fixed Q-targets** for updating neural netork's weights are also included.

The scripts are based on Python3.6, tkinter, and tensorflow. The core code of DQN's neural network and optimization strategy referenced this github tutorial https://morvanzhou.github.io/tutorials/, which trains a DQN for a simple grid maze game.

## How to run:
First activate anaconda, then run:
> $ python3 src/run_this.py

## Dependency
* Python 3.6  
* tkinter  
* tensorflow  
* other common libraries. 

# Files
* [pendulum_window.py](src/pendulum_window.py)
* [pendulum_simulation.py](src/pendulum_simulation.py)
* [DQN_brain.py](src/DQN_brain.py)
* [run_this.py](src/run_this.py)

# Algorithm
Deep Q-Network. 

For details, please see this [paper]((https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)) or this Chinese [tutorial](https://morvanzhou.github.io/tutorials/).


# Reference
* The core AI of my homework, i.e. Deep Q-Network, references this Github tutorial: https://morvanzhou.github.io/tutorials/. Thanks Morvan!
* [Deepmind's website for DQN](https://deepmind.com/research/dqn/)
* Paper: [Human-level control through deep reinforcement
learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
