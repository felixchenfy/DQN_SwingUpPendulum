
# Feiyu Chen  
# Project: Swing up a 1-link pendulum using Reinforcement Learning
This is the final project of EECS 395/495: Optimization in Deep Learning and Machine Learning.

# Introction
## Intro
* The goal of the game is to swing up a pendulum and keep it inverted.
* User or AI player could input 3 types of torque (zero, positive, negative) by pressing "j" or "k".
* The GUI of the game utilizes the "tkinter" library. The motion of link is solved from Eular-Lagrange Equations.
* The AI of this game uses a reinforcement learning algorithm called Deep Q-network. It's an modified version of the original Q-learning algorithm by using **Neural Network** to evaluate the quality of taking action A at state S, and thus to know the best action to take. Two extra techniques called **Experience Replay** and **Fixed Q-targets** are also included for updating neural netork's weights.

The scripts are based on Python3.6, tkinter, and tensorflow. The core code of DQN's neural network and optimization strategy referenced this github tutorial https://morvanzhou.github.io/tutorials/, which trains a DQN for a simple grid maze game.

## How to run:
First activate anaconda, then run:
> $ python3 src/run_this.py

## Dependency
* Python 3.6  
* tkinter  
* tensorflow  
* other common libraries. 

## Algorithm
The reinforcement learning (RL) algorithm I adopted is called Deep Q-Network.  

The main idea of RL looks like this:  
There is a robot with state S. It can take an action A from {A}. Every time the robot takes an action, we receives a reward. Based on the reward, the robot could know if it is a good choice to take this action. In this way, the "Quality Table" or the "Neural Network" can be updated to reduce the difference between Estimated Quality and Real Quality. It requires iterations of trainings to get converged.

The main difference between **Deep Q-network** and **Q learning** is that DQN uses Neural Networt to evaluate the quality of taking action A at state S, instead of using a look-up table (Use NN to fit the discrete look-up table).

For details, please see this [paper]((https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)) or this Chinese [tutorial](https://morvanzhou.github.io/tutorials/).

## Reference
* The core AI of my homework, i.e. Deep Q-Network, references this Github tutorial: https://morvanzhou.github.io/tutorials/. Thanks Morvan!
* [Deepmind's website for DQN](https://deepmind.com/research/dqn/)
* Paper: [Human-level control through deep reinforcement
learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)


# Files
## 1. [pendulum_window.py](src/pendulum_window.py)  
**This implements the geometry of the pendulum.**  
It creates a window and draws a pendulum.   
> $ python3 src/pendulum_window.py  

Run it, and the user could press keyboard "a" and "d" to change the angle of the pendulum.  

## 2. [pendulum_simulation.py](src/pendulum_simulation.py)  
**This implements the dynamics of the simualted pendulum.**  
It inherits the above class, adds the dynamics to compute the angular acceleration, and simulates the motion of the pendulum. You can also modify friction and noise to test if the trained AI is robust.  
> $ python3 src/pendulum_simulation.py  

Run it, and the user could press keyboard "j" and "k" to apply torque to this link.

## 3. [DQN_brain.py](src/DQN_brain.py)  
**This implements the Neural Network part of DQN.**   
Details about the setting and training of DQN is described in "[How I Trained This DQN](#How-I-Trained-This-DQN)" section

## 4. [run_this.py](src/run_this.py)  
**This is the main file.**

Modify arguements, and run it to see the final result.
Two major arguements needs to be set:
1. Inverted_Pendulum=False
    * If true, the pendulum starts with an inverted state.
    * If false, the pendulum is hanging downwards.

2. Training_Mode=False
    * If true, the programs will keep on training the DQN (or you can set to retrain). The animation will be like 20 times faster than real time.
    * If false, it's in testing mode. You can view how AI swings up the pendulum in a very efficient manner.

Besides, you can also set "Random_Init_Pose=True", and the pendulum will start with random pose, so as to demonstrate the robustness of the trained DQN.

## 5. [weights/](weights/)
This is where the trained weights are stored.


# How I Trained This DQN

This might be the most useful part of my project. I gained some understandings and thoughts about how to train Deep Q-Network, through the process of tuning many of its settings and parameters.

I listed the details of setting up DQN, my testing procedures, and what I learned in the following sections.

## 0. Simulation settings

Some major params for simulation is set as follows:
* Torque=2: The torque appied by player is set as 2. It takes at least 3 swings (e.g., left, right, left) to swing up the pendulum.
* Friction=-0.1*dq
* Noise=(+,-)0.01*Random()
* Low-level simulation time=0.001s for updating the state of pendulum.

Others related to training the DQN:
* t_train=0.01: For training the DQN, the period of taking action and makeing observation is 0.01s.  
* For visualation, the refresh period of window is set as 0.1s during the training stage, and set as 0.01s in testing stage. Besides, during training, the game is not display in real time, but about 20 times faster than real time.

The simulation time is about 20 times faster than real-world time. You may even increase it by disable the visualization, or increase the low-level simulation time.

## 1. States of Pendulum
The states of a pendulum is the angle $q$ and angular velocity $dq$.

The key points are:
* Set q in [-Pi, Pi]. We know that Neural Network is for fitting a function. Restricting the definition domain makes it converge faster and better. 
* Set q=0 when the pendulum is upwards. I believe this is a better choice, since it makes $q$ continuous at the goal state (upwards), and makes the discontinous point in the farthest point (q=Pi, downwards).
* Not using states (x,y,dq), where $x=l*cos(q)$ and $y=l*sin(q)$, to replace (q,dq). Though it solves the discontinuousty problem, it increases the states dimension from 2 to 3 and makes network difficult to train. For example, when training the pendulum to swing up for a long time with states near(x=0, y=-l), the weights associated with keeping pendulum inverted near (x=0, y=l) will be affected (which won't happen when using $q$). In other words, training task A ruins the weights of task B.

## 2. Reward function
The "Swing up a pendulum" actually involves two tasks:
1. Swing up the pendulum
2. Keep it inverted

I tested two different types of rewards. The first cannot work. The second works:  
1. $(-c1*q^2-c2*dq^2)$.
    Though this is beautiful and fits the tuition that we want to make $q=0$ and $dq=0$, it actually cannot work. The pendulum always gets into the local minimum in the half way of swinging up.  
    
    Image, when you play this game, you need to swing the pendulum to left, to right, to left for several times to make pendulum acquire enough energy for the final swinging up. However, in case of using this reward, the pendulum will be unwilling to swing downwards, since both $-q^2$ and $-dq^2$ decrease. Thus, it falls into a local minumum.
2. Using $R1=(c3-c4*(c5-Abs(dq))^2)$ to swing up the pendulum when $q>2/3*Pi$, and using $R2=(-c1*q^2-c2*dq^2)$ to drive the pendulum inverted when $q<=2/3*Pi$. 

    In this way, the pendulum wants to achieve a speed of $c5$ during the swing-up stage. Besides, you need to make sure the rewards is roughly continuous.

The reward I adopted is:
``` Python
r = lambda q_, dq_: -(q_**2 + 0.01*dq_**2)
if abs_q<pi*2/3:
    reward= r(abs_q,abs_dq)
else:
    reward= r(pi*2/3,4)-0.1*(4.0-abs_dq)**2
``` 
The weights between q and dq need some time to finetune.

## 3. Structure of Neural Network

* Input:
2 neurons (states: q and dq)  
* Hidden: 4 fully connected layers  
Sizes of 10, 15, 20, 15.  
Use Relu for activation.  
* Output: 3 neurous  
Fully connected and direct output.  
The three classes are: No toruque, counter-clockwise, or clockwise torque.

How I choose this structure?  
I start from 1 hidden layer, and increases the depth. I evaluated them by applying it to train the inverted pendulum, which converges fastly. When the depth goes to 5, the inverted pendulum cannot be converged even after 5000 seconds (simulation time). Meanwhile, 4 layers is good, so I set it to 4. This leaves enough capacity for training the "swing up pendulum".

I also tried "dropout" or changing the output layer to "softmax". But the converge becomes very slow. I don't know why. Since I'm still a novice in deep learning and reinforcement, I didn't further investigated these parameters.

## 4. Tuning Parameters
I conclude some experience for tuning the pamameters.

* learning_rate for Gradient Descent  
    I used "RMSPropOptimizer" and set learning rate as 0.0005. I've 
    tried a range from 0.0001 and 0.01. My conclusion is that 0.01 is too large, because the pendulum never swing itself up. I think the way to tune this is just to keep decreasing it by 1/10 until get a good perforcement. 
* reward_decay  
    Since I set the learning frequency as high as 1/0.01=100, the reward_decay should be close to 1, where I set as 0.995.
    
    0.995^500=0.08, so roughly the future rewards in 5 seconds are considered.

    The larger it is, the further future is considered.
* e-greedy  
    This decides the probability for pendulumn to choose the best action to take. The smaller it is, the training explores more unknown spaces.
* Others  
    There are other important params such as: replace_target_iter, batch_size, memory_size. They are definitely important, but I just don't have a good idea about how to set them. I've tuned them for several values, but there is no apparent conclusion. One thing I do know is that when making faster observation, these values should be inceased.

## 5. Two-phase training

It would be very difficult to train this "swing up pendulum" from scratch, since there is too few chances for the pendulum to stay upright and experience the success. Thus, I trained it in two phases:  
1. Train an inverted pendulum.
2. Train to swing up.

More specifically, I trained this in four different stages:
1. Train an inverted pendulum which starts from up-right position.
2. Train an inverted pendulum, which starts randomly at an angle of $[-pi/3, pi/3]$ and a random velocity towards the center.
3. Train the swing-up pendulum, with random initial angle and velocity.
4. Train the swing-up pendulum, with static initial angle of $-pi$.

The result and detials are in next section.

## Result


inv_pend train for 120k, down_pend train for 1000k, basically converged. 

Still oscillation in down poses, might due to too large step size? I will save it first.

    
# Result



