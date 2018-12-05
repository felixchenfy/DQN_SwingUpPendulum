#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Run this file by:
# $ python3 src/run_this.py --testing true --swing_up_pend true
# testing: Testing mode, or Training mode
# swing_up_pend: Swing-up Pendulum, or Inverted Pendulum

from pendulum_simulation import Pendulum
from DQN_brain import DeepQNetwork
from math import pi
import time, math
import numpy as np
import argparse

import sys, os
PROJECT_PATH=os.path.join(os.path.dirname(__file__))+ "/../"
sys.path.append(PROJECT_PATH)

# This is the main file of my project.

# Set command line input
parser = argparse.ArgumentParser()
parser.add_argument('--testing',required=False,default="True",
    help='Testing mode, or Training mode')
parser.add_argument('--swing_up_pend',required=False,default="True",
    help='Swing-up Pendulum, or Inverted Pendulum')
args = parser.parse_args(sys.argv[1:])
def check_input(s):
    if s.lower() not in {"true","false"}:
        raise argparse.ArgumentTypeError('T/true or F/false expected')
check_input(args.testing)
check_input(args.swing_up_pend)

# Choose a scenario
Swing_Up_Pendulum = args.swing_up_pend.lower()=="true"
Inverted_Pendulum = not Swing_Up_Pendulum

# Set mode
Testing_Mode=args.testing.lower()=="true"
Training_Mode = not Testing_Mode

# Whether use random initial position (q, dq)
Random_Init_Pose=False

# Max steps per episode
if Inverted_Pendulum:
    Max_Steps_Per_Episode=1000
if Swing_Up_Pendulum:
    Max_Steps_Per_Episode=2000
if Testing_Mode: 
    Max_Steps_Per_Episode=800 # test for only 8 seconds

# time period of taking action and observation
observation_interval=0.01 # seconds

# Specify coordinate:
# When link1 is vertically upward, the angle is 0,
#   which is different from pendulum_simulation.py, so a offset is needed.
Q_OFFSET=pi/2

# Transform angle to [-pi, pi). When link1 is vertically upward, angle=0
def trans_angle(theta):
    return (theta-Q_OFFSET+pi) % (2*pi)-pi

# set weights input/output
Start_New_Train=False
if Training_Mode:
    if Start_New_Train:
        load_path=None
    else:
        load_path=PROJECT_PATH+"/weights/model.ckpt"
    
    save_path=PROJECT_PATH+"/weights/model.ckpt"

    dt_disp = 0.1 # display for every dt_disp seconds. 
    display_after_n_seconds=0
    flag_real_time_display=False
    # flag_real_time_display=True

if Testing_Mode:
    load_path=PROJECT_PATH+"/weights/model.ckpt"

    dt_disp = 0.01 # display for every dt_disp seconds. Set high to display more smoothly
    display_after_n_seconds=0
    flag_real_time_display=True

# main class
class RL_Pendulum(Pendulum):
    def __init__(self, q_init, dq_init):
        # init
        self.state0_init=q_init
        self.state1_init=dq_init
        super(RL_Pendulum, self).__init__(q_init+Q_OFFSET, dq_init)

        # actions
        # (1) no torque; (2) torque to left(-); (3) toruqe to right(+);
        self.action_space = ['o', 'l', 'r']
        self.n_actions = len(self.action_space)

        # states        
        self.n_states=len(self.get_states_for_RL())

        # current action
        self.action=0 # action is in [0, 1, 2]

    def reset(self, q1_init=None, dq1_init=None):
        if q1_init is None:
            super(RL_Pendulum, self).reset(self.state0_init+Q_OFFSET, self.state1_init)
        else:
            super(RL_Pendulum, self).reset(q1_init+Q_OFFSET, dq1_init)
        return self.get_states_for_RL()

    def get_states(self):
        q=trans_angle(self.q)
        dq=self.dq
        return np.array([q, dq])

    # this might be same as get_states(), depending on how to choose states
    def get_states_for_RL(self): 
        if 0: # this is a test of using x,y to replace q. But failed. Not so good.
            x=math.cos(self.q)
            y=math.sin(self.q)
            return np.array([x, y, self.dq])
        else:
            q=trans_angle(self.q)
            return np.array([q, self.dq])

    # step an action into the simualtion for time t
    def step(self, action, t):

        # based on action, choose the torque
        self.action=action
        if action==0:
            self.torque=0
        elif action==1:
            self.torque=-self.Torque_Maginitude # Clockwise
        elif action==2:
            self.torque=+self.Torque_Maginitude # CounterClockwise

        # compute next states
        self.update_states(sim_time_length=t)
        states_new_for_DL=self.get_states_for_RL()
        states_new=self.get_states()
        
        # compute reward
        reward=self.compute_reward(states_new)
        
        # done: true or false. If done==True, stop simulation of current episode.
        if Swing_Up_Pendulum:
            # for this problem, there is no completion of episode
            done=False 
        elif Inverted_Pendulum:
            q = states_new[0]
            if abs(q)>pi/2:
                done=True
            else:
                done=False

        # return states, reward, done
        return states_new_for_DL, reward, done

    # setting rewards
    def compute_reward(self, states):
        abs_q=abs(states[0])
        abs_dq=abs(states[1])

        # reward
        r = lambda q_, dq_: -(q_**2 + 0.01*dq_**2)
        if abs_q<pi*2/3:
            reward= r(abs_q,abs_dq)
        else:
            reward= r(pi*2/3,4)-0.1*(4.0-abs_dq)**2

        return reward

# random initial pose, to test if this will accelerate the converge of training
def rand_init_pose():
    dq_random_max=2
    if Inverted_Pendulum:
        q_random_max=pi/3
        q1_init=(np.random.random()-0.5)*2 *q_random_max
        dq1_init=np.random.random()*dq_random_max
        if q1_init>0: # make init velocity towards the center
            dq1_init=-dq1_init
        states = env.reset(q1_init=q1_init,dq1_init=dq1_init)
    if Swing_Up_Pendulum:
        q_random_max=pi
        q1_init=(np.random.random()-0.5) *2*q_random_max
        dq1_init=(np.random.random()-0.5)*2*dq_random_max
        states = env.reset(q1_init=q1_init,dq1_init=dq1_init)
    return states

# main loop
def run_pendulum():

    # print mode
    print("What mode? Is Testing_Mode ? {}".format(Testing_Mode==True))
    time.sleep(1)

    # load trained data, to keep on previous training
    if load_path is not None:
        _ = RL.saver.restore(RL.sess, load_path)
        print("loading weight from {}".format(load_path))
        time.sleep(1)

    # display settings: display after n seconds
    display_after_n_step=(display_after_n_seconds/observation_interval)

    # start
    steps = 0
    episode=0

    while True:
        episode+=1

        # initial states
        if Random_Init_Pose is True:
            states = rand_init_pose()
        else:
            states = env.reset()

        # start simulation and training of each episode    
        episode_steps=0
        while True:

            # check max steps
            if episode_steps==Max_Steps_Per_Episode:
                break
            episode_steps+=1

            if steps%100==0:
                print("episode=%d, episode_step=%d, total steps=%d, time=%.2f"%(episode,episode_steps, steps,env.t_sim))

            # RL: choose action based on states
            action = RL.choose_action(states)

            # RL: take action, get next states and reward
            states_new, reward, done = env.step(action, observation_interval)

            RL.store_transition(states, action, reward, states_new)

            # if (steps > 1000) and (steps % 20 == 0): # 20 is too slow
            if (steps > 1000) and (steps % 10 == 0):
                RL.learn()

            # swap states
            states = states_new

            # break while loop when end of this episode
            if done:
                break
            steps += 1

            # plot
            if steps>display_after_n_step and steps%(int(dt_disp/observation_interval))==0:
                env.assign_q_to_window()
                env.rotate_link1_to(env.link1_theta)

                # if there is a torque input, set the corresponding light to "red"
                env.canvas.itemconfig(env.light1, fill='black')
                env.canvas.itemconfig(env.light2, fill='black')
                if env.action==1:
                    env.canvas.itemconfig(env.light1, fill='red')
                elif env.action==2:
                    env.canvas.itemconfig(env.light2, fill='red')
                env.render()

                # real time display, so sleep until simulation time
                if flag_real_time_display:
                    if not hasattr(env, "start_display_"):
                        env.start_display_=True
                        step_display=0
                        t_current0=time.time()
                    step_display+=1
                    t_current=time.time()-t_current0
                    t_sleep = step_display*dt_disp-t_current
                    if t_sleep > 0:
                        time.sleep(t_sleep)

    # end of game
    print('game over')
    env.window.destroy()


if __name__ == "__main__":

    # run pendulum game
    if Swing_Up_Pendulum:
        q_init=pi
    elif Inverted_Pendulum:
        q_init=0
    env = RL_Pendulum(q_init=q_init, dq_init=0)

    # deep Q-network
    if Training_Mode:
        e_greedy=0.9
    else: # testing mode
        e_greedy=1.0

    RL = DeepQNetwork(env.n_actions, env.n_states,
                        learning_rate=0.0005,
                        reward_decay=0.995,
                        e_greedy=e_greedy,
                        replace_target_iter=400,
                        batch_size=128,
                        memory_size=4000,
                        flag_record_history=False,
                        e_greedy_increment=None,
                        # output_graph=True,
                    )
  

    # start sim and train
    env.after(100, run_pendulum)
    env.mainloop()


    if Training_Mode:
        RL.save_model(save_path)
    # RL.plot_cost()
    try:
        os.system('xset r on')
    except:
        pass