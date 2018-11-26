
from pendulum_simulation import Pendulum
from RL_brain import DeepQNetwork
from math import pi
import time, math
import numpy as np


def angle_trans_to_pi(theta):# transform angle to [-pi, pi). When link1 is vertically upward, angle=0
    return (theta+pi) % (2*pi)-pi-pi/2

# Choose a scenario
Inverted_Pendulum=True
Swing_Up_Pendulum=not Inverted_Pendulum

Random_Init=False

# set mode
Training_Mode=True
Testing_Mode=not Training_Mode

# set weights input/output
if Training_Mode:
    load_path="./tmp/model.ckpt"
    # load_path=None
    save_path="./tmp/model.ckpt"

    dt_disp = 0.1
    display_after_n_seconds=0
    flag_real_time_display=False

if Testing_Mode:
    load_path="./good_weights_upward_pendulum/model.ckpt"

    dt_disp = 0.01 # display for every dt_disp seconds
    display_after_n_seconds=0
    flag_real_time_display=True


class RL_Pendulum(Pendulum):
    def __init__(self, q_init, dq_init):
        super(RL_Pendulum, self).__init__(q_init, dq_init)
        self.action_space = ['o', 'l', 'r']
        # actions: (1) no torque; (2) torque to left(-); (3) toruqe to right(+);
        self.n_actions = len(self.action_space)
        self.n_features=len(self.get_states_for_DL())
        self.action=0
        return

    def reset(self, q1_init=None, dq1_init=None):
        super(RL_Pendulum, self).reset(q1_init, dq1_init)
        return self.get_states_for_DL()

    def get_states_for_DL(self):
        q=self.q
        dq=self.dq
        # x=math.cos(q)
        # y=math.sin(q)
        # return np.array([x, y, dq])
        return np.array([q, dq])

    def step(self, action, time):
        self.action=action
        if action==0:
            self.torque=0
        elif action==1:
            self.torque=-self.Torque_Maginitude # Clockwise
        elif action==2:
            self.torque=+self.Torque_Maginitude # CounterClockwise

        # compute next states
        self.update_states(sim_time_length=time)
        states_new_xydq=self.get_states_for_DL()
        states_new=self.get_states()
        
        # compute reward
        reward=self.compute_reward(states_new)
        
        # done: stop event
        if Swing_Up_Pendulum:
            # for this problem, there is no completion of episode
            done=False 
        elif Inverted_Pendulum:
            theta = angle_trans_to_pi(states_new[0])
            if abs(theta)>pi/6:
                done=True
            else:
                done=False

        # return states_new_xydq, reward, done
        return states_new, reward, done

    def compute_reward(self, states):
        q=states[0]
        dq=states[1]
        theta = angle_trans_to_pi(q)

        # reward
        # if abs(theta)>pi*1.0/2:
        #     reward= -300 + 100*min(1.0, dq)**2
        # else:
        #     reward= -(100*abs(theta)**2 + dq**2)

        reward= -(abs(theta)**2 + 0.01*dq**2)
        
        return reward



def run_pendulum(observation_interval_=0.01):

   # learning frequency
    observation_interval=observation_interval_ # seconds. How long to make a new observation of states.

    # print mode
    print("What mode? Is Testing_Mode ? {}".format(Testing_Mode==True))
    time.sleep(1)

    # load trained data
    if load_path is not None:
        _ = RL.saver.restore(RL.sess, load_path)
        print("loading weight from {}".format(load_path))
        time.sleep(1)

    # display settings
    display_after_n_step=(display_after_n_seconds/observation_interval)

    # start
    step = 0
    episode=0
    while True:
        episode+=1

        # initial states
        if Random_Init is True:
            if Inverted_Pendulum:
                q1_init=(np.random.random()-0.5) *pi/3 /1.1+pi/2
                states = env.reset(q1_init=q1_init,dq1_init=0)
            if Swing_Up_Pendulum:
                None
        else:
            states = env.reset()
            
            
        while True:
            
            if step%100==0:
                print("episode=%d, step=%d, time=%.2f"%(episode,step,env.t_sim))

            # RL choose action based on states
            action = RL.choose_action(states)

            # RL take action and get next states and reward
            observation_, reward, done = env.step(action, observation_interval)

            RL.store_transition(states, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap states
            states = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1

            # plot
            if step>display_after_n_step and step%(int(dt_disp/observation_interval))==0:
                env.assign_q_to_window()
                env.rotate_link1_to(env.link1_theta)

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
    # pendulum game
    if Swing_Up_Pendulum:
        q_init=-pi/2
    elif Inverted_Pendulum:
        q_init=pi/2
    env = RL_Pendulum(q_init=q_init, dq_init=0)

    # deep Q-network
    if Training_Mode:
        e_greedy=0.90
    else: # testing mode
        e_greedy=1.0

    RL = DeepQNetwork(env.n_actions, env.n_features,
                        learning_rate=0.005,
                        reward_decay=0.95,
                        e_greedy=e_greedy,
                        replace_target_iter=200,
                        memory_size=3000,
                        flag_record_history=False,
                        e_greedy_increment=0.001,
                        #   output_graph=True,
                    )

    env.after(100, run_pendulum)
    env.mainloop()


    if Training_Mode:
        RL.save_model(save_path)
    # RL.plot_cost()