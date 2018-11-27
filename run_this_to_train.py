
from pendulum_simulation import Pendulum
from RL_brain import DeepQNetwork
from math import pi
import time, math
import numpy as np

Q_OFFSET=pi/2
def trans_angle(theta):# transform angle to [-pi, pi). When link1 is vertically upward, angle=0
    return (theta-Q_OFFSET+pi) % (2*pi)-pi
# theta=pi/2, return 0
# theta=pi/2+0.1, return 0.1

# Choose a scenario
Inverted_Pendulum=False
Swing_Up_Pendulum=not Inverted_Pendulum

# Whether use random initial position (q, dq)
Random_Init=True
if Inverted_Pendulum:
    Max_Steps_Per_Episode=1000
else:
    Max_Steps_Per_Episode=4000

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
    # flag_real_time_display=True


if Testing_Mode:
    load_path="./tmp/model.ckpt"

    dt_disp = 0.01 # display for every dt_disp seconds
    display_after_n_seconds=0
    flag_real_time_display=True


class RL_Pendulum(Pendulum):
    def __init__(self, q_init, dq_init):
        self.state0_init=q_init
        self.state1_init=dq_init
        super(RL_Pendulum, self).__init__(q_init+Q_OFFSET, dq_init)

        self.action_space = ['o', 'l', 'r']
        # actions: (1) no torque; (2) torque to left(-); (3) toruqe to right(+);
        self.n_actions = len(self.action_space)
        self.n_features=len(self.get_states_for_DL())
        self.action=0
        return

    def reset(self, q1_init=None, dq1_init=None):
        super(RL_Pendulum, self).reset(self.state0_init+Q_OFFSET, self.state1_init)
        return self.get_states_for_DL()

    def get_states(self):
        q=trans_angle(self.q)
        dq=self.dq
        return np.array([q, dq])

    def get_states_for_DL(self):
        if 0: # this is a test of using x,y to replace q. But failed. Not so good.
            x=math.cos(self.q)
            y=math.sin(self.q)
            return np.array([x, y, self.dq])
        else:
            q=trans_angle(self.q)
            return np.array([q, self.dq])

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
        states_new_for_DL=self.get_states_for_DL()
        states_new=self.get_states()
        
        # compute reward
        reward=self.compute_reward(states_new)
        
        # done: stop event
        if Swing_Up_Pendulum:
            # for this problem, there is no completion of episode
            done=False 
        elif Inverted_Pendulum:
            q = states_new[0]
            if abs(q)>pi/2:
                done=True
            else:
                done=False

        # return states_new_xydq, reward, done
        return states_new_for_DL, reward, done

    def compute_reward(self, states):
        q=states[0]
        dq=states[1]
        # print(q, dq)
        q=abs(q)
        dq=abs(dq)

        # reward
        r = lambda q_, dq_: -(q_**2 + 0.01*dq_**2)
        if q<pi*2/3:
            reward= r(q,dq)
        else:
            reward= r(pi*2/3,4)-0.1*(4.0-dq)**2

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
        else:
            states = env.reset()
            
        episode_steps=0
        while True:

            # check max steps
            if episode_steps==Max_Steps_Per_Episode:
                break
            episode_steps+=1

            if step%100==0:
                print("episode=%d, episode_step=%d, total steps=%d, time=%.2f"%(episode,episode_steps, step,env.t_sim))

            # RL choose action based on states
            action = RL.choose_action(states)

            # RL take action and get next states and reward
            observation_, reward, done = env.step(action, observation_interval)

            RL.store_transition(states, action, reward, observation_)

            if (step > 200) and (step % 10 == 0):
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
        q_init=pi
    elif Inverted_Pendulum:
        q_init=0
    env = RL_Pendulum(q_init=q_init, dq_init=0)

    # deep Q-network
    if Training_Mode:
        e_greedy=0.9
    else: # testing mode
        e_greedy=1.0

    RL = DeepQNetwork(env.n_actions, env.n_features,
                        learning_rate=0.005,
                        reward_decay=0.95,
                        e_greedy=e_greedy,
                        replace_target_iter=500,
                        batch_size=256,
                        memory_size=8000,
                        flag_record_history=False,
                        e_greedy_increment=None,
                        #   output_graph=True,
                    )

    env.after(100, run_pendulum)
    env.mainloop()


    if Training_Mode:
        RL.save_model(save_path)
    # RL.plot_cost()