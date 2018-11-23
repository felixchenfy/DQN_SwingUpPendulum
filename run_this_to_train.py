
from pendulum_simulation import Pendulum
from RL_brain import DeepQNetwork
from math import pi
import time

def angle_trans_to_pi(theta):# transform angle to [-pi, pi). When link1 is vertically upward, angle=0
    return (theta+pi) % (2*pi)-pi-pi/2

class RL_Pendulum(Pendulum):
    def __init__(self):
        super(RL_Pendulum, self).__init__()
        self.action_space = ['o', 'l', 'r']
        # actions: (1) no torque; (2) torque to left(-); (3) toruqe to right(+);
        self.n_actions = len(self.action_space)
        self.n_features=2 # q and dq
        self.action=0
        return

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
        states_=self.get_states() #[self.q, self.dq]
        
        # compute reward
        reward=self.compute_reward(states_)
        
        # compute stop event
        if 0:
            # for this problem, there is no completion of episode
            done=False 
        else:
            theta = angle_trans_to_pi(states_[0])
            if abs(theta)>pi/6:
                done=True
            else:
                done=False

        return states_, reward, done

    def compute_reward(self, states):
        q=states[0]
        dq=states[1]
        theta = angle_trans_to_pi(q)

        if abs(theta)>pi/2:
            reward== -10 + dq**2
        else:
            reward= -(theta**2 + 0.01*dq**2)

        # max_reward=pi
        # if abs(theta)<max_reward:
        #     reward=(max_reward-abs(theta))**3
        # else:
        #     reward=0

        # reward+=5*min(1, abs(dq))
        return reward

def run_pendulum():

    # for learning frequency
    observation_interval=0.01 # seconds. How long to make a new observation of states.

    # for display
    dt_disp = 0.1
    display_after_n_seconds=200
    display_after_n_step=(display_after_n_seconds/observation_interval)
    flag_real_time_display=True

    step = 0
    episode=0
    while True:
        episode+=1

        # initial states
        states = env.reset()
        while True:

            if step%100==0:
                print("episode=%d, step=%d, time=%.2f"%(episode,step,env.t_sim))

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
                        env.reset_real_time()
                    t_sleep = env.t_sim-display_after_n_seconds-env.get_real_time()
                    if t_sleep > 0:
                        time.sleep(t_sleep)

    # end of game
    print('game over')
    env.window.destroy()


if __name__ == "__main__":
    # pendulum game
    env = RL_Pendulum()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.005,
                      reward_decay=0.99,
                      e_greedy=0.99,
                      replace_target_iter=200,
                      memory_size=2000,
                      flag_record_history=False,
                      e_greedy_increment=0.001
                    #   output_graph=True
                      )

    env.after(100, run_pendulum)
    env.mainloop()
    RL.plot_cost()