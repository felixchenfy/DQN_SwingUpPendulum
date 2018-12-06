"""
This part of code is the Deep Q Network (DQN) brain.

view the tensorboard picture about this DQN structure on: https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/4-3-DQN3/#modification

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: r1.2
"""

import numpy as np
import tensorflow as tf

# np.random.seed(1)
# tf.set_random_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_states,
            learning_rate=0.01,
            reward_decay=0.8,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=0.001,
            save_steps=-1,
            output_graph=False,
            record_history=True,
            observation_interval=0.01,
    ):
        self.n_actions = n_actions
        self.n_states = n_states
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.record_history=record_history
        self.observation_interval=observation_interval
  
        # total learning step
        self.learn_step_counter = 0
        self.action_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_states * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()

        # save data
        self.save_steps=save_steps
        self.steps=0
        self.saver = tf.train.Saver()
        self.model_path = "tmp/model.ckpt"


        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()


        if output_graph:
            # $ tensorboard --logdir=logs
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

        self.cost_history = []
        self._tmp_cost_history = []

    def _build_net(self):
        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_states], name='s')  # input State
        self.s_ = tf.placeholder(tf.float32, [None, self.n_states], name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)


        # ------------------ build eval_net ------------------
        n_neuron_laywer1=10
        n_neuron_laywer2=15
        n_neuron_laywer3=20
        n_neuron_laywer4=15
        # n_neuron_laywer5=10
        with tf.variable_scope('eval_net'):
            e1 = tf.layers.dense(self.s, n_neuron_laywer1, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1')
            e2 = tf.layers.dense(e1, n_neuron_laywer2, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e2')
            e3 = tf.layers.dense(e2, n_neuron_laywer3, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e3')
            e4 = tf.layers.dense(e3, n_neuron_laywer4, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e4')
            # e5 = tf.layers.dense(e4, n_neuron_laywer5, tf.nn.relu, kernel_initializer=w_initializer,
                                #  bias_initializer=b_initializer, name='e5')
            self.q_eval = tf.layers.dense(e4, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q')

        # ------------------ build target_net (should be the same as eval_net) ------------------
        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.s_, n_neuron_laywer1, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t1')
            t2 = tf.layers.dense(t1, n_neuron_laywer2, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t2')
            t3 = tf.layers.dense(t2, n_neuron_laywer3, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t3')
            t4 = tf.layers.dense(t3, n_neuron_laywer4, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t4')
            # t5 = tf.layers.dense(t4, n_neuron_laywer5, tf.nn.relu, kernel_initializer=w_initializer,
                                #  bias_initializer=b_initializer, name='t5')
            self.q_next = tf.layers.dense(t4, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q_next')


        # # ------------------ build eval_net ------------------
        # # Result: This doesn't work
        # n_neuron_laywer1=10
        # n_neuron_laywer2=10
        # dropout_rate=0.25
        # with tf.variable_scope('eval_net'):
        #     e1 = tf.layers.dense(self.s, n_neuron_laywer1, tf.nn.relu, kernel_initializer=w_initializer,
        #                          bias_initializer=b_initializer, name='e1')
        #     dropout1 = tf.layers.dropout(e1, rate=dropout_rate)
        #     e2 = tf.layers.dense(dropout1, n_neuron_laywer2, tf.nn.relu, kernel_initializer=w_initializer,
        #                          bias_initializer=b_initializer, name='e2')
        #     dropout2 = tf.layers.dropout(e2, rate=dropout_rate)
        #     self.q_eval = tf.layers.dense(dropout2, self.n_actions, tf.nn.softmax, kernel_initializer=w_initializer,
        #                                   bias_initializer=b_initializer, name='q')

        # # ------------------ build target_net ------------------
        # with tf.variable_scope('target_net'):
        #     t1 = tf.layers.dense(self.s_, n_neuron_laywer1, tf.nn.relu, kernel_initializer=w_initializer,
        #                          bias_initializer=b_initializer, name='t1')
        #     dropout1 = tf.layers.dropout(t1, rate=dropout_rate)
        #     t2 = tf.layers.dense(dropout1, n_neuron_laywer2, tf.nn.relu, kernel_initializer=w_initializer,
        #                          bias_initializer=b_initializer, name='t2')
        #     dropout2 = tf.layers.dropout(t2, rate=dropout_rate)
        #     self.q_next = tf.layers.dense(dropout2, self.n_actions, tf.nn.softmax, kernel_initializer=w_initializer,
        #                                   bias_initializer=b_initializer, name='q_next')

        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')    # shape=(None, )
            # Only returns the predition value of the biggest one, because that's the action taken at current state.
            # only this value is compared with q_eval, and then used for gradient descent.
            self.q_target = tf.stop_gradient(q_target)

        with tf.variable_scope('q_eval'):
            # Same as above, only extract the prediction value corresponding to the current action.
            # But this time, since this is the network for predicted quality (similar to testing),
            #   we cannot choose the max,
            #   Instead, extract it from the index of action.
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)    # shape=(None, )

        with tf.variable_scope('loss'):
            # The loss is the difference between the prediction between q_eval and q_target,
            #   on the specific {state, action} pair
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))

        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, states):
        # to have batch dimension when feed into tf placeholder
        states = states[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the states and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: states})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)

        self.action_step_counter+=1
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: batch_memory[:, :self.n_states],
                self.a: batch_memory[:, self.n_states],
                self.r: batch_memory[:, self.n_states + 1],
                self.s_: batch_memory[:, -self.n_states:],
            })

        if self.record_history:
            self._tmp_cost_history.append(cost)
            self.store_interval_time = 5.0 # store history for every X seconds
            store_interval = int(self.store_interval_time*1/self.observation_interval) # store for every X steps

            if self.action_step_counter >= (1+len(self.cost_history))*store_interval:
                num=len(self._tmp_cost_history)
                if num==0:
                    self.cost_history.append(None)
                else:
                    mean_cost=sum(self._tmp_cost_history)/num
                    self.cost_history.append(mean_cost)
                self._tmp_cost_history=[]

        self.steps+=1
        if self.save_steps>0 and self.steps%self.save_steps==0:
            self.save_model()
            
        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def save_model(self, model_path=None):
        if model_path is None:
            model_path=self.model_path
        save_path = self.saver.save(self.sess, model_path)
        print("\nsave weights to %s"%save_path)
        
    def plot_cost(self):
        import matplotlib.pyplot as plt
        
        cost_his=self.cost_history
        n=len(cost_his)
        t=np.arange(1,n+1)*self.store_interval_time
        plt.plot(t, cost_his)
        plt.ylabel('Cost')
        plt.xlabel('Time (second, in simulation)')
        plt.show()

if __name__ == '__main__':
    DQN = DeepQNetwork(3,4, output_graph=True)