"""
this file is the entry to the A3C with DDN as the policy network
"""

import queue
import random
import threading
import time
from collections import deque

import numpy as np
import tensorflow as tf
from skimage.color import rgb2gray
from skimage.transform import resize
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop

import gym
from ddn_preprocessor import DDNPreprocessor
from dual_decomposition_network import build_dual_decomposition_model

# global variables for A3C
global episode
global global_vars 

episode = 0
EPISODES = 8000000
ENV_NAME = "FreewayDeterministic-v4"

PLANNING_HORIZON_LENGTH = 10
BP_ITERATIONS = 10


scores = deque(maxlen=100)
global_vars = None
preprocessor_for_ddn = DDNPreprocessor()


#somewhere accessible to both:
callback_queue = queue.Queue()
def from_dummy_thread(func_to_call_from_main_thread):
    callback_queue.put(func_to_call_from_main_thread)

def from_main_thread_blocking():
    callback = callback_queue.get()  # blocks until an item is available
    callback()


# 210*160*3(color) --> 84*84(mono)
# def observation_to_state(next_observe, observe):
#     processed_observe = np.maximum(next_observe, observe)
#     processed_observe = np.uint8(resize(rgb2gray(processed_observe), (84, 84), mode='constant') * 255)
#     return np.float32(processed_observe / 255.)


def build_ac_model(state_size, action_size):
    input = tf.placeholder(shape=[None] + state_size, dtype=tf.float32)
    conv = tf.layers.conv2d(inputs=input, filters=16, kernel_size=[8, 8], strides=4, activation=tf.nn.relu)
    conv = tf.layers.conv2d(inputs=conv, filters=32, kernel_size=[4, 4], strides=2, activation=tf.nn.relu)
    flat = tf.layers.flatten(conv)
    fc = tf.layers.dense(inputs=flat, units=256, activation=tf.nn.relu)

    policy = tf.layers.dense(inputs=fc, units=action_size, activation=tf.nn.softmax)
    value = tf.layers.dense(inputs=fc, units=1, activation=None)  # None is linear activation

    ddn_agent, ddn_policy_orig, ddn_pred_orig = build_dual_decomposition_model(sim_steps=PLANNING_HORIZON_LENGTH, bp_iters=BP_ITERATIONS)
    ddn_policy = tf.map_fn(lambda t: t[0], tf.map_fn(tf.transpose, ddn_policy_orig))  # batch enabled
    
    return ddn_agent, input, policy, value, ddn_policy


# global agent
class A3CAgent:
    def __init__(self, action_size):
        global global_vars 
        # environment settings
        self.state_size = [210, 160, 3]
        self.action_size = action_size

        self.discount_factor = 0.99
        self.no_op_steps = 30

        # optimizer parameters
        self.actor_learning_rate = 0.00028
        self.critic_learning_rate = 0.00028
        self.threads = 1

        with tf.variable_scope('global'):
            self.ddn_agent, self.input, self.policy, self.value, self.ddn_policy = build_ac_model(self.state_size, self.action_size)
        global_vars = sorted(tf.trainable_variables(scope='global'), key=lambda v: v.name)
        self.optimizer = [self.actor_optimizer(), self.critic_optimizer()]

        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())

        self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
        self.summary_writer = None  # tf.summary.FileWriter('./summary/freeway_a3c', self.sess.graph)

    def train(self):
        workers = [
            Worker(i, self.action_size, self.state_size, [self.ddn_agent, self.input, self.policy, self.value, self.ddn_policy], self.sess, self.optimizer,
                self.discount_factor, [self.summary_op, self.summary_placeholders, self.update_ops, self.summary_writer]) 
            for i in range(self.threads)
        ]

        for worker in workers:
            time.sleep(1)
            worker.start()
        
        while True:
            from_main_thread_blocking()

    def actor_optimizer(self):
        global global_vars 
        action = tf.placeholder('float', shape=[None, self.action_size])
        advantages = tf.placeholder('float', shape=[None, ])
        policy = tf.reshape(self.ddn_policy, shape=[-1, self.action_size])
        # policy = self.policy

        pi_sub_theta_of_action = tf.reduce_sum(action * policy, axis=1)
        eligibility = tf.log(pi_sub_theta_of_action + 1e-10) * advantages
        actor_loss = -tf.reduce_sum(eligibility)

        entropy = tf.reduce_sum(policy * tf.log(policy + 1e-10), axis=1)
        entropy = tf.reduce_sum(entropy)

        loss = actor_loss + 0.01 * entropy

        optimizer = tf.train.RMSPropOptimizer(learning_rate=self.actor_learning_rate, epsilon=0.01, decay=0.99)
        updates = optimizer.minimize(loss, var_list=global_vars, gate_gradients=tf.train.Optimizer.GATE_GRAPH)
        return [updates, action, advantages]

    def critic_optimizer(self):
        global global_vars
        discounted_reward = tf.placeholder('float', shape=(None, ))
        value = self.value

        loss = tf.reduce_mean(tf.square(discounted_reward - value))

        optimizer = tf.train.RMSPropOptimizer(learning_rate=self.critic_learning_rate, epsilon=0.01, decay=0.99)
        updates = optimizer.minimize(loss, var_list=global_vars, gate_gradients=tf.train.Optimizer.GATE_GRAPH)
        return [updates, discounted_reward]
        # optimizer = RMSprop(lr=self.critic_learning_rate, rho=0.99, epsilon=0.01)
        # updates = optimizer.get_updates(params=self.critic.trainable_weights, loss=loss)
        # train = K.function([self.critic.input, discounted_reward], [loss], updates=updates)
        # return train

    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        episode_avg_max_q = tf.Variable(0.)
        episode_duration = tf.Variable(0.)

        tf.summary.scalar('Total Reward/Episode', episode_total_reward)
        tf.summary.scalar('Average Max Prob/Episode', episode_avg_max_q)
        tf.summary.scalar('Duration/Episode', episode_duration)

        summary_vars = [episode_total_reward, episode_avg_max_q, episode_duration]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op


class Worker(threading.Thread):
    def __init__(self, i, action_size, state_size, model, sess, optimizer, discount_factor, summary_ops):
        threading.Thread.__init__(self)

        self.index = i

        self.action_size, self.state_size = action_size, state_size
        self.sess, self.optimizer = sess, optimizer
        self.discount_factor = discount_factor
        self.summary_op, self.summary_placeholders, self.update_ops, self.summary_writer = summary_ops

        self.states, self.actions, self.rewards = [],[],[]
        self.ddn_state_list = []

        self.ddn_agent, self.input, self.policy, self.value, self.ddn_policy = model
        self.local_ddn_agent, self.local_input, self.local_policy, self.local_value, self.local_ddn_policy = self.build_local_model()

        self.avg_p_max = 0
        self.avg_loss = 0
        self.t_max = 6  # 20  # t_max -> max batch size for training
        self.t = 0

        self.sess.run(tf.global_variables_initializer())

    def run(self):
        global episode
    
        env = gym.make(ENV_NAME)
        step = 0
        while episode < EPISODES:
            done = False
            dead = False
            score, start_life = 0, 5  # 1 episode = 5 lives
            observe = env.reset()

            # this is one of DeepMind's idea.
            # just do nothing at the start of episode to avoid sub-optimal
            for _ in range(random.randint(1, 30)):
                observe, _, _, _ = env.step(0)

            while not done:
                # env.render()  # not allowed, can only be called at the main thread
                from_dummy_thread(lambda: env.render())

                step += 1
                self.t += 1

                # get action for the current states_stack and go one step in environment
                # action, policy = self.get_action(states_stack)

                ddn_state = preprocessor_for_ddn.obs_to_state(observe)
                ddn_input = np.reshape(ddn_state, (-1, ddn_state.shape[0]))
                # ddn_action, ddn_policy = self.get_ddn_action(ddn_input)
                action, policy = self.get_ddn_action(ddn_input)
                print(policy)

                # if episode % 10 == 0:
                #     print(policy)

                # if len(self.ddn_state_list) > 0:
                #     # print(len(self.ddn_state_list))
                #     following_states = self.sess.run(self.ddn_agent.final_state, {self.ddn_agent.init_state_pl: ddn_input})

                #     final_state = following_states
                #     ball_row = np.exp(final_state[0])
                #     ball_col = np.exp(final_state[1])
                #     row = np.argmax(ball_row)
                #     col = np.argmax(ball_col)
                #     print('prediction (row, col): ' + str((row, col)))
                #     print('row speed: ' + str(np.argmax(np.exp(final_state[3])) -  10 ))
                #     print('col speed: ' + str(np.argmax(np.exp(final_state[4])) - 10 ))
                #     print('paddle speed: ' + str(np.argmax(np.exp(final_state[5])) - 10 ))
                #     # print('row collide: ' + str(np.argmax(np.exp(final_state[7]))))
                #     # print('col diff: ' + str(np.argmax(np.exp(final_state[8]))))
                #     print('paddle next: ' + str(np.argmax(np.exp(final_state[2]))))
                #     print('')


                if dead:
                    dead = False

                observe, reward, done, info = env.step(action)

                state = observe
                state = state[np.newaxis, :]

                self.avg_p_max += np.amax(self.sess.run(self.policy, {self.input: state}))  # np.amax(self.actor.predict(states_stack))

                if start_life > info['ale.lives']:
                    dead = True
                    start_life = info['ale.lives']

                score += reward
                reward = np.clip(reward, -1., 1.)

                self.memorize(state, action, reward, ddn_state)

                if self.t >= self.t_max or done:  # key step here
                    self.train_model(done)
                    self.update_local_model()
                    self.t = 0

                if done:
                    episode += 1
                    scores.append(score)
                    print("episode:", episode, " workere:", self.index, "  score:", score, "  step:", step, " average-score", (sum(scores) / len(scores)))

                    # stats = [score, self.avg_p_max / float(step), step]
                    # for i in range(len(stats)):
                    #     self.sess.run(self.update_ops[i], feed_dict={ self.summary_placeholders[i]: float(stats[i]) })
                    # summary_str = self.sess.run(self.summary_op)
                    # self.summary_writer.add_summary(summary_str, episode + 1)

                    self.avg_p_max = 0
                    self.avg_loss = 0
                    step = 0

    # In Policy Gradient, Q function is not available. Instead agent uses sample returns for evaluating policy
    def td_target(self, rewards, done):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        if not done:
            running_add = self.sess.run(self.value, {self.input: self.states[-1]})[0]  # self.critic.predict(self.states_stack_list[-1])[0]
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    # update policy network and value network
    def train_model(self, done):
        discounted_rewards = self.td_target(self.rewards, done)
        states = np.concatenate(self.states, axis=0)

        values = self.sess.run(self.value, {self.input: states})  # self.critic.predict(states)
        values = np.reshape(values, len(values))
        advantages = discounted_rewards - values  # TD error

        # self.sess.run(self.optimizer[0][0], {
        #     self.input: states,
        #     self.optimizer[0][1]: self.actions,
        #     self.optimizer[0][2]: advantages,
        # })

        # print(len(self.ddn_state_list))
        # self.sess.run(self.optimizer[0][0], {
        #     self.ddn_agent.init_state_pl: self.ddn_state_list,
        #     self.optimizer[0][1]: self.actions,
        #     self.optimizer[0][2]: advantages,
        # })

        # self.sess.run(self.optimizer[1][0], {
        #     self.input: states,
        #     self.optimizer[1][1]: discounted_rewards,
        # })
        self.states, self.actions, self.rewards = [], [], []
        self.ddn_state_list = []

    def build_local_model(self):
        global global_vars 

        with tf.variable_scope('local_' + str(self.index)):
            ddn_agent, input, policy, value, ddn_policy = build_ac_model(self.state_size, self.action_size)
        local_vars = sorted(tf.trainable_variables(scope='local_' + str(self.index)), key=lambda v: v.name)
        self.sync_ops = tf.group([local_var.assign(global_var) for global_var, local_var in zip(global_vars, local_vars)])

        self.sess.run(self.sync_ops)
        return ddn_agent, input, policy, value, ddn_policy

    def update_local_model(self):
        self.sess.run(self.sync_ops)

    def get_ddn_action(self, ddn_input):
        policy = self.sess.run(self.ddn_policy, {self.ddn_agent.init_state_pl: ddn_input})[0]
        action_index = np.random.choice(self.action_size, 1, p=policy)[0]
        return action_index, policy

    def get_action(self, states_stack):
        policy = self.sess.run(self.local_policy, {self.local_input: states_stack})[0]  # self.local_actor.predict(states_stack)[0]  # distribution over actions
        action_index = np.random.choice(self.action_size, 1, p=policy)[0]  # sampling an action
        return action_index, policy

    def memorize(self, state, action, reward, ddn_state):
        self.states.append(state)
        act = np.zeros(self.action_size)
        act[action] = 1  # store action as list of indicator variables
        self.actions.append(act)
        self.rewards.append(reward)

        self.ddn_state_list.append(ddn_state)


if __name__ == "__main__":
    global_agent = A3CAgent(action_size=3)
    global_agent.train()
