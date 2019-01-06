import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tcl

import tf_utils
from lib.basic_infer_unit import InferNetPipeLine
from freeway_agent import FreewayAgent

Temperature = 1
LAYER_OVER_POLICY = False


class ACNetFreeway(object):
    '''Actor-critic network class for a3c'''

    def __init__(
            self, state_size, action_size, lr,
            name, SIM_STEPS, BP_STEPS, MULT_FAC,
            env_args, BETA,
            global_name,
            sequential=False):
        self.state_size = state_size
        self.action_size = action_size
        self.net_scope_name = name
        self.SIM_STEPS = SIM_STEPS
        self.BP_STEPS = BP_STEPS
        self.MULT_FAC = MULT_FAC
        self.env_args = env_args
        self.BETA = BETA
        self.sequential = sequential

        self.optimizer = tf.train.RMSPropOptimizer(lr)
        self.input_s, self.input_a, self.advantage, self.target_v, self.policy, self.value, self.action_est, self.model_variables = self._build_network(
            name)

        # 0.5, 0.2, 1.0
        self.value_loss = 0.5 * tf.reduce_mean(tf.square(self.target_v - tf.reshape(self.value, [-1])))
        self.entropy_loss = tf.reduce_mean(tf.reduce_sum(self.policy * tf.log(self.policy), axis=1))
        self.policy_loss = tf.reduce_mean(-tf.log(self.action_est) * self.advantage)
        self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.model_variables])
        # self.loss = 0.5 * self.value_loss + self.policy_loss + 0.2 * self.entropy_loss
        self.loss = self.value_loss + self.policy_loss + self.BETA * self.entropy_loss
        self.gradients = tf.gradients(self.loss, self.model_variables)

        # if name != global_name:
        self.var_norms = tf.global_norm(self.model_variables)

        # global_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, global_name)
        global_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        print('global_variables: \n')
        print(global_variables)

        z = list(filter(lambda p: p[0] is not None, zip(self.gradients, global_variables)))
        self.apply_gradients = self.optimizer.apply_gradients(
            z)  ##zip(self.gradients, global_variables))  TODO: temp add

        self.gradients = [p[0] for p in z]  # TODO: temp add

    def _build_network(self, name):
        # input_s = tf.placeholder(tf.float32, [None, self.state_size])
        input_a = tf.placeholder(tf.int32, [None])
        advantage = tf.placeholder(tf.float32, [None])
        target_v = tf.placeholder(tf.float32, [None])

        with tf.variable_scope(name):
            # layer_1 = tf_utils.fc(
            #     input_s,
            #     self.n_h1,
            #     scope="fc1",
            #     activation_fn=tf.nn.relu,
            #     initializer=tf.contrib.layers.variance_scaling_initializer(
            #         mode="FAN_IN"))
            # layer_2 = tf_utils.fc(
            #     layer_1,
            #     self.n_h2,
            #     scope="fc2",
            #     activation_fn=tf.nn.relu,
            #     initializer=tf.contrib.layers.variance_scaling_initializer(
            #         mode="FAN_IN"))
            # policy = tf_utils.fc(
            #     layer_2,
            #     self.action_size,
            #     activation_fn=tf.nn.softmax,
            #     scope="policy",
            #     initializer=tf_utils.normalized_columns_initializer(0.01)) + 1e-8
            # value = tf_utils.fc(layer_2, 1, activation_fn=None,
            #                     scope="value", initializer=tf_utils.normalized_columns_initializer(1.0))

            self.agent = FreewayAgent(
                simulate_steps=self.SIM_STEPS,
                max_bp_steps=self.BP_STEPS,
                mult_fac=self.MULT_FAC,
                discount_factor=1,
                scope=self.net_scope_name,
                # goal_position=self.env_args['goal_position'],
                # disappearance_probability=self.env_args['disappearance_probability'],
                # sequential=self.sequential
            )

            # Extract Policy and value nodes

            input_s = self.agent.init_state_pl
            final_action_belief = self.agent.final_action_belief * Temperature
            self.final_state = self.agent.final_state

            if LAYER_OVER_POLICY:
                policy = tf_utils.fc(
                    final_action_belief,
                    self.action_size,
                    activation_fn=tf.nn.softmax,
                    scope="policy",
                    initializer=tf_utils.normalized_columns_initializer(0.01)
                ) + 1e-8
            else:
                policy = tf.nn.softmax(final_action_belief)[0] + 1e-8
                # input_s1, input_s2 = tf.split(input_s, [30, 2], axis=1)
                # input_s_new = tf.expand_dims(input_s1, 2) + tf.expand_dims(input_s2, 1)
                input_s_new = input_s
                input_s_new = tf.exp(input_s_new)
                input_s_new = tcl.flatten(input_s_new)

            # layer1 = tf_utils.fc(
            #     input_s_new,
            #     300,
            #     scope="fc1",
            #     activation_fn=tf.nn.relu,
            #     initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN")
            # )

            # value = tf_utils.fc(layer1, 1, activation_fn=None, scope="value", initializer=tf_utils.normalized_columns_initializer(1.0))

            value = self._create_value_network(input_s_new)

            action_mask = tf.one_hot(input_a, self.action_size, 1.0, 0.0)
            action_est = tf.reduce_sum(policy * action_mask, 1)

        model_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        return input_s, input_a, advantage, target_v, policy, value, action_est, model_variables

    def _create_value_network(self, state):
        VALUE_NETWORK_FILTERS = [6, 12, 24]
        KERNEL_SIZE_VALUE_CNN = 3  # [3,3]
        STRIDE_SIZE_VALUE_CNN = 2  # [2,2]
        POOL_SIZE_VALUE_CNN = 2  # [2,2]
        PADDING_VALUE_CNN = "SAME"

        state = tf.reshape(state, [-1, self.state_size, 1])

        l1 = tf.layers.conv1d(
            state,
            filters=VALUE_NETWORK_FILTERS[0],
            kernel_size=KERNEL_SIZE_VALUE_CNN,
            activation=tf.nn.relu,
            padding=PADDING_VALUE_CNN
        )
        max_pool_l1 = tf.layers.max_pooling1d(
            l1,
            pool_size=POOL_SIZE_VALUE_CNN,
            strides=STRIDE_SIZE_VALUE_CNN,
            padding=PADDING_VALUE_CNN
        )
        l2 = tf.layers.conv1d(
            max_pool_l1, filters=VALUE_NETWORK_FILTERS[1],
            kernel_size=KERNEL_SIZE_VALUE_CNN,
            activation=tf.nn.relu,
            padding=PADDING_VALUE_CNN
        )
        max_pool_l3 = tf.layers.max_pooling1d(
            l2,
            pool_size=POOL_SIZE_VALUE_CNN,
            strides=STRIDE_SIZE_VALUE_CNN,
            padding=PADDING_VALUE_CNN
        )

        out = tf.layers.dense(tf.layers.flatten(max_pool_l3), units=1)
        return out

    def get_action(self, state, sess):
        state = np.reshape(state, [-1, self.state_size])
        feed_dict = {self.input_s: state}

        if type(self.agent.infer_net) == InferNetPipeLine:
            feed_dict[self.agent.infer_net.simulate_steps] = self.SIM_STEPS
            feed_dict[self.agent.infer_net.max_steps] = self.SIM_STEPS + (self.BP_STEPS - 1) * 2

        # policy = sess.run(self.policy, feed_dict)
        [policy, final_state] = sess.run([self.policy, self.final_state], feed_dict)
        # print(np.where(state[0][370:530]))
        print(f"prediction: {[np.argmax(s) for s in final_state]}")
        # print(final_state[0])
        print(f"policy: {policy[0]}\n==============================================")

        # policy = tf_utils.run_with_timeline_recorded(sess, self.policy, feed_dict)
        # return np.random.choice(range(self.action_size), p=policy[0])
        return np.argmax(policy[0])

    def predict_policy(self, state, sess):
        state = np.reshape(state, [-1, self.state_size])
        feed_dict = {self.input_s: state}

        if (type(self.agent.infer_net) == InferNetPipeLine):
            feed_dict[self.agent.infer_net.simulate_steps] = self.SIM_STEPS
            feed_dict[self.agent.infer_net.max_steps] = self.SIM_STEPS + (self.BP_STEPS - 1) * 2
        policy = sess.run(self.policy, feed_dict)
        return policy[0]

    def predict_value(self, state, sess):
        state = np.reshape(state, [-1, self.state_size])
        feed_dict = {self.input_s: state}
        if (type(self.agent.infer_net) == InferNetPipeLine):
            feed_dict[self.agent.infer_net.simulate_steps] = self.SIM_STEPS
            feed_dict[self.agent.infer_net.max_steps] = self.SIM_STEPS + (self.BP_STEPS - 1) * 2
        return sess.run(self.value, feed_dict)
