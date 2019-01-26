'''Actor-critic network class for a3c'''
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tcl

from agents.base import FreewayBaseAgent
import tf_utils
from lib.BasicInferUnit import InferNetPipeLine, InferNetRNN

Temperature = 1
LAYER_OVER_POLICY = False


class ACNetFreeway(object):
    def __init__(self, rl_lr, supervised_lr,
                 name, SIM_STEPS, BP_STEPS, MULT_FAC, BETA, global_name='global', sequential=False):

        self.name = name
        self.SIM_STEPS = SIM_STEPS
        self.BP_STEPS = BP_STEPS
        self.MULT_FAC = MULT_FAC
        # self.env_args = env_args
        self.BETA = BETA
        self.sequential = sequential
        self.rl_lr = rl_lr
        self.supervised_lr = supervised_lr

        self.rl_optimizer = tf.train.RMSPropOptimizer(self.rl_lr)
        self.supervised_optimizer = tf.train.RMSPropOptimizer(self.supervised_lr)

        self.input_s, self.input_sprime, self.input_r, self.input_a, self.advantage, \
        self.target_v, self.policy, self.value, self.action_est, self.model_variables_rl, self.model_variables_sl = self._build_network(name)
        # 0.5, 0.2, 1.0
        self.value_loss = 0.5 * tf.reduce_mean(tf.square(self.target_v - tf.reshape(self.value, [-1])))
        self.entropy_loss = tf.reduce_mean(tf.reduce_sum(self.policy * tf.log(self.policy), axis=1))
        self.policy_loss = tf.reduce_mean(-tf.log(self.action_est) * self.advantage)
        self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.model_variables_rl] + [tf.nn.l2_loss(v) for v in self.model_variables_sl])
        # self.loss = 0.5 * self.value_loss + self.policy_loss + 0.2 * self.entropy_loss
        self.rl_loss = self.value_loss + self.policy_loss + self.BETA * self.entropy_loss
        self.gradients_rl = tf.gradients(self.rl_loss, self.model_variables_rl)
        self.gradients_rl, self.grad_rl_norm = tf.clip_by_global_norm(self.gradients_rl, 5.0)

        self.supervised_loss = self.get_sl_loss()
        self.gradients_sl = tf.gradients(self.supervised_loss, self.model_variables_sl)
        self.gradients_sl, self.grad_sl_norm = tf.clip_by_global_norm(self.gradients_sl, 5.0)

        if name != global_name:
            self.var_norms_rl = tf.global_norm(self.model_variables_rl)
            global_variables_rl = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, global_name + '/rl_params')
            self.apply_gradients_rl = self.rl_optimizer.apply_gradients(zip(self.gradients_rl, global_variables_rl))

            self.var_norms_sl = tf.global_norm(self.model_variables_sl)
            global_variables_sl = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, global_name + '/sl_params')
            self.apply_gradients_sl = self.supervised_optimizer.apply_gradients(zip(self.gradients_sl, global_variables_sl))

            self.var_norms_total = tf.global_norm(self.model_variables_rl + self.model_variables_sl)

    def get_sl_loss(self):
        cross_state_factors = self.agent.state_transition_factors
        reward_factors_instate = self.agent.reward_factors_instate
        reward_factor_crossstate = self.agent.reward_factors_crossstate
        reward_targets = self.input_r

        state_nodes = self.get_variable_values(self.input_s, self.agent.all_state_dim)
        next_state_nodes = self.get_variable_values(self.input_sprime, self.agent.all_state_dim)
        action_nodes = self.get_variable_values(self.input_a, self.agent.action_dim)
        self.state_nodes, self.next_state_nodes, self.action_nodes = state_nodes, next_state_nodes, action_nodes
        transition_loss = self.get_loss_transition_cross_entropy(action_nodes, cross_state_factors,
                                                         next_state_nodes, state_nodes)

        reward_loss = self.get_loss_for_reward_factors(reward_factors_instate, reward_factor_crossstate,
                                                    state_nodes, next_state_nodes, action_nodes, reward_targets)
        self.reward_loss = reward_loss
        self.transition_loss = transition_loss
        sl_loss = transition_loss + reward_loss
        return tf.reduce_mean(sl_loss)


    def get_variable_values(self, placeholder, split_desc):
        nodes = tf.split(placeholder, split_desc, axis=1)
        for idx, var in enumerate(nodes):
            nodes[idx] = tf.expand_dims(tf.argmax(var, axis=1), axis=1)
        return nodes


    def get_loss_for_transition_factor(self, action_nodes, factor, next_state_nodes, state_nodes):
        potential = factor['Factor'][0].sl_params
        state_node_ids, action_node_ids, next_state_node_ids = factor['cnodes'], factor['action'], factor['nextnodes']

        state_variable_assignments = [state_nodes[state_var] for state_var in state_node_ids]
        action_variable_assignments = [action_nodes[action_var] for action_var in action_node_ids]

        label_indices = self.get_label_indices(next_state_node_ids, next_state_nodes)

        state_action_indices = tf.concat(state_variable_assignments + action_variable_assignments, axis=1)
        next_state_factor_potential = tcl.flatten(tf.gather_nd(potential, state_action_indices))
        labels = tf.stop_gradient(tf.one_hot(indices=label_indices, depth=next_state_factor_potential.shape[1]))

        loss_for_potential = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels,
                                                                        logits=next_state_factor_potential,
                                                                        dim=-1,
                                                                        name='sl_loss_' + factor['name'])
        return loss_for_potential


    def get_loss_transition_cross_entropy(self, action_nodes, cross_state_factors, next_state_nodes, state_nodes):
        loss_transition_cross_entropy = None
        for factor in cross_state_factors:
            loss_for_potential = self.get_loss_for_transition_factor(action_nodes, factor, next_state_nodes,
                                                                     state_nodes)
            if loss_transition_cross_entropy is None:
                loss_transition_cross_entropy = loss_for_potential
            else:
                loss_transition_cross_entropy += loss_for_potential
        return loss_transition_cross_entropy


    def get_loss_for_reward_factors(self, reward_factors_instate, reward_factor_crossstate,
                                    state_nodes, next_state_nodes, action_nodes, reward_targets):
        next_state_factor_potential = None
        for factor in reward_factors_instate:
            reward = self.get_reward_value_instate(factor, state_nodes)
            if next_state_factor_potential is None:
                next_state_factor_potential = reward
            else:
                next_state_factor_potential += reward

        for factor in reward_factor_crossstate :
            reward = self.get_reward_value_crossstate(factor, state_nodes, next_state_nodes, action_nodes)
            if next_state_factor_potential is None:
                next_state_factor_potential = reward
            else:
                next_state_factor_potential += reward

        reward_mse_loss = 0.5 * tf.square(reward_targets - next_state_factor_potential)
        return reward_mse_loss

    def get_reward_value_instate(self, factor, state_nodes):
        potential = factor['Factor'][0].sl_params
        state_node_ids = factor['nodes']
        state_variable_assignments = [state_nodes[state_var] for state_var in state_node_ids]

        state_action_indices = tf.concat(state_variable_assignments, axis=1)
        reward = tf.gather_nd(potential, state_action_indices)
        return reward


    def get_reward_value_crossstate(self, factor, state_nodes, next_state_nodes, action_nodes):
        potential = factor['Factor'][0].sl_params
        state_node_ids, action_node_ids, next_state_node_ids = factor['cnodes'], factor['action'], factor['nextnodes']

        state_variable_assignments = [state_nodes[state_var] for state_var in state_node_ids]
        action_variable_assignments = [action_nodes[action_var] for action_var in action_node_ids]
        next_state_variable_assignments = [next_state_nodes[state_var] for state_var in next_state_nodes]

        state_action_indices = tf.concat(state_variable_assignments + action_variable_assignments + next_state_variable_assignments, axis=1)
        reward = tf.gather_nd(potential, state_action_indices)
        return reward

    def get_label_indices(self, next_state_node_ids, next_state_nodes):
        next_state_variable_assignments = None
        multiplier = 1
        for state_var in reversed(next_state_node_ids):
            if next_state_variable_assignments is None:
                next_state_variable_assignments = next_state_nodes[state_var]
            else:
                next_state_variable_assignments += multiplier * next_state_nodes[state_var]
            multiplier *= self.agent.all_state_dim[state_var]
        return next_state_variable_assignments

    def _build_network(self, name):
        input_r = tf.placeholder(tf.float32, [None])
        advantage = tf.placeholder(tf.float32, [None])
        target_v = tf.placeholder(tf.float32, [None])

        with tf.variable_scope(name):
            self.agent = FreewayBaseAgent(simulate_steps=self.SIM_STEPS,
                                          max_bp_steps=self.BP_STEPS,
                                          mult_fac=self.MULT_FAC,
                                          discount_factor=.99,
                                          # max_x=self.env_args['max_x'],
                                          # max_y=self.env_args['max_y'],
                                          # goal_position=self.env_args['goal_position'],
                                          # disappearance_probability=self.env_args['disappearance_probability'],
                                          scope=self.name,
                                          sequential=self.sequential)

            # Extract Policy and value nodes

            # input_s = self.agent.init_state_pl
            input_s = tf.placeholder(tf.float32, [None, np.sum(self.agent.all_state_dim)])
            input_sprime = tf.placeholder(tf.float32, [None, np.sum(self.agent.all_state_dim)])

            input_a = tf.placeholder(tf.int64, [None] + self.agent.action_dim)

            fab = self.agent.final_action_belief * Temperature

            if LAYER_OVER_POLICY:
                policy = tf_utils.fc(
                    fab,
                    np.sum(self.agent.action_dim),
                    activation_fn=tf.nn.softmax,
                    scope="policy",
                    initializer=tf_utils.normalized_columns_initializer(0.01)) + 1e-8
            else:
                policy = tf.nn.softmax(fab)[0] + 1e-8  # TODO:dirty fix by zhen, maybe a bug of tf (the suffix 0).
                # input_s1, input_s2, input_s3 = tf.split(self.agent.init_state_pl, [5,6,2], axis=1)
                # input_s_new = tf.expand_dims(tf.expand_dims(input_s1, 2),3) + \
                #               tf.expand_dims(tf.expand_dims(input_s2, 1), 3) + \
                #               tf.expand_dims(tf.expand_dims(input_s2, 1), 2)
                input_s_new = self.agent.init_state_pl
                input_s_new = tf.exp(input_s_new)
                input_s_new = tcl.flatten(input_s_new)

            layer1 = tf_utils.fc(
                input_s_new,
                300,
                scope="fc1",
                activation_fn=tf.nn.relu,
                initializer=tf.contrib.layers.variance_scaling_initializer(
                    mode="FAN_IN"))

            value = tf_utils.fc(layer1, 1, activation_fn=None,
                                scope="value", initializer=tf_utils.normalized_columns_initializer(1.0))

            input_a_cast = tf.cast(input_a, tf.float32)
            action_est = tf.reduce_sum(policy * input_a_cast, 1)
            # action_mask = tf.one_hot(input_a, 3, 1.0, 0.0)
            # action_est = tf.reduce_sum(policy * action_mask, 1)

        model_variables_rl = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name + '/rl_params')
        model_variables_sl = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name + '/sl_params')

        return input_s, input_sprime, input_r, input_a, advantage, target_v, policy, value, action_est, model_variables_rl, model_variables_sl

    def get_action(self, state, sess):
        state = np.reshape(state, [-1, np.sum(self.agent.all_state_dim)])
        feed_dict = {self.agent.init_state_pl: state}

        if (type(self.agent.infer_net) == InferNetPipeLine):
            feed_dict[self.agent.infer_net.simulate_steps] = self.SIM_STEPS
            feed_dict[self.agent.infer_net.max_steps] = self.SIM_STEPS + (self.BP_STEPS - 1) * 2
        policy = sess.run(self.policy, feed_dict)
        action = np.zeros([np.shape(state)[0], int(np.sum(self.agent.action_dim))])
        for idx, p in enumerate(policy) :
            action[idx, np.random.choice(range(int(np.sum(self.agent.action_dim))), p=p)] = 1
        return action

    def predict_policy(self, state, sess):
        state = np.reshape(state, [-1, np.sum(self.agent.all_state_dim)])
        feed_dict = {self.agent.init_state_pl: state}

        if type(self.agent.infer_net) == InferNetPipeLine:
            feed_dict[self.agent.infer_net.simulate_steps] = self.SIM_STEPS
            feed_dict[self.agent.infer_net.max_steps] = self.SIM_STEPS + (self.BP_STEPS - 1) * 2
        policy = sess.run(self.policy, feed_dict)
        return policy[0]

    def predict_value(self, state, sess):
        state = np.reshape(state, [-1, np.sum(self.agent.all_state_dim)])
        feed_dict = {self.agent.init_state_pl: state}
        if type(self.agent.infer_net) == InferNetPipeLine:
            feed_dict[self.agent.infer_net.simulate_steps] = self.SIM_STEPS
            feed_dict[self.agent.infer_net.max_steps] = self.SIM_STEPS + (self.BP_STEPS - 1) * 2
        return sess.run(self.value, feed_dict)
