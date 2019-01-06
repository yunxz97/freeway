import time
from copy import deepcopy

import numpy as np
import tensorflow as tf

import freeway_factors
from lib.basic_infer_unit import InferNetPipeLine, InferNetRNN

variable_mapping = {
    "chicken_y": 0,
    "car1_x": 1,
    "car2_x": 2,
    "car3_x": 3,
    "car4_x": 4,
    "car5_x": 5,
    "car6_x": 6,
    "car7_x": 7,
    "car8_x": 8,
    "car9_x": 9,
    "car10_x": 10,
    "car1_hit": 11,
    "car2_hit": 12,
    "car3_hit": 13,
    "car4_hit": 14,
    "car5_hit": 15,
    "car6_hit": 16,
    "car7_hit": 17,
    "car8_hit": 18,
    "car9_hit": 19,
    "car10_hit": 20,
    "hit": 21
}

variable_range = {
    "chicken_y": 210, # 15 ~ 195
    "car1_x": 160, # 8 ~ 160
    "car2_x": 160,
    "car3_x": 160,
    "car4_x": 160,
    "car5_x": 160,
    "car6_x": 160,
    "car7_x": 160,
    "car8_x": 160,
    "car9_x": 160,
    "car10_x": 160,
    "car1_hit": 2,
    "car2_hit": 2,
    "car3_hit": 2,
    "car4_hit": 2,
    "car5_hit": 2,
    "car6_hit": 2,
    "car7_hit": 2,
    "car8_hit": 2,
    "car9_hit": 2,
    "car10_hit": 2,
    "hit": 2
}

Temperature = 10


class FreewayAgent:
    def __init__(self, **kwargs):

        if 'simulate_steps' not in kwargs.keys():
            raise ValueError('Planning steps must be specified by argument simulate_steps!')

        for key, val in kwargs.items():
            setattr(self, key, val)

        if not hasattr(self, 'discount_factor'):
            self.discount_factor = 1

        if not hasattr(self, 'scope'):
            self.scope = "unnamed_worker"

        self.all_state_dim = [
            variable_range['chicken_y'],
            variable_range['car1_x'],
            variable_range['car2_x'],
            variable_range['car3_x'],
            variable_range['car4_x'],
            variable_range['car5_x'],
            variable_range['car6_x'],
            variable_range['car7_x'],
            variable_range['car8_x'],
            variable_range['car9_x'],
            variable_range['car10_x'],
            variable_range["car1_hit"],
            variable_range["car2_hit"],
            variable_range["car3_hit"],
            variable_range["car4_hit"],
            variable_range["car5_hit"],
            variable_range["car6_hit"],
            variable_range["car7_hit"],
            variable_range["car8_hit"],
            variable_range["car9_hit"],
            variable_range["car10_hit"],
            variable_range["hit"],
        ]
        
        self.action_dim = [3]

        with tf.variable_scope(self.scope + '_infer_net'):
            self.add_in_state_factor()
            self.add_cross_state_factor()
            self.infer_net = InferNetPipeLine(
                self.all_state_dim,
                self.action_dim,
                self.simulate_steps,
                self.in_state_factor,
                self.cross_state_factor,
                self.max_bp_steps
            )

            self.init_state_pl = self.infer_net.init_belief
            self.final_action_belief = self.infer_net.final_action

            self.final_state = self.infer_net.final_state

            self.policy = tf.nn.softmax(self.final_action_belief[0] * Temperature, axis=1) + 1e-8

            self.hidden1 = tf.concat([self.init_state_pl, self.final_action_belief[0]], axis=1)

            # Critic Network (Value Network)
            self.value = tf.layers.dense(
                # inputs=tf.concat([self.init_state_pl, self.final_action_belief[0][0]], axis=1),
                inputs=self.hidden1,
                units=1,
                name=self.scope + 'ValueLayer',
                # kernel_initializer=tf.initializers.zeros)
                kernel_initializer=tf.initializers.random_normal(0.01)
            )

    def add_in_state_factor(self):
        self.car_hit_factors = [freeway_factors.CarHitFactor(car=i+1, train=True) for i in range(10)]
        self.hit_factor = freeway_factors.HitFactor(train=True)
        self.dest_reward_factor = freeway_factors.DestinationRewardFactor(train=True)

        self.in_state_factor = [
            self.create_in_state_factor(
                [variable_mapping["chicken_y"], variable_mapping["car"+str(i+1)+"_x"],
                 variable_mapping["car"+str(i+1)+"_hit"]],
                self.car_hit_factors[i]
            ) for i in range(10)
        ]
        self.in_state_factor.append(
            self.create_in_state_factor(
                [variable_mapping["car" + str(i) + "_hit"] for i in range(1, 11)] + [variable_mapping["hit"]],
                self.hit_factor)
        )
        self.in_state_factor.append(
            self.create_in_state_factor(
                [variable_mapping["chicken_y"]],
                self.dest_reward_factor)
        )

    def create_in_state_factor(self, factor_nodes, factor):
        cfactor = dict()
        cfactor['name'] = factor.__class__.__name__
        cfactor['nodes'] = factor_nodes
        cfactor['Factor'] = [factor] * (self.simulate_steps)
        return cfactor

    def add_cross_state_factor(self):
        self.car_move_factors = [freeway_factors.CarMovementFactor(car=i+1, train=True) for i in range(10)]
        self.chicken_move_factor = freeway_factors.ChickenMovementFactor(train=True)

        self.cross_state_factor = [
            self.create_cross_state_factor(
                [variable_mapping["car"+str(i+1)+"_x"]],
                [],
                [variable_mapping["car"+str(i+1)+"_x"]],
                self.car_move_factors[i]
            ) for i in range(10)
        ]

        self.cross_state_factor.append(
            self.create_cross_state_factor(
                [variable_mapping["chicken_y"], variable_mapping['hit']],
                [0],
                [variable_mapping["chicken_y"]],
                self.chicken_move_factor)
        )

    def create_cross_state_factor(self, cfactor_nodes, action_nodes, next_nodes, factor):
        cfactor = dict()
        cfactor['name'] = factor.__class__.__name__
        cfactor['cnodes'] = cfactor_nodes
        cfactor['action'] = action_nodes
        cfactor['nextnodes'] = next_nodes
        cfactor['Factor'] = [factor] * (self.simulate_steps - 1)
        
        return cfactor
