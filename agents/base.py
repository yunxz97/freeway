from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from constants import SCREEN_WIDTH, SCREEN_HEIGHT, SL_ENABLE
import tensorflow as tf
from lib.BasicInferUnit import InferNetPipeLine
import factors.base as freeway_factors
from constants import BP_STEPS, SIM_STEPS, TEMPERATURE, USE_CONV

BP = True

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

variable_range = [
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    SCREEN_WIDTH,
    SCREEN_WIDTH,
    SCREEN_WIDTH,
    SCREEN_WIDTH,
    SCREEN_WIDTH,
    SCREEN_WIDTH,
    SCREEN_WIDTH,
    SCREEN_WIDTH,
    SCREEN_WIDTH,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2
]


class FreewayBaseAgent:
    def __init__(self, variable_range=variable_range, factors=freeway_factors, **kwargs):

        if 'simulate_steps' not in kwargs.keys():
            raise ValueError('Planning steps must be specified by argument simulate_steps!')

        for key, val in kwargs.items():
            setattr(self, key, val)

        if not hasattr(self, 'discount_factor'):
            self.discount_factor = .99

        if not hasattr(self, 'scope'):
            self.scope = "Main"

        self.all_state_dim = variable_range
        self.action_dim = [3]
        self.simulate_steps = SIM_STEPS
        self.max_bp_steps = BP_STEPS

        # with tf.variable_scope(self.scope + '_infer_net'):
        self.factors = factors
        self.reward_factors_instate_st = []
        self.reward_factors_instate_stprime = []
        self.reward_factors_crossstate = []
        self.state_transition_factors = []
        self.in_state_supervised_factors = []
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
        self.init_action_pl = self.infer_net.init_action
        self.obj_v = self.infer_net.objv
        self.final_action_belief = self.infer_net.final_final_action
        self.final_state = self.infer_net.final_state

        # self.final_state = self.infer_net.final_state

        self.policy = tf.nn.softmax(self.final_action_belief[0] * TEMPERATURE) + 1e-8


        # self.hidden1 = tf.concat([self.init_state_pl, self.final_action_belief[0]], axis=1)
        #
        # Critic Network (Value Network)
        self.value = tf.layers.dense(
            # inputs=tf.concat([self.init_state_pl, self.final_action_belief[0][0]], axis=1),
            inputs=self.policy,
            units=1,
            name=self.scope + 'ValueLayer',
            # kernel_initializer=tf.initializers.zeros)
            kernel_initializer=tf.initializers.random_normal(0.01))

    def add_in_state_factor(self):
        self.car_hit_factors = [self.factors.CarHitFactor(car=i+1, train=True) for i in range(10)]
        self.hit_factor = self.factors.HitFactor(train=True)
        # self.dest_reward_factor = self.factors.DestinationRewardFactor(train=True)
        self.dest_reward_factor = self.factors.DestinationRewardFactor(train=True)

        self.in_state_factor = [
            self.create_in_state_factor(
                [variable_mapping["chicken_y"], variable_mapping["car"+str(i+1)+"_x"],
                 variable_mapping["car"+str(i+1)+"_hit"]],
                self.car_hit_factors[i],
                gather_nodes= [variable_mapping["chicken_y"], variable_mapping["car"+str(i+1)+"_x"]],
                target_nodes= [variable_mapping["car"+str(i+1)+"_hit"]]
            ) for i in range(10)
        ]
        self.in_state_factor.append(
            self.create_in_state_factor(
                [variable_mapping["car" + str(i) + "_hit"] for i in range(1, 11)] + [variable_mapping["hit"]],
                self.hit_factor)
        )
        # self.in_state_factor.append(
        #     self.create_in_state_factor(
        #         [variable_mapping["chicken_y"]],
        #         self.dest_reward_factor)
        # )
        self.in_state_factor.append(
            self.create_in_state_factor(
                [variable_mapping["chicken_y"]],
                self.dest_reward_factor)
        )
        if SL_ENABLE:
            self.reward_factors_instate_stprime.append(self.in_state_factor[-1])

    def create_in_state_factor(self, factor_nodes, factor, gather_nodes=None, target_nodes=None):
        if gather_nodes is None:
            gather_nodes = []
        if target_nodes is None:
            target_nodes = []
        cfactor = dict()
        cfactor['name'] = factor.__class__.__name__
        cfactor['nodes'] = factor_nodes
        cfactor['gather_nodes'] = gather_nodes
        cfactor['target_nodes'] = target_nodes
        cfactor['Factor'] = [factor] * (self.simulate_steps)
        return cfactor

    def add_cross_state_factor(self):
        if USE_CONV:
            self.car_move_factors = [self.factors.CarMovementConvFactor(car=i+1, train=True) for i in range(10)]
        else:
            self.car_move_factors = [self.factors.CarMovementFactor(car=i + 1, train=True) for i in range(10)]
        self.chicken_move_factor = self.factors.ChickenMovementFactor(train=True)

        self.cross_state_factor = [
            self.create_cross_state_factor(
                [variable_mapping["car"+str(i+1)+"_x"]],
                [],  # dummy action
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
        self.state_transition_factors.extend(self.cross_state_factor)

    def create_cross_state_factor(self, cfactor_nodes, action_nodes, next_nodes, factor):
        cfactor = dict()
        cfactor['name'] = factor.__class__.__name__
        cfactor['cnodes'] = cfactor_nodes
        cfactor['action'] = action_nodes
        cfactor['nextnodes'] = next_nodes
        cfactor['Factor'] = [factor] * (self.simulate_steps - 1)
        
        return cfactor
