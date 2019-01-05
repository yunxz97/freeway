import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tcl
from tensorflow.keras.models import Model
import tf_utils
from freeway_agent import FreewayAgent
from lib.basic_infer_unit import InferNetPipeLine, InferNetRNN
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, Lambda


def build_dual_decomposition_model(sim_steps, bp_iters):
    agent = FreewayAgent(
        simulate_steps=sim_steps,
        max_bp_steps=bp_iters, 
        mult_fac=0,
        discount_factor=3,
        scope='ddn',
    )

    input_s = agent.init_state_pl
    final_action_belief = agent.final_action_belief
    final_state = agent.final_state
    policy = agent.policy

    # return input_s, final_action_belief, final_state
    return agent, policy, final_state
