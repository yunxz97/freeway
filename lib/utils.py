# utils.py --- 
# Copyright (C) 2017 
# Author: Zhen Zhang  <zhangz@comp.nus.edu.sg>
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License as
#  published by the Free Software Foundation, either version 3 of the
#  License, or (at your option) any later version.

#  This program is distributed in the hope that it will be useful, but
#  WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#  General Public License for more details.

#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see http://www.gnu.org/licenses/.

import tensorflow as tf
# import numpy as np
# from . import state
# from . import action
# from . import ObservationToState as Ob2State
# import pysc2
# from absl import flags
# from absl.flags import FLAGS
# from pysc2.agents import random_agent
# from pysc2.agents import scripted_agent
# from copy import deepcopy


# from pysc2.env import sc2_env
# from pysc2.env import environment
# from pysc2.lib import actions
# from pysc2.maps import mini_games

# def spec_to_pl(specs):
#     res = []
#     for key, values in specs.items():
#         cstate = list()
#         for idx in range(values['dim']):
#             s = tf.placeholder(tf.float32, [None, values['size'][idx]])
#             cstate.append(s)
#         res += cstate
#     return res


# def spec_dims(specs):
#     res = []

#     for key, values in specs.items():
#         for idx in range(values['dim']):
#             res +=[values['size'][idx]]

#     return res

# def mineral_init_env(env, random_agent):
#     obs = env.reset()
#     agent=random_agent
#     prev_state_tensor, _ = Ob2State.extract_unit_mproperties_mineral_shards_v2(obs)
#     cidx = 0
#     for mini_agnets in prev_state_tensor:
#         action = agent.step(obs[0], args=[[0], [mini_agnets[0], mini_agnets[1]]])
#         obs = env.step([action])
#         action = agent.step(obs[0], function_id=4, args=[[1], [cidx]])
#         cidx = cidx + 1
#     print('Select Groups Created as ')
#     print(obs[0].observation['single_select'])
#     return obs
        
# def init(FLAGS) :
#     '''
#     Function: Initializes the environment to sample transitions

#     Args:
#         No Arguments

#     Returns:
#         env : The SC2env
#         observation_spec : The observation specification of the environment
#         action_spec : The action specification of the environment

#     '''

#     env = sc2_env.SC2Env(map_name=FLAGS.map,
#                          agent_race=None,
#                          bot_race=None,
#                          difficulty=None,
#                          step_mul=8, #three steps persecond 24/3 = 8
#                          game_steps_per_episode=2500, #default value
#                          screen_size_px=(FLAGS.screen_resolution, FLAGS.screen_resolution),
#                          minimap_size_px=(64, 64),
#                          visualize=FLAGS.render)
#     print(env.action_spec())
#     print(env.observation_spec())
#     print('Initializing temporary environment to retrive action_spec...')
#     action_spec = env.action_spec()
#     print('Initializing temporary environment to retrive observation_spec...')
#     observation_spec = env.observation_spec()
#     return env, observation_spec, action_spec


# def feed_state(feed_dict, state_pl, state_tensor, state_spec):
#     final_tensor = []
#     for i in range(len(state_tensor)):
#         state_tensor_node = lib.state.generate_state_tensor(state_spec,
#                                                             state_tensor[i])
#         state_tensor_node = np.concatenate(state_tensor_node, axis=1)
#         final_tensor.append(state_tensor_node)
        
#     final_tensor = np.concatenate(final_tensor, axis=1)
#     feed_dict[state_pl] = final_tensor
#     #print(state_pl)
#     #print(final_tensor.shape)


# def feed_action(feed_dict, action_pl, action_tensor):
#     feed_dict[action_pl] = np.concatenate(action_tensor, axis=1)



def split_as_slices(var, slices_list):
    all_dims = []
    for sls in slices_list:
        all_dims += sls
    splited_var = tf.split(var, all_dims, axis=1)

    res = ()
    start_idx = 0
    for sls in slices_list:
        pres = splited_var[start_idx:(len(sls) + start_idx)]
        start_idx+= len(sls)
        res += (pres,)
    return res
