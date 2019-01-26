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
# import lib.state
# import lib.action
# import lib.ObservationToState as Ob2State
# import platform
# import os


def to_valid_coordinate(cx, cy, screen_size):
    cx = min(max(cx, 0), screen_size[0])
    cy = min(max(cy, 0), screen_size[1])
    return cx, cy


def split_as_slices(var, slices_list, axis=1):
    all_dims = []
    for sls in slices_list:
        all_dims += sls
    splited_var = tf.split(var, all_dims, axis=axis)

    res = ()
    start_idx = 0
    for sls in slices_list:
        pres = splited_var[start_idx:(len(sls) + start_idx)]
        start_idx += len(sls)
        res += (pres, )
    return res
