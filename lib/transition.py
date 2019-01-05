
from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.layers as tcl

# from . import PGM as PGM
# from . import state as state
# from . import utils as utils

MaxPossibleSteps = 31 # 0 - 30, hardcoded to one hot vector
V_table=[[-12, -4],[-12, -3],[-12, -2],[-12, -1],[-12, 0],[-12, 1],[-12, 2],[-12, 3],[-12, 4],[-11, -6],[-11, -5],[-11, 5],[-11, 6],[-10, -8],[-10, -7],[-10, 7],[-10, 8],[-9, -9],[-9, -8],[-9, 8],[-9, 9],[-8, -10],[-8, -9],[-8, 9],[-8, 10],[-7, -10],[-7, 10],[-6, -11],[-6, 11],[-5, -11],[-5, 11],[-4, -12],[-4, 12],[-3, -12],[-3, 12],[-2, -12],[-2, 12],[-1, -12],[-1, 12],[0, -12],[0, 12],[1, -12],[1, 12],[2, -12],[2, 12],[3, -12],[3, 12],[4, -12],[4, 12],[5, -11],[5, 11],[6, -11],[6, 11],[7, -10],[7, 10],[8, -10],[8, -9],[8, 9],[8, 10],[9, -9],[9, -8],[9, 8],[9, 9],[10, -8],[10, -7],[10, 7],[10, 8],[11, -6],[11, -5],[11, 5],[11, 6],[12, -4],[12, -3],[12, -2],[12, -1],[12, 0],[12, 1],[12, 2],[12, 3],[12, 4]]
PossibleV = int(len(V_table))

MinusInfinity = -10000

MaxX = 255
MaxY = 255


class Factors:

    ExpandHOPs = dict()
    
    def __init__(self):
        pass

    @staticmethod
    def DualVal(Hops, Msgs):
        if Hops is None:
            hops = tf.expand_dims(Factors.potential, axis=0)
        else:
            expand_hops = []
            for h in Hops:
                if(id(h) not in Factors.ExpandHOPs.keys()):
                    Factors.ExpandHOPs[id(h)] = tf.expand_dims(h.potential, axis=0)
                expand_hops.append(Factors.ExpandHOPs[id(h)])
            
            hops = tf.concat(expand_hops, axis=0)
            
        MsgDims = len(Msgs[0])
        AllMsgs = [tf.concat([tf.expand_dims(msg[lidx], axis=0) for msg in Msgs], axis=0)
                   for lidx in range(MsgDims)]
            
        for lidx in range(MsgDims):
            hat_bi = - AllMsgs[lidx]
            for expand_idx in range(MsgDims):
                if(expand_idx == lidx):
                    continue
                hat_bi = tf.expand_dims(hat_bi, expand_idx + 1)
            hops += hat_bi
        return tf.reduce_sum(tf.reduce_max(hops, axis=[lidx + 1 for lidx in range(MsgDims)]))

    @staticmethod
    def Decoding(Hops, NodeBeliefs, Msgs):
        zeros_brd_cast = tf.reduce_sum(tf.zeros_like(NodeBeliefs[0][0]), axis=1, keep_dims=True)
        assert(len(Msgs) == len(NodeBeliefs))
        if Hops is None:
            hops = tf.expand_dims(Factors.potential, axis=0)
        else:
            expand_hops = []
            chop = Hops[0].potential
            
            for idx in range(2, len(chop.shape) + 1):
                zeros_brd_cast = tf.expand_dims(zeros_brd_cast, idx)
            
            for h in Hops:
                if(id(h) not in Factors.ExpandHOPs.keys()):
                    
                    Factors.ExpandHOPs[id(h)] = tf.expand_dims(h.potential, axis=0) 
                expand_hops.append(Factors.ExpandHOPs[id(h)] + zeros_brd_cast)
            hops = tf.concat(expand_hops, axis=0)
            
        MsgDims = len(NodeBeliefs[0])

        LocalBeliefs = [tf.concat([bn[lidx] for bn in NodeBeliefs], axis=0)
                        for lidx in range(MsgDims)]
        AllMsgs = [tf.concat([msg[lidx] for msg in Msgs], axis=0)
                        for lidx in range(MsgDims)]

        for lidx in range(MsgDims):
            hat_msg = LocalBeliefs[lidx] - AllMsgs[lidx]
            c_marginalized = []
            for expand_idx in range(MsgDims):
                if(expand_idx == lidx):
                    continue
                c_marginalized.append(expand_idx + 1)
                hat_msg = tf.expand_dims(hat_msg, expand_idx + 1)
            hops += hat_msg
        hops_shape = hops.shape

        dim_prod = []
        for y in reversed(hops_shape[1:]):
            y = int(y)
            if(len(dim_prod) >= 1):
                dim_prod.append(y * dim_prod[-1])
            else:
                dim_prod.append(y)
        argmax_idx = tf.argmax(tf.reshape(hops, [-1, dim_prod[-1]]), axis=1)
        output_idx = []

        
        
        for idx in range(1, len(dim_prod)):
            if(idx == 1):
                output_idx.append(argmax_idx // tf.to_int64(dim_prod[idx]))
            output_idx.append(argmax_idx % (tf.to_int64(dim_prod[idx])) / tf.to_int64(dim_prod[idx + 1]))
        return output_idx
    
    
    @staticmethod
    def Primal(Hops, Msgs, xHat):
        zeros_brd_cast = tf.reduce_sum(tf.zeros_like(NodeBeliefs[0][0]), axis=1, keep_dims=True)
        assert(len(Msgs) == len(NodeBeliefs))
        if Hops is None:
            hops = tf.expand_dims(Factors.potential, axis=0)
        else:
            expand_hops = []
            chop = Hops[0].potential
            
            for idx in range(2, len(chop.shape) + 1):
                zeros_brd_cast = tf.expand_dims(zeros_brd_cast, idx)
            
            for h in Hops:
                if(id(h) not in Factors.ExpandHOPs.keys()):
                    
                    Factors.ExpandHOPs[id(h)] = tf.expand_dims(h.potential, axis=0) 
                expand_hops.append(Factors.ExpandHOPs[id(h)] + zeros_brd_cast)
            hops = tf.concat(expand_hops, axis=0)
            
        MsgDims = len(Msgs[0])
        AllMsgs = [tf.concat([msg[lidx] for msg in Msgs], axis=0)
                   for lidx in range(MsgDims)]

        for lidx in range(MsgDims):
            hat_msg =  - AllMsgs[lidx]
            c_marginalized = []
            for expand_idx in range(MsgDims):
                if(expand_idx == lidx):
                    continue
                c_marginalized.append(expand_idx + 1)
                hat_msg = tf.expand_dims(hat_msg, expand_idx + 1)
            hops += hat_msg
            
        final_idx = []
        for pidx, indices in zip(range(len(xHat)), xHat):
            pidx = tf.to_int64(pidx)
            
    @staticmethod
    def LoopyBP(Hops, NodeBeliefs, Msgs, redis_factor=None, damping=0.25):
        """
        Now midify to pencial updating.
        \lambda_{f\rightarrow i}(x_i) += \frac{1}{2}(\max_{x_{f\setminus i}}b_f(x_f) - b_i(x_i)}
        """
        ozeros_brd_cast = tf.reduce_sum(tf.zeros_like(NodeBeliefs[0][0]), axis=1, keep_dims=True)
        zeros_brd_cast = ozeros_brd_cast
        assert(len(Msgs) == len(NodeBeliefs))
        if Hops is None:
            hops = tf.expand_dims(Factors.potential, axis=0)
        else:
            expand_hops = []
            chop = Hops[0].potential
            print(chop)
            
            for idx in range(2, len(chop.get_shape()) + 1):
                zeros_brd_cast = tf.expand_dims(zeros_brd_cast, idx)
            
            for h in Hops:
                if(id(h) not in Factors.ExpandHOPs.keys()):
                    Factors.ExpandHOPs[id(h)] = tf.expand_dims(h.potential, axis=0) 
                expand_hops.append(Factors.ExpandHOPs[id(h)] + zeros_brd_cast)
            hops = tf.concat(expand_hops, axis=0)
            
        MsgDims = len(NodeBeliefs[0])

        LocalBeliefs = [tf.concat([bn[lidx] for bn in NodeBeliefs], axis=0)
                        for lidx in range(MsgDims)]
        AllMsgs = [tf.concat([msg[lidx] for msg in Msgs], axis=0)
                        for lidx in range(MsgDims)]
        if(redis_factor is not None):
            print(redis_factor)
            print(ozeros_brd_cast)
            RedisFactor = [tf.concat([rf[lidx]+ozeros_brd_cast for rf in redis_factor], axis=0)
                           for lidx in range(MsgDims)]
        else:
            RedisFactor = None

        sum_local_belief_msg = []
        to_be_marginalized = []
        for lidx in range(MsgDims):
            hat_msg = LocalBeliefs[lidx] - AllMsgs[lidx]
            c_marginalized = []
            for expand_idx in range(MsgDims):
                if(expand_idx == lidx):
                    continue
                c_marginalized.append(expand_idx + 1)
                hat_msg = tf.expand_dims(hat_msg, expand_idx + 1)

            hops += hat_msg
            to_be_marginalized.append(c_marginalized)

        inv_factor_size = 1.0 / MsgDims
        
        if RedisFactor is not None:
            nLocalBeliefs = [RedisFactor[lidx] * tf.reduce_max(hops, to_be_marginalized[lidx]) for lidx in range(MsgDims)]
        else:
            nLocalBeliefs = [inv_factor_size * tf.reduce_max(hops, to_be_marginalized[lidx]) for lidx in range(MsgDims)]
        nmsg_merged = [AllMsgs[lidx] + nLocalBeliefs[lidx] - LocalBeliefs[lidx]
                       for lidx in range(MsgDims)]

        #split_dim0 = [int(b[0].shape[0]) for b in NodeBeliefs]
        nBelief = [tf.split(nLocalBeliefs[lidx], len(NodeBeliefs), axis=0) for lidx in range(MsgDims)]
        nMsg = [tf.split(nmsg_merged[lidx], len(NodeBeliefs), axis=0) for lidx in range(MsgDims)]

        return [[nBelief[lidx][idx] for lidx in range(MsgDims)]  for idx in range(len(NodeBeliefs))], \
               [[nMsg[lidx][idx] for lidx in range(MsgDims)] for idx in range(len(NodeBeliefs))]


        
        
    
    def _MaxMarginal(self, NodeBeliefs, Msgs, damping=1.0, UseBP=True, verbose=False):
        assert(len(NodeBeliefs) == len(Msgs))
        if(len(NodeBeliefs) == 0):
            if not verbose:
                return Msgs, NodeBeliefs
            else:
                return Msgs, NodeBeliefs, []
        MsgDims = len(NodeBeliefs[0])
        inv_msg_dims = 1.0 / MsgDims
        HigherBeliefs = tf.expand_dims(self.potential, axis=0)
        LocalBeliefs = []
        AllMsgs = []
        for lidx in range(MsgDims):
            LocalBeliefs.append(tf.concat([tf.expand_dims(bn[lidx], axis=0) for bn in NodeBeliefs], axis=0))
            AllMsgs.append(tf.concat([tf.expand_dims(msg[lidx], axis=0) for msg in Msgs], axis=0))

            localbelief = LocalBeliefs[lidx]
            localmsg = AllMsgs[lidx]
            
            for expand_idx in range(MsgDims):
                if(expand_idx) == lidx:
                    continue
                localbelief = tf.expand_dims(localbelief, expand_idx + 1)
                localmsg = tf.expand_dims(localmsg, expand_idx + 1)
            HigherBeliefs = HigherBeliefs + localbelief - localmsg

        
        #print(inv_msg_dims)
        nLocalBelief = []
        for lidx in range(MsgDims):

            to_be_marginalized = []
            localbelief = LocalBeliefs[lidx]

            for expand_idx in range(MsgDims):
                if(expand_idx) == lidx:
                    continue
                localbelief = tf.expand_dims(localbelief, expand_idx + 1)
                to_be_marginalized.append(expand_idx + 1)

            if(UseBP):
                AllMsgs[lidx] = (1 - damping) * AllMsgs[lidx] + damping * tf.reduce_max(HigherBeliefs - localbelief, axis=to_be_marginalized)
            else:
                LowerBeliefs = inv_msg_dims * tf.reduce_max(HigherBeliefs, axis=to_be_marginalized)
                AllMsgs[lidx] += LowerBeliefs - LocalBeliefs[lidx]
                nLocalBelief.append(LowerBeliefs)
                
        NMsg = [[AllMsgs[lidx][idx] for lidx in range(MsgDims)] for idx in range(len(Msgs))]
        if not UseBP:
            NBelief = [[nLocalBelief[lidx][idx] for lidx in range(MsgDims)] for idx in range(len(Msgs))]
            if not verbose:
                return NMsg, NBelief
            else:
                return NMsg, NBelief, [HigherBeliefs, LocalBeliefs, AllMsgs]
        return NMsg

# from .in_state_factor import SoilderMineralActive, MineralActiveTransition, SoilderMineralActiveXY, SoilderMineralActiveJoint, SoilderMineralActiveJointConv, SoilderDisPos, IsTerminateFactor, SoilderDisPosTerminal
# from .cross_state_factor import SelectFactor, MineralShardsTransition, MoveFactorCommandsVid, MoveFactorCommandsSteps, MoveFactorPosition, MoveFactorPositionJoint

# from .TreeFactor import TreeFactor
