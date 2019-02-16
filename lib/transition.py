# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

# import gin
import numpy as np
import tensorflow as tf
from .common_ops import to_log_scale
from .ops_utils import get_MRFInferGPU
import tensorflow.contrib.layers as tcl

MaxPossibleSteps = 31  # 0 - 30, hardcoded to one hot vector
V_table = [[-12, -4], [-12, -3], [-12, -2], [-12, -1], [-12, 0], [-12, 1],
           [-12, 2], [-12, 3], [-12, 4], [-11, -6], [-11, -5], [-11, 5],
           [-11, 6], [-10, -8], [-10, -7], [-10, 7], [-10, 8], [-9, -9],
           [-9, -8], [-9, 8], [-9, 9], [-8, -10], [-8, -9], [-8, 9], [-8, 10],
           [-7, -10], [-7, 10], [-6, -11], [-6, 11], [-5, -11], [-5, 11],
           [-4, -12], [-4, 12], [-3, -12], [-3, 12], [-2, -12], [-2, 12],
           [-1, -12], [-1, 12], [0, -12], [0, 12], [1, -12], [1, 12], [2, -12],
           [2, 12], [3, -12], [3, 12], [4, -12], [4, 12], [5, -11], [5, 11],
           [6, -11], [6, 11], [7, -10], [7, 10], [8, -10], [8, -9], [8, 9],
           [8, 10], [9, -9], [9, -8], [9, 8], [9, 9], [10, -8], [10, -7],
           [10, 7], [10, 8], [11, -6], [11, -5], [11, 5], [11, 6], [12, -4],
           [12, -3], [12, -2], [12, -1], [12, 0], [12, 1], [12, 2], [12, 3],
           [12, 4]]
PossibleV = int(len(V_table))

MinusInfinity = -10000

MaxX = 255
MaxY = 255

MRFInferGPU = get_MRFInferGPU()

feed_dict = None


class Factors:

    ExpandHOPs = dict()

    def __init__(self, potential=None, name=None, trainable=False):
        """
        """
        if (potential is not None):
            assert (name is not None)
            if (isinstance(potential, tf.Variable)):
                self.potential = potential
            else:
                with tf.variable_scope(name):
                    self.potential = tf.get_variable(
                        'potential', dtype=np.float32, initializer=potential)
            self.belief = self.potential

    @staticmethod
    def ConcatLocalVariables(LocalVars):
        nvars = len(LocalVars[0])
        res = [
            tf.concat([lv[lidx] for lv in LocalVars], axis=0)
            for lidx in range(nvars)
        ]
        return res

    @staticmethod
    def LoopyBP(Hops, NodeBeliefs, Msgs, redis_factor=None, damping=0.25):
        """
        \\[
        \\lambda_{f\\rightarrow i}(x_i) +=
        \\frac{1}{2}(\\max_{x_{f\\setminus i}}b_f(x_f) - b_i(x_i)}
        \\]
        """
        # print('node potentials: ', NodeBeliefs)

        assert (len(Msgs) == len(NodeBeliefs))
        batch_size = tf.shape(NodeBeliefs[0][0])[0]
        expand_hops = []
        for h in Hops:
            # if(id(h) not in Factors.ExpandHOPs.keys()):
            #     Factors.ExpandHOPs[id(h)] = tf.expand_dims(
            #         h.potential, axis=0)
            prev_shape = [int(shp) for shp in h.potential.shape]
            chop = tf.reshape(h.potential, [1, np.prod(prev_shape)])
            brd_shape = [batch_size, np.prod(prev_shape)]
            chop = tf.broadcast_to(chop, brd_shape)
            chop = tf.reshape(chop, [batch_size] + prev_shape)
            expand_hops.append(chop)
        hops = tf.concat(expand_hops, axis=0)

        MsgDims = len(NodeBeliefs[0])

        LocalBeliefs = Factors.ConcatLocalVariables(NodeBeliefs)
        AllMsgs = Factors.ConcatLocalVariables(Msgs)

        sp_shapes = [int(lbelief.shape[1]) for lbelief in LocalBeliefs]
        mlb = tf.concat(LocalBeliefs, axis=1)
        mmsg = tf.concat(AllMsgs, axis=1)

        upd_hops = MRFInferGPU.msg_gather(hops, LocalBeliefs, AllMsgs)
        nLocalBeliefs = MRFInferGPU.upd_msg(upd_hops,
                                            [tf.float32] * len(LocalBeliefs))

        # nmbeliefs, nmsg_merged = MRFInferGPU.upd_msg_belief(
        #     mlb, tf.concat(nLocalBeliefs, axis=1), mmsg)
        # nLocalBeliefs = tf.split(nmbeliefs, sp_shapes, axis=1)
        mhat_msgs = mlb - mmsg
        nmsg_merged = tf.split(tf.concat(nLocalBeliefs, axis=1) - mhat_msgs, sp_shapes, axis=1)
        # nmsg_merged = tf.split(nmsg_merged, sp_shapes, axis=1)

        nBelief = [
            tf.split(nLocalBeliefs[lidx], len(NodeBeliefs), axis=0)
            for lidx in range(MsgDims)
        ]
        nMsg = [
            tf.split(nmsg_merged[lidx], len(NodeBeliefs), axis=0)
            for lidx in range(MsgDims)
        ]

        return [[nBelief[lidx][idx] for lidx in range(MsgDims)]
                for idx in range(len(NodeBeliefs))], \
            [[nMsg[lidx][idx]
              for lidx in range(MsgDims)] for idx in range(len(NodeBeliefs))]

    def getTransitionLoss(self, gather_indices, labels, all_labels, name):
        #TODO: rather than passing gather_indices
        # we should pass a list of assignments, i.e., the assignments for each
        # node, and then we combine them inside this function
        next_state_factor_potential = tcl.flatten(
            tf.gather_nd(self.sl_params, gather_indices)
        )  #If no sl params, not to be learnt using Supervised Learning
        labels = tf.stop_gradient(
            tf.one_hot(
                indices=labels, depth=next_state_factor_potential.shape[1]))
        loss_for_potential = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=labels,
            logits=next_state_factor_potential,
            dim=-1,
            name='sl_loss_' + name)
        return loss_for_potential

    def getRewardValue(self, gather_indices):
        return tf.gather_nd(self.sl_params, gather_indices)

    def add_RL_Params(self):
        print("No RL Params")

    def _MaxMarginal(self,
                     NodeBeliefs,
                     Msgs,
                     damping=1.0,
                     UseBP=True,
                     verbose=False):
        assert (len(NodeBeliefs) == len(Msgs))
        if (len(NodeBeliefs) == 0):
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
            LocalBeliefs.append(
                tf.concat(
                    [tf.expand_dims(bn[lidx], axis=0) for bn in NodeBeliefs],
                    axis=0))
            AllMsgs.append(
                tf.concat([tf.expand_dims(msg[lidx], axis=0) for msg in Msgs],
                          axis=0))

            localbelief = LocalBeliefs[lidx]
            localmsg = AllMsgs[lidx]

            for expand_idx in range(MsgDims):
                if (expand_idx) == lidx:
                    continue
                localbelief = tf.expand_dims(localbelief, expand_idx + 1)
                localmsg = tf.expand_dims(localmsg, expand_idx + 1)
            HigherBeliefs = HigherBeliefs + localbelief - localmsg

        # print(inv_msg_dims)
        nLocalBelief = []
        for lidx in range(MsgDims):

            to_be_marginalized = []
            localbelief = LocalBeliefs[lidx]

            for expand_idx in range(MsgDims):
                if (expand_idx) == lidx:
                    continue
                localbelief = tf.expand_dims(localbelief, expand_idx + 1)
                to_be_marginalized.append(expand_idx + 1)

            if (UseBP):
                AllMsgs[lidx] = (1 - damping) * AllMsgs[lidx] + damping * \
                    tf.reduce_max(HigherBeliefs - localbelief,
                                  axis=to_be_marginalized)
            else:
                LowerBeliefs = inv_msg_dims * \
                    tf.reduce_max(HigherBeliefs, axis=to_be_marginalized)
                AllMsgs[lidx] += LowerBeliefs - LocalBeliefs[lidx]
                nLocalBelief.append(LowerBeliefs)

        NMsg = [[AllMsgs[lidx][idx] for lidx in range(MsgDims)]
                for idx in range(len(Msgs))]
        if not UseBP:
            NBelief = [[nLocalBelief[lidx][idx] for lidx in range(MsgDims)]
                       for idx in range(len(Msgs))]
            if not verbose:
                return NMsg, NBelief
            else:
                return NMsg, NBelief, [HigherBeliefs, LocalBeliefs, AllMsgs]
        return NMsg

    @staticmethod
    def expand_hops(hops, batch_size, broadcast=False):
        res = []
        for h in hops:
            prev_shape = [int(shp) for shp in h.potential.shape]
            chop = tf.reshape(h.potential, [1] + prev_shape)
            if broadcast:
                chop = tf.reshape(h.potential, [1, np.prod(prev_shape)])
                brd_shape = [batch_size, np.prod(prev_shape)]
                chop = tf.broadcast_to(chop, brd_shape)
                chop = tf.reshape(chop, [batch_size] + prev_shape)
            res.append(chop)
        res = tf.concat(res, axis=0)
        return res

    @staticmethod
    def ravel_assignments(assignments, shape):
        assert (len(assignments) == len(shape))
        res = assignments[0]
        for idx in range(1, len(assignments)):
            res = res * shape[idx] + assignments[idx]
        return res

    @staticmethod
    def primal(hops, assignments):
        batch_size = tf.shape(assignments[0][0])[0]
        nassignments = [
            tf.concat([assign[lidx] for assign in assignments], axis=0)
            for lidx in range(len(assignments[0]))
        ]
        hops = Factors.expand_hops(hops, batch_size, True)
        hop_shapes = [int(s) for s in list(hops.shape[1:])]
        nidx = Factors.ravel_assignments(nassignments, list(hops.shape[1:]))
        hops = tf.reshape(hops, [batch_size, np.prod(hop_shapes)])
        return tf.batch_gather(hops, nidx)

    @staticmethod
    def partial_primal(hops, partial_assignments, partial_idx, beliefs, msgs):
        ndims = len(hops[0].potential.shape)
        # print(partial_assignments)
        batch_size = tf.shape(partial_assignments[0])[0]
        hops = Factors.expand_hops(hops, batch_size, True)

        msgs_ = Factors.ConcatLocalVariables(msgs)
        beliefs_ = Factors.ConcatLocalVariables(beliefs)

        for idx in partial_idx:
            beliefs_[idx] = tf.fill(
                [batch_size] + list(beliefs_[idx].shape[1:]), 0.0)
        hops = MRFInferGPU.msg_gather(hops, beliefs_, msgs_)

        inc_dims = [idx + 1 for idx in partial_idx]
        exc_dims = [idx + 1 for idx in range(ndims) if idx not in partial_idx]

        inc_shapes = [int(hops.shape[idx]) for idx in inc_dims]
        exc_shapes = [int(hops.shape[idx]) for idx in exc_dims]

        final_permute_order = [0] + inc_dims + exc_dims

        hops = tf.transpose(hops, final_permute_order)
        nidx = Factors.ravel_assignments(partial_assignments, inc_shapes)
        hops = tf.reshape(hops, [batch_size, np.prod(inc_shapes)] + exc_shapes)
        partial_potential = tf.batch_gather(hops, nidx)

        if len(exc_dims) == 0:
            return partial_potential

        partial_potential = tf.reshape(
            partial_potential,
            [batch_size, tf.shape(nidx)[1],
             np.prod(exc_shapes)])

        return tf.reduce_max(partial_potential, axis=2)

    @staticmethod
    def max_marginal_assignment(Higher_Order_Potentials,
                                Msgs,
                                NodeBeliefs,
                                verbose=False):
        MsgDims = len(Msgs[0])
        HigherBeliefs = tf.concat([
            tf.expand_dims(hop.potential, axis=0)
            for hop in Higher_Order_Potentials
        ],
                                  axis=0)
        HigherBeliefs = tf.expand_dims(HigherBeliefs, axis=1)
        # print("HB",HigherBeliefs)
        '''
        Create \\theta^{\\delta}_f
        '''
        for lidx in range(MsgDims):
            # print(Msgs[0][lidx])
            localmsg = tf.concat(
                [tf.expand_dims(msg[lidx], axis=0) for msg in Msgs], axis=0)
            localbeliefs = tf.concat(
                [tf.expand_dims(bn[lidx], axis=0) for bn in NodeBeliefs],
                axis=0)
            # print(lidx, localmsg)
            for expand_idx in range(MsgDims):
                if (expand_idx) == lidx:
                    continue
                # print(expand_idx, localmsg)
                localmsg = tf.expand_dims(localmsg, expand_idx + 2)
                localbeliefs = tf.expand_dims(localbeliefs, expand_idx + 2)
            # print(lidx, localmsg)
            HigherBeliefs = HigherBeliefs - localmsg + localbeliefs
        '''
        Find argmax
        '''
        argmax_vars = []
        for lidx in range(MsgDims):
            to_be_marginalized = []
            for expand_idx in range(MsgDims):
                if (expand_idx) == lidx:
                    continue
                to_be_marginalized.append(expand_idx + 2)
            argmax_vars.append(
                tf.argmax(
                    tf.reduce_max(HigherBeliefs, axis=to_be_marginalized),
                    axis=2))
        assignments = [[argmax_vars[lidx][idx] for lidx in range(MsgDims)]
                       for idx in range(len(Msgs))]
        return assignments, HigherBeliefs

    @staticmethod
    def max_marginal_assignment_value(Higher_Order_Potentials,
                                      Msgs,
                                      NodeBeliefs,
                                      verbose=False):
        MsgDims = len(Msgs[0])
        HigherBeliefs = tf.concat([
            tf.expand_dims(hop.potential, axis=0)
            for hop in Higher_Order_Potentials
        ],
                                  axis=0)
        HigherBeliefs = tf.expand_dims(HigherBeliefs, axis=1)
        # print("HB",HigherBeliefs)
        '''
        Create \\theta^{\\delta}_f
        '''
        for lidx in range(MsgDims):
            # print(Msgs[0][lidx])
            localmsg = tf.concat(
                [tf.expand_dims(msg[lidx], axis=0) for msg in Msgs], axis=0)
            localbeliefs = tf.concat(
                [tf.expand_dims(bn[lidx], axis=0) for bn in NodeBeliefs],
                axis=0)
            # print(lidx, localmsg)
            for expand_idx in range(MsgDims):
                if (expand_idx) == lidx:
                    continue
                # print(expand_idx, localmsg)
                localmsg = tf.expand_dims(localmsg, expand_idx + 2)
                localbeliefs = tf.expand_dims(localbeliefs, expand_idx + 2)
            # print(lidx, localmsg)
            HigherBeliefs = HigherBeliefs - localmsg + localbeliefs
        '''
        Find max sum
        '''
        to_be_marginalized = []
        for lidx in range(MsgDims):
            # for expand_idx in range(MsgDims):
            to_be_marginalized.append(lidx + 2)

        max_over_all = tf.reduce_max(HigherBeliefs, axis=to_be_marginalized)
        sum_all_timeSteps = tf.reduce_sum(max_over_all)
        return sum_all_timeSteps


# @gin.configurable
class ConvFactor1D(Factors):
    cnt = 0

    def __init__(self,
                 nchannels,
                 ksize,
                 nlabels,
                 kernel=None,
                 name=None,
                 circular_padding=False,
                 trainable=False,
                 validate_kernel_shape=True,
                 with_rl_parmas=False):

        self.nchannels = nchannels
        self.ksize = ksize
        assert (ksize >= 1 and isinstance(ksize, int))
        self.nlabels = nlabels
        self.circular_padding = circular_padding

        self.paddings = [[0, 0], [int(ksize / 2), int(ksize / 2)], [0, 0]]

        if kernel is not None and validate_kernel_shape:
            assert (int(kernel.shape[0]) == ksize)
            assert (int(kernel.shape[1]) == 1)
            assert (int(kernel.shape[2]) == nchannels)
        if name is not None:
            self.name = name
        else:
            self.name = 'ConvFactor1D_{}'.format(ConvFactor1D.cnt)
            ConvFactor1D.cnt += 1
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE) as vs:
            if isinstance(kernel, tf.Variable):
                self.k = kernel
            else:
                if kernel is None:
                    kernel = np.random.randn(self.ksize, 1, self.nchannels)
                    kernel = tf.constant(kernel.astype(np.float32))
                with tf.variable_scope('sl_params'):
                    self.k = tf.get_variable(
                        'conv_kernel', initializer=kernel, trainable=trainable)
                    self.sl_params = self.k
                if with_rl_parmas:
                    with tf.variable_scope('rl_params'):
                        self.rl_params = tf.get_variable(
                            'conv_kernel_rl',
                            shape=self.sl_params.shape,
                            dtype=tf.float32)
                        self.k = self.k + self.rl_params
                # kernel must be a distribution
                self.k = tf.exp(self.k - tf.reduce_logsumexp(
                    self.k, axis=0, keepdims=True))
                self.kt = tf.reverse(self.k, [0])

    def padding_inputs(self, pos, pad_val=-1e20):
        if self.circular_padding is False:
            return tf.pad(pos, self.paddings, constant_values=pad_val)
        else:
            hksize = int(self.ksize / 2)
            left_pad = pos[:, -hksize:, :]
            right_pad = pos[:, :hksize, :]
            padded_val = tf.concat([left_pad, pos, right_pad], axis=1)
            return padded_val

    def getTransitionLoss(self, gather_indices, labels, all_labels, name):
        curr_pos = tf.one_hot(all_labels[0], self.nlabels)
        act = tf.one_hot(all_labels[1], self.nchannels)

        curr_pos = tf.expand_dims(curr_pos, 2)
        # with tf.control_dependencies([tf.print(curr_pos[:, :, 0])]):
        curr_pos_padded = self.padding_inputs(curr_pos, 0)

        mov_direcs = act
        mov_direcs_expanded = tf.expand_dims(mov_direcs, axis=1)
        c2n = tf.nn.conv1d(curr_pos_padded, self.k, 1,
                           'VALID') * mov_direcs_expanded

        pred_next = tf.reduce_sum(c2n, axis=2)
        logz = tf.log(tf.reduce_sum(pred_next, axis=1, keepdims=True))
        pred_next = tf.log(tf.clip_by_value(pred_next, 1e-10, 1))
        loss = -tf.batch_gather(
            pred_next, tf.expand_dims(tf.cast(all_labels[-1], tf.int32),
                                      1)) + logz
        # rlabels = tf.stop_gradient(tf.one_hot(all_labels[-1], self.nlabels))
        # loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        #     labels=rlabels, logits=pred_next)
        # print(tf.gradients(loss, self.sl_params))
        loss = tf.squeeze(loss, axis=1)

        return loss

    @staticmethod
    def LoopyBP(Hops, NodeBeliefs, Msgs, redis_factor=None, damping=0.25):
        """
        Hops must be a list of duplicated conv factors.
        node order: curr_pos, mov_direc, next_pos
        """
        assert (len(NodeBeliefs[0]) == 3)
        MsgDims = len(NodeBeliefs[0])
        LocalBeliefs = [
            tf.concat([bn[lidx] for bn in NodeBeliefs], axis=0)
            for lidx in range(MsgDims)
        ]
        AllMsgs = [
            tf.concat([msg[lidx] for msg in Msgs], axis=0)
            for lidx in range(MsgDims)
        ]
        # print(LocalBeliefs)
        sp_shapes = [int(lbelief.shape[1]) for lbelief in LocalBeliefs]
        mlb = tf.concat(LocalBeliefs, axis=1)
        mmsg = tf.concat(AllMsgs, axis=1)
        # print(mlb, mmsg)
        mhat_msgs = mlb - mmsg
        hat_msgs = tf.split(mhat_msgs, sp_shapes, axis=1)
        # print(hat_msgs)
        k = Hops[0].k
        kt = Hops[0].kt

        curr_pos = tf.expand_dims(hat_msgs[0], 2)

        curr_pos_padded = Hops[0].padding_inputs(curr_pos)
        next_pos = tf.expand_dims(hat_msgs[2], 2)
        next_pos_padded = Hops[0].padding_inputs(next_pos)

        mov_direcs = hat_msgs[1]
        mov_direcs_expanded = tf.expand_dims(mov_direcs, axis=1)
        c2n = tf.nn.conv1d(curr_pos_padded, k, 1,
                           'VALID') + mov_direcs_expanded
        n2c = tf.nn.conv1d(next_pos_padded, kt, 1,
                           'VALID') + mov_direcs_expanded

        inv_scale_factor = 1.0 / 3
        ncurr_pos = inv_scale_factor * (
            tf.reduce_max(n2c, axis=2) + hat_msgs[0])
        nnext_pos = inv_scale_factor * (
            tf.reduce_max(c2n, axis=2) + hat_msgs[2])
        nmov_direcs = inv_scale_factor * tf.reduce_max(
            tf.concat([curr_pos + n2c, next_pos + c2n], axis=1), axis=1)

        nLocalBeliefs = [ncurr_pos, nmov_direcs, nnext_pos]
        nmlb = tf.concat(nLocalBeliefs, axis=1)

        nmsg_merged = nmlb - mhat_msgs

        nmsg_merged = tf.split(nmsg_merged, sp_shapes, axis=1)

        nBelief = [
            tf.split(nLocalBeliefs[lidx], len(NodeBeliefs), axis=0)
            for lidx in range(MsgDims)
        ]
        nMsg = [
            tf.split(nmsg_merged[lidx], len(NodeBeliefs), axis=0)
            for lidx in range(MsgDims)
        ]

        return [[nBelief[lidx][idx] for lidx in range(MsgDims)]
                for idx in range(len(NodeBeliefs))], \
            [[nMsg[lidx][idx]
              for lidx in range(MsgDims)] for idx in range(len(NodeBeliefs))]


class ConvFactor(Factors):
    def __init__(self,
                 nchannels,
                 ksize,
                 var_shape,
                 kernel=None,
                 name=None,
                 trainable=False):
        self.nchannels = nchannels
        self.ksize = ksize
        self.padding_size = [[0, 0],
                             [int((ksize - 1) / 2),
                              int((ksize - 1) / 2)],
                             [int((ksize - 1) / 2),
                              int((ksize - 1) / 2)], [0, 0]]
        self.var_shape = var_shape
        if (kernel is not None):
            with tf.variable_scope(name) as vs:
                self.kernel = tf.get_variable(
                    'kernel',
                    dtype=tf.float32,
                    initializer=kernel,
                    trainable=trainable)
                self.kernelT = tf.reverse(kernel, axis=[0, 1])
        else:
            with tf.variable_scope(name):
                self.kernel = tf.get_variable(
                    'kernel',
                    shape=[ksize, ksize, 1, nchannels],
                    dtype=tf.float32,
                    trainable=trainable)
                self.kernelT = tf.reverse(kernel, axis=[0, 1])

    @staticmethod
    def LoopyBP(Hops, NodeBeliefs, Msgs, redis_factor=None, damping=0.25):
        MsgDims = len(NodeBeliefs[0])
        LocalBeliefs = [
            tf.concat([tf.expand_dims(bn[lidx], axis=2) for bn in NodeBeliefs],
                      axis=2) for lidx in range(MsgDims)
        ]
        AllMsgs = [
            tf.concat([tf.expand_dims(msg[lidx], axis=2) for msg in Msgs],
                      axis=2) for lidx in range(MsgDims)
        ]

        sp_shapes = [int(lbelief.shape[1]) for lbelief in LocalBeliefs]
        mlb = tf.concat(LocalBeliefs, axis=1)
        mmsg = tf.concat(AllMsgs, axis=1)
        mhat_msgs = mlb - mmsg

        hat_msgs = tf.split(mhat_msgs, sp_shapes, axis=1)

        convk = tf.concat([hop.kernel for hop in Hops], axis=2)
        convkt = tf.concat([hop.kernelT for hop in Hops], axis=2)

        cfeature = tf.reshape(hat_msgs[0],
                              [-1] + Hops[0].var_shape + [len(Hops)])
        nfeature = tf.reshape(hat_msgs[2],
                              [-1] + Hops[0].var_shape + [len(Hops)])

        cfeature_pad = tf.pad(
            cfeature, Hops[0].padding_size, constant_values=-1e6)
        nfeature_pad = tf.pad(
            nfeature, Hops[0].padding_size, constant_values=-1e6)

        # print(cfeature)
        mbelief = tf.reshape(
            tf.transpose(hat_msgs[1], [0, 2, 1]),
            [-1, 1, 1, Hops[0].nchannels * len(Hops)])

        c2n = tf.nn.depthwise_conv2d(cfeature_pad, convk,
                                     (1, 1, 1, 1), 'VALID') + mbelief
        n2c = tf.nn.depthwise_conv2d(nfeature_pad, convkt,
                                     (1, 1, 1, 1), 'VALID') + mbelief

        c2n = tf.reshape(c2n, [-1] + Hops[0].var_shape + [len(Hops)] +
                         [Hops[0].nchannels]) + tf.expand_dims(nfeature, 4)
        n2c = tf.reshape(n2c, [-1] + Hops[0].var_shape + [len(Hops)] +
                         [Hops[0].nchannels]) + tf.expand_dims(cfeature, 4)

        cbelief = tf.reshape(
            tf.reduce_max(n2c, axis=[4]),
            [-1] + [np.prod(Hops[0].var_shape)] + [len(Hops)]) * 1.0 / 3
        nbelief = tf.reshape(
            tf.reduce_max(c2n, axis=[4]),
            [-1] + [np.prod(Hops[0].var_shape)] + [len(Hops)]) * 1.0 / 3

        abelief = tf.transpose(tf.reduce_max(c2n, axis=[1, 2]),
                               [0, 2, 1]) * 1.0 / 3

        nLocalBeliefs = [cbelief, abelief, nbelief]

        nmlb = tf.concat(nLocalBeliefs, axis=1)

        nmsgs_merge = nmlb - mhat_msgs

        nMsgs = tf.split(nmsgs_merge, sp_shapes, axis=1)
        # print('Conv Factor nLocalBeliefs', nLocalBeliefs)
        nlb = [tf.split(lb, len(Hops), 2) for lb in nLocalBeliefs]
        # print('Conv Factor nlb', nlb)

        nmsg = [tf.split(lb, len(Hops), 2) for lb in nMsgs]

        nbeliefs = [[nlb[lidx][t][:, :, 0] for lidx in range(MsgDims)]
                    for t in range(len(nlb[0]))]
        # print('Conv Factor nbeliefs', nbeliefs)

        nmsgs = [[nmsg[lidx][t][:, :, 0] for lidx in range(MsgDims)]
                 for t in range(len(nlb[0]))]
        # print('Conv Factor nmsgs', nmsgs)

        return nbeliefs, nmsgs


class ShiftFactor2D(Factors):
    def __init__(self, var_shape, offsets, name):
        self.var_shape = var_shape
        if not isinstance(offsets, tf.Variable):
            with tf.variable_scope(name):
                self.offsets = tf.get_variable(
                    'offsets', initializer=offsets, dtype=tf.float32)
        else:
            self.offsets = offsets

    @staticmethod
    def expand_hops(hops, batch_size):
        all_offsets = tf.concat([
            tf.broadcast_to(
                hop.offsets,
                [batch_size, int(hop.offsets.shape[1]), 2]) for hop in hops
        ],
                                axis=0)
        return all_offsets

    @staticmethod
    def partial_primal(hops, partial_assignments, partial_idx, beliefs, msgs):
        batch_size = tf.shape(beliefs[0][0])[0]
        var_shape = hops[0].var_shape
        nchanels = int(hops[0].offsets.shape[1])
        msgs_ = Factors.ConcatLocalVariables(msgs)
        beliefs_ = Factors.ConcatLocalVariables(beliefs)

        for idx in partial_idx:
            beliefs_[idx] = tf.fill(
                [batch_size] + list(beliefs_[idx].shape[1:]), 0.0)

        all_offsets = ShiftFactor2D.expand_hops(hops, batch_size)
        cfeature = tf.reshape(beliefs_[0] - msgs_[0], [batch_size] + var_shape)
        nfeature = tf.reshape(beliefs_[2] - msgs_[2], [batch_size] + var_shape)
        avar = tf.reshape(beliefs_[1] - msgs_[1], [batch_size, nchanels])

        c2n, n2c = MRFInferGPU.msg_gather_shift2d(all_offsets,
                                                  [cfeature, avar, nfeature])

        if partial_idx == [0]:
            n2c = tf.reshape(n2c, [batch_size, np.prod(var_shape), nchanels])
            res = tf.batch_gather(n2c, partial_assignments[0])
            res = tf.reduce_max(res, axis=2)
        if partial_idx == [1]:
            n2c = tf.reduce_max(n2c, axis=[1, 2])
            c2n = tf.reduce_max(n2c, axis=[1, 2])

            alltoa = [tf.expand_dims(n2c, 2), tf.expand_dims(c2n, 2)]
            alltoa = tf.concat(alltoa, axis=2)
            alltoa = tf.reduce_max(alltoa, axis=2)
            res = tf.batch_gather(alltoa, partial_assignments[0])
        if partial_idx == [2]:
            c2n = tf.reshape(c2n, [batch_size, np.prod(var_shape), nchanels])
            res = tf.batch_gather(c2n, partial_assignments[0])
            res = tf.reduce_max(res, 2)
        if (partial_idx == [0, 1]):
            final_assign = partial_assignments[
                0] * nchanels + partial_assignments[1]
            n2c = tf.reshape(n2c, [batch_size, np.prod(var_shape) * nchanels])
            res = tf.batch_gather(n2c, final_assign)
        if partial_idx == [1, 2]:
            final_assign = partial_assignments[
                1] * nchanels + partial_assignments[0]
            c2n = tf.reshape(c2n, [batch_size, np.prod(var_shape) * nchanels])
            res = tf.batch_gather(c2n, final_assign)
        if (partial_idx == [0, 1, 2]):
            hvalue = ShiftFactor2D.primal(hops, [partial_assignments])
            lvalue = [
                tf.batch_gather(msg, assign)
                for msg, assign in zip(msgs_, partial_assignments)
            ]
            lvalue = tf.add_n(lvalue)
            res = hvalue - lvalue
        print(partial_idx)
        return res

    @staticmethod
    def primal(hops, assignments):
        batch_size = tf.shape(assignments[0][0])[0]
        shape_x, shape_y = hops[0].var_shape
        nassignments = [
            tf.concat([assign[lidx] for assign in assignments], axis=0)
            for lidx in range(len(assignments[0]))
        ]
        all_offsets = ShiftFactor2D.expand_hops(hops, batch_size)

        prev_pos = nassignments[0]
        current_offsets = tf.batch_gather(all_offsets, nassignments[1])
        current_offsets = tf.cast(current_offsets, tf.int32)
        print(current_offsets)
        print(nassignments)
        next_pos = nassignments[2]

        prev_x = tf.div(prev_pos, shape_y)
        prev_y = prev_pos - prev_x * shape_y

        next_x = tf.div(next_pos, shape_y)
        next_y = next_pos - next_x * shape_y

        shift_x = current_offsets[:, :, 0]
        shift_y = current_offsets[:, :, 1]

        nx = prev_x + shift_x
        ny = prev_y + shift_y

        is_valid_x = tf.logical_and(nx >= 0, nx < shape_x)
        is_valid_y = tf.logical_and(ny >= 0, ny < shape_y)

        is_valid = tf.logical_and(is_valid_x, is_valid_y)

        nx = tf.where(is_valid, nx, prev_x)
        ny = tf.where(is_valid, ny, prev_y)

        x_equal = tf.equal(nx, next_x)
        y_equal = tf.equal(ny, next_y)

        xy_equal = tf.logical_and(x_equal, y_equal)
        xy_equal = tf.cast(xy_equal, tf.float32)

        return to_log_scale(xy_equal)

    @staticmethod
    def LoopyBP(Hops, NodeBeliefs, Msgs, redis_factor=None, damping=0.25):
        MsgDims = len(NodeBeliefs[0])
        batch_size = tf.shape(NodeBeliefs[0][0])[0]
        LocalBeliefs = [
            tf.concat([bn[lidx] for bn in NodeBeliefs], axis=0)
            for lidx in range(MsgDims)
        ]
        AllMsgs = [
            tf.concat([msg[lidx] for msg in Msgs], axis=0)
            for lidx in range(MsgDims)
        ]
        all_offsets = ShiftFactor2D.expand_hops(Hops, batch_size)

        sp_shapes = [int(lbelief.shape[1]) for lbelief in LocalBeliefs]
        mlb = tf.concat(LocalBeliefs, axis=1)
        mmsg = tf.concat(AllMsgs, axis=1)
        mhat_msgs = mlb - mmsg

        hat_msgs = tf.split(mhat_msgs, sp_shapes, axis=1)

        cfeature = tf.reshape(hat_msgs[0], [-1] + Hops[0].var_shape)
        nfeature = tf.reshape(hat_msgs[2], [-1] + Hops[0].var_shape)

        avar = hat_msgs[1]

        c2n, n2c = MRFInferGPU.msg_gather_shift2d(all_offsets,
                                                  [cfeature, avar, nfeature])

        cbelief = tf.reshape(
            tf.reduce_max(n2c, axis=3),
            [-1] + [np.prod(Hops[0].var_shape)]) * 1.0 / 3
        nbelief = tf.reshape(
            tf.reduce_max(c2n, axis=3),
            [-1] + [np.prod(Hops[0].var_shape)]) * 1.0 / 3

        abelief = tf.reduce_max(n2c, axis=[1, 2]) * 1.0 / 3

        nLocalBeliefs = [cbelief, abelief, nbelief]
        nmbeliefs = tf.concat(nLocalBeliefs, axis=1)

        nmbeliefs, nmsg_merged = MRFInferGPU.upd_msg_belief(
            mlb, nmbeliefs, mmsg)

        lop_shapes = [
            np.prod(Hops[0].var_shape),
            int(hat_msgs[1].shape[1]),
            np.prod(Hops[0].var_shape)
        ]

        nLocalBeliefs = tf.split(nmbeliefs, lop_shapes, axis=1)
        nmsg_merged = tf.split(nmsg_merged, lop_shapes, axis=1)

        nBelief = [
            tf.split(nLocalBeliefs[lidx], len(NodeBeliefs), axis=0)
            for lidx in range(MsgDims)
        ]
        nMsg = [
            tf.split(nmsg_merged[lidx], len(NodeBeliefs), axis=0)
            for lidx in range(MsgDims)
        ]

        return [[nBelief[lidx][idx] for lidx in range(MsgDims)]
                for idx in range(len(NodeBeliefs))], \
            [[nMsg[lidx][idx]
              for lidx in range(MsgDims)] for idx in range(len(NodeBeliefs))]


# from .TreeFactor import TreeFactor
# from .cross_state_factor import SelectFactor, MineralShardsTransition, MoveFactorCommandsVid, MoveFactorCommandsSteps, MoveFactorPosition, MoveFactorPositionJoint
# from .in_state_factor import SoilderMineralActive, MineralActiveTransition, SoilderMineralActiveXY, SoilderMineralActiveJoint, SoilderMineralActiveJointConv, SoilderDisPos, IsTerminateFactor, SoilderDisPosTerminal, TerminalMineralFactor

if __name__ == '__main__':
    map_shape = [4, 5]
    kernel = np.zeros([3, 3, 1, 5])
    kernel[1, 1, 0, 0] = 1
    kernel[2, 1, 0, 1] = 1
    kernel[0, 1, 0, 2] = 1
    kernel[1, 2, 0, 3] = 1
    kernel[1, 0, 0, 4] = 1

    offsets = np.asarray([[[0, 0], [1, 0], [-1, 0], [0, 1],
                           [0, -1]]]).astype(np.float32)

    cposition = tf.placeholder(tf.float32, [None, np.prod(map_shape)])
    caction = tf.placeholder(tf.float32, [None, 5])
    nposition = tf.zeros_like(cposition)

    rcposition = np.zeros([1, 4, 5], dtype=np.float32)
    rcposition[0, 2, 3] = 1

    raction = np.zeros([1, 5], dtype=np.float32)
    raction[0, 4] = 1
    feed_dict = {
        cposition: np.reshape(np.log(np.clip(rcposition, 1e-6, 1.0)), [1, -1]),
        caction: np.log(np.clip(raction, 1e-6, 1.0))[:]
    }

    assign = [[
        tf.constant([[13, 13]], dtype=tf.int32),
        tf.constant([[4, 3]], dtype=tf.int32),
        tf.constant([[12, 14]], dtype=tf.int32)
    ]]

    partial_assign = [[
        tf.constant([[4, 3]], dtype=tf.int32),
        tf.constant([[12, 14]], dtype=tf.int32)
    ]]
    partial_idx = [1, 2]
    cfactor = ConvFactor(5, 3, map_shape, kernel.astype(np.float32),
                         'move_factor')
    beliefs = [[cposition, caction, nposition],
               [cposition, caction, nposition]]
    msgs = [[tf.zeros_like(b) for b in bt] for bt in beliefs]

    sfactor = ShiftFactor2D(map_shape, offsets, 'move_factor')
    primal_value = sfactor.primal([sfactor], assign)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    nbeliefs, nmsgs = ShiftFactor2D.LoopyBP([sfactor, sfactor], beliefs, msgs)
    print(nmsgs)
    print(nbeliefs)
    [rnmsgs, rnbeliefs] = sess.run(
        [nmsgs, nbeliefs],
        feed_dict={
            cposition:
            np.reshape(np.log(np.clip(rcposition, 1e-6, 1.0)), [1, -1]),
            caction:
            np.log(np.clip(raction, 1e-30, 1.0))[:]
        })
    for i in range(2):
        original_cpos = np.log(np.clip(rcposition, 1e-6, 1.0))[0]

        print(original_cpos)

        infered_cpos = np.reshape(rnbeliefs[i][0], [4, 5])
        infered_npos = np.reshape(rnbeliefs[i][2], [4, 5])
        infered_act = rnbeliefs[i][1]

        print('infered_cpos ', infered_cpos)
        print('infered_npos ', infered_npos)
        print('infered_act ', infered_act)
    primal_value_res = sess.run(primal_value)
    print(primal_value_res)
