import numpy as np
import tensorflow as tf

from . import transition, utils

try:
    from tensorflow.python.ops.rnn_cell_impl import RNNCell
except ImportError:
    from tensorflow.contrib.rnn import RNNCell

# RNN cell
class BasicInferUnit(RNNCell):
    """
    The basic inference unit in the network.
    state: b_{s_{tp1}}, m(b_{s_{tp1}}->b_{s_{tp1}})
    input: b_{s_t}, b_{a_t}, m(b_{s_t}->b_{s_t}), m(t->b_{s_t}), m(t->b_{a_t}), m(t->b_{s_tp1})
    output_state: b_{s_t}
    output: b_{s_{tp1}}, b_{a_t}, m(b_{s_t}->b_{s_t}), m(t->b_{s_t}), m(t->b_{a_t}), m(t->b_{s_tp1})
    """
    def __init__(self, 
                sizesnodes, sizeanodes,
                in_state_factor, cross_state_factor):
        
        self.numsnode = len(sizesnodes)
        self.sizesnodes = sizesnodes
        self.numanode = len(sizeanodes)
        self.sizeanodes = sizeanodes
    
        self.in_state_factor = in_state_factor
        msg_in_state_siz = []
        for factor in in_state_factor:
            cmsg_siz = []
            included_nodes = factor['nodes']
            for node in included_nodes:
                cmsg_siz.append(self.sizesnodes[node])
            msg_in_state_siz += cmsg_siz

        self.in_state_msg_siz = msg_in_state_siz

        self.cross_state_msg_t_siz = []
        self.cross_state_msg_tprime_siz = []
        self.cross_state_msg_act_siz = []

        self.cross_state_factor = cross_state_factor
        for factor in cross_state_factor:
            st_node = factor['cnodes']
            at_node = factor['action']
            stprime_node = factor['nextnodes']

            st_node_states = [self.sizesnodes[n] for n in st_node]
            at_node_states = [self.sizeanodes[n] for n in at_node]
            stprime_node_states = [self.sizesnodes[n] for n in stprime_node]

            self.cross_state_msg_t_siz += st_node_states
            self.cross_state_msg_act_siz += at_node_states
            self.cross_state_msg_tprime_siz += stprime_node_states
        #In this file a->b means message from a to b
        #input: bs_t, bs_t -> bs_t, a_prevt, ft->s_prevt, ft->a_prevt, ft->bs_t
        self.input_parts = [self.sizesnodes, self.in_state_msg_siz, self.sizeanodes, self.cross_state_msg_t_siz,
                            self.cross_state_msg_act_siz, self.cross_state_msg_tprime_siz]
        #state_input: bs_tp, bs_tp->bs_tp
        self.state_parts = [self.sizesnodes, self.in_state_msg_siz]

    @property
    def all_input_dims(self):
        """
        input: b_{s_t}, b_{a_t}, m(b_{s_t}->b_{s_t}), m(t->b_{s_t}), m(t->b_{a_t}), m(t->b_{s_tp1})
        """
        res = []
        for sls in self.input_parts:
            res += sls
        return res

    @property
    def input_size(self):
        """
        input: b_{s_t}, b_{a_t}, m(b_{s_t}->b_{s_t}), m(t->b_{s_t}), m(t->b_{a_t}), m(t->b_{s_tp1})
        """
        return np.sum(self.all_input_dims)

    def split_input(self, input_var):
        return utils.split_as_slices(input_var, self.input_parts)
    
    @property
    def all_state_dims(self):
        res = []
        for sls in self.state_parts:
            res += sls
        return res
    @property
    def state_size(self):
        total_dim = np.sum(self.sizesnodes)\
            + np.sum(self.in_state_msg_siz)
        return total_dim

    def split_state(self, state_var):
        return utils.split_as_slices(state_var, self.state_parts)

    @property
    def output_size(self):
        """
        output: b_tprev, a_tprev, b_tprev->b_tprev, f_t -> b_tprev, f_t -> a_tprev, f_t -> b_t 
        """
        return self.input_size


    def __call__(self, inputs, state, scope=None):
        """
        See link:
        The basic inference unit in the network.
        state: b_{s_{tp1}}, m(b_{s_{tp1}}->b_{s_{tp1}})
        input: b_{s_t}, b_{a_t}, m(b_{s_t}->b_{s_t}), m(t->b_{s_t}), m(t->b_{a_t}), m(t->b_{s_tp1})
        output_state: b_{s_t}, m(b_{s_{t}}->b_st)
        output: b_{s_{tp1}}, b_{a_t}, m(b_{s_t}->b_{s_t}), m(t->b_{s_t}), m(t->b_{a_t}), m(t->b_{s_tp1})
        """

        bs_t, imsgs_t, ba_t, cmsgs_t, cmsga_t, cmsgs_tp = self.split_input(inputs)
        bs_tp, imsgs_tp = self.split_state(state)

        # In state msg passing on tp
        is_msg_start_idx = 0
        for factor in self.in_state_factor:
            belief = [[bs_tp[n] for n in factor['nodes']]]
            msgs = [imsgs_tp[is_msg_start_idx:(is_msg_start_idx + len(factor['nodes']))]]
            nbeliefs, nmsg = type(factor['Factor'][0]).LoopyBP([factor['Factor'][0]], belief, msgs, None)

            for idx, nidx in enumerate(factor['nodes']):
                bs_tp[nidx] = nbeliefs[0][idx]
            imsgs_tp[is_msg_start_idx:(is_msg_start_idx + len(factor['nodes']))] = nmsg[0]
            
            is_msg_start_idx += len(factor['nodes'])
            
        cs_msg_start_t_idx = 0
        cs_msg_start_prev_t_idx = 0
        cs_msg_action_t_idx = 0
        
        
        for factor in self.cross_state_factor:
            t_s_node = factor['cnodes']
            a_node = factor['action']
            tp_s_node = factor['nextnodes']

            belief_t_s = [bs_t[nidx] for nidx in t_s_node]
            belief_action = [ba_t[nidx] for nidx in a_node]
            belief_tp_s = [bs_tp[nidx] for nidx in tp_s_node]

            msg_all = [
                cmsgs_t[cs_msg_start_prev_t_idx:(cs_msg_start_prev_t_idx + len(t_s_node))] \
                + cmsga_t[cs_msg_action_t_idx:(cs_msg_action_t_idx + len(a_node))] \
                + cmsgs_tp[cs_msg_start_t_idx:(cs_msg_start_t_idx + len(tp_s_node))]
            ]

            beliefs = [belief_t_s + belief_action + belief_tp_s]

            nbeliefs, nmsg = type(factor['Factor'][0]).LoopyBP([factor['Factor'][0]], beliefs, msg_all, None)

            for idx, nidx in enumerate(t_s_node):
                bs_t[nidx] = nbeliefs[0][idx]
            cmsgs_t[cs_msg_start_prev_t_idx:(cs_msg_start_prev_t_idx + len(t_s_node))] = \
                nmsg[0][0:len(t_s_node)]
                
            for ii, nidx in enumerate(a_node):
                idx = ii + len(t_s_node)
                ba_t[nidx] = nbeliefs[0][idx]
            cmsga_t[cs_msg_action_t_idx:(cs_msg_action_t_idx + len(a_node))] = \
                nmsg[0][len(t_s_node):(len(t_s_node) + len(a_node))]

            for ii, nidx in enumerate(tp_s_node):
                idx = ii + len(t_s_node) + len(a_node)
                bs_tp[nidx] = nbeliefs[0][idx]
            cmsgs_tp[cs_msg_start_t_idx:(cs_msg_start_t_idx + len(tp_s_node))] = \
                nmsg[0][(len(t_s_node)+len(a_node)):]

            cs_msg_start_prev_t_idx += len(t_s_node)
            cs_msg_action_t_idx += len(a_node)
            cs_msg_start_t_idx += len(tp_s_node)

        #enf of cross_state msg passing
        
        is_msg_start_idx = 0
        for factor in self.in_state_factor:
            belief = [[bs_t[n] for n in factor['nodes']]]
            msgs = [imsgs_t[is_msg_start_idx:(is_msg_start_idx + len(factor['nodes']))]]
            nbeliefs, nmsg = type(factor['Factor'][0]).LoopyBP([factor['Factor'][0]], belief, msgs, None)

            for idx, nidx in enumerate(factor['nodes']):
                bs_t[nidx] = nbeliefs[0][idx]
            imsgs_t[is_msg_start_idx:(is_msg_start_idx + len(factor['nodes']))] = nmsg[0]
            
            is_msg_start_idx += len(factor['nodes'])
        

        #Output Organization: bs_prevt, bs_prevt->bs_prevt, a_prevt, ft->bs_prevt, ft->a_prevt, ft->bs_t
        #Organization of output states: bs_t, bs_t -> bs_t

        output = tf.concat(bs_tp + imsgs_tp + ba_t + cmsgs_t + cmsga_t + cmsgs_tp, axis=1)
        output_state = tf.concat(bs_t + imsgs_t, axis=1)

        return output, output_state


    
class InferNetPipeLine:
    def __init__(self, 
                sizesnodes,
                sizeanodes,
                simulate_steps,
                in_state_factor,
                cross_state_factor,
                BPIter=5,
                value_factor = None):
        self.sub_cell = BasicInferUnit(sizesnodes, sizeanodes, in_state_factor, cross_state_factor)
        self.sizesnodes = sizesnodes

        self.value_factor = value_factor
        self.create_init_state()
        
        self.all_input = tf.zeros([1, simulate_steps - 1, self.sub_cell.input_size])\
            + tf.expand_dims(self.input_zero_brd, axis=2)
        self.all_input = tf.concat([self.all_input, self.final_step_input], axis=1)

        self.objv = []
        init_state = self.zero_state
        all_input = self.all_input

        step = tf.constant(0, dtype=tf.int32)
        max_steps = tf.placeholder(tf.int32)
        simulate_steps = tf.placeholder(tf.int32)
        self.max_steps = max_steps
        self.simulate_steps = simulate_steps

        def update_value_msg_state(state, msg):
            state = tf.reshape(state, [-1, self.sub_cell.state_size])
            belief, others = tf.split(state, [np.sum(self.sizesnodes),
                                              self.sub_cell.state_size - np.sum(self.sizesnodes)], axis=1)
            
            belief = tf.split(belief, self.sizesnodes, axis=1)
            msg = tf.split(msg, self.sizesnodes, axis=1)

            nbelief, msg = self.value_factor.LoopyBP([self.value_factor], [belief], [msg])

            return tf.expand_dims(tf.concat([tf.concat(nbelief[0], axis=1), others], axis=1),1), tf.concat(msg[0], axis=1)

        def while_condition(step, infer_input, infer_state, infer_prev_output, infer_prev_state, value_msg=None):
            return tf.less(step, max_steps)



        def infer_loop_body(step, infer_input, infer_state, infer_prev_output, infer_prev_state, value_msg=None):
            """
            step: current step
            max_steps: max_steps = simulate_steps + bp_steps
            simulate_steps: simulate_steps
            """
            infer_input_ = tf.reshape(infer_input, [-1, self.sub_cell.input_size])
            infer_input_state_ = tf.reshape(infer_state, [-1, self.sub_cell.state_size])

            infer_output, infer_output_state = self.sub_cell(infer_input_, infer_input_state_)

            infer_output = tf.reshape(infer_output, tf.shape(infer_input))
            infer_output_state = tf.reshape(infer_output_state, tf.shape(infer_state))
            
            infer_output_bs, infer_output_others = tf.split(infer_output, [self.sub_cell.state_size, self.sub_cell.input_size - self.sub_cell.state_size],
                                                            axis=2)
            prev_output_bs, prev_output_others = tf.split(infer_prev_output, [self.sub_cell.state_size, self.sub_cell.input_size - self.sub_cell.state_size],
                                                            axis=2)

            infer_state_prefix = tf.cond(tf.logical_or(tf.less(step, simulate_steps - 1),
                                                       tf.equal(tf.floormod(step - simulate_steps, 2), 0)),
                                         lambda: infer_output_state,
                                         lambda: infer_output_state[:, 1:, :])
            if(value_msg is not None):
                nstate, value_msg = tf.cond(tf.equal(tf.floormod(step, 2), 1),
                                            lambda: update_value_msg_state(prev_output_bs[:, -1:, :], value_msg),
                                            lambda: (prev_output_bs[:, -1:, :], value_msg))
            else:
                nstate = tf.cond(tf.equal(tf.floormod(step, 2), 1),
                                 lambda: prev_output_bs[:, -1:, :],
                                 lambda: prev_output_bs[:, 0:0, :])
                
            infer_state_ = tf.cond(tf.equal(tf.floormod(step, 2), 1),
                                    lambda: tf.concat([infer_state_prefix,
                                                       nstate],
                                                      axis=1),
                                    lambda: infer_state_prefix)

            

            infer_input_prefix = tf.cond(tf.less(step, simulate_steps - 1),
                                         lambda: all_input[:, (step+1):(step+2), :],
                                         lambda: tf.cond(tf.logical_and(tf.greater(step, simulate_steps - 1),
                                                                        tf.equal(tf.floormod(step - simulate_steps, 2), 0)),
                                                         lambda: tf.concat([infer_prev_state[:,0:1,:],
                                                                            prev_output_others[:, 0:1, :]], axis=2),
                                                         lambda: all_input[:, 0:0, :])
                                         )
            
                                         
            infer_input_suffix = tf.cond(tf.less(step, 1),
                                         lambda: all_input[:, 0:0, :],
                                         lambda: tf.cond(tf.logical_and(tf.greater(step, simulate_steps - 1),
                                                                       tf.equal(tf.floormod(step - simulate_steps, 2), 0)),
                                                         lambda: tf.concat([
                                                             tf.slice(infer_output_bs, [0, 0, 0],
                                                                      [tf.shape(infer_output)[0],
                                                                       tf.shape(infer_prev_output)[1] - 1,
                                                                       tf.shape(infer_output_bs)[2]]),
                                                             tf.slice(prev_output_others, [0, 1, 0],
                                                                      [tf.shape(infer_prev_output)[0],
                                                                       tf.shape(infer_prev_output)[1] - 1,
                                                                       tf.shape(prev_output_others)[2]])], axis=2),
                                                         lambda: tf.concat([
                                                             tf.slice(infer_output_bs, [0, 0, 0],
                                                                      [tf.shape(infer_output)[0],
                                                                       tf.shape(infer_prev_output)[1],
                                                                       tf.shape(infer_output_bs)[2]]),
                                                             tf.slice(prev_output_others, [0, 0, 0],
                                                                      [tf.shape(infer_prev_output)[0],
                                                                       tf.shape(infer_prev_output)[1],
                                                                       tf.shape(prev_output_others)[2]])], axis=2)
                                         ))

            infer_input_ = tf.concat([infer_input_prefix, infer_input_suffix], axis=1)
                                         

            if(value_msg is not None):
                return tf.add(step, 1), infer_input_, infer_state_, infer_output, infer_output_state, value_msg
            else:
                return tf.add(step, 1), infer_input_, infer_state_, infer_output, infer_output_state

        def infer_loop_body_full(step, infer_input, infer_state, infer_prev_output, infer_prev_state, value_msg):
            return infer_loop_body(step, infer_input, infer_state, infer_prev_output, infer_prev_state, value_msg)

        def while_condition_full(step, infer_input, infer_state, infer_prev_output, infer_prev_state, value_msg):
            return while_condition(step, infer_input, infer_state, infer_prev_output, infer_prev_state, value_msg)


        infer_input = all_input[:, 0:1, :]
        infer_input = tf.placeholder_with_default(infer_input, shape=[None, None, self.sub_cell.input_size])
        infer_state = tf.expand_dims(self.zero_state, 1)
        infer_state = tf.placeholder_with_default(infer_state, shape=[None, None, self.sub_cell.state_size])
        infer_prev_output = tf.placeholder_with_default(infer_input[:, 0:0, :], shape=[None, None, self.sub_cell.input_size])
        infer_prev_state = tf.placeholder_with_default(infer_state[:, 0:0, :], shape=[None, None, self.sub_cell.state_size])
        value_msg = self.value_msg
        s = 0

        
        #while(s < 16):
        #   step, max_steps, simulate_steps, infer_input, infer_state, infer_prev_output, infer_prev_state, all_input = \
        #       infer_loop_body(step, max_steps, simulate_steps, infer_input, infer_state, infer_prev_output, infer_prev_state, all_input)
        #   s+= 1
        if(self.value_factor is not None):
            _, _, _, final_output, final_state, _ = tf.while_loop(
                cond=while_condition_full,
                body=infer_loop_body_full,
                loop_vars=(step, infer_input, infer_state, infer_prev_output, infer_prev_state, value_msg)
            )
        else:
            _, _, _,final_output, final_state = tf.while_loop(
                cond=while_condition,
                body=infer_loop_body,
                loop_vars=(step, infer_input, infer_state, infer_prev_output, infer_prev_state)
            )

        final_state1, _, final_action, _, _, _ = self.sub_cell.split_input(final_output[:, 0, :])
        self.final_action = final_action
        self.final_state = final_state1
        self.final_output = final_output


    def create_init_state(self):
        self.init_belief = tf.placeholder(tf.float32, [None] + [np.sum(self.sub_cell.sizesnodes)])
        self.input_zero_brd =  tf.reduce_sum(tf.zeros_like(self.init_belief), axis=1, keep_dims=True)
        init_msg = tf.zeros([1, np.sum(self.sub_cell.in_state_msg_siz)]) + \
           self.input_zero_brd

        def update_value_msg(belief, msg):
            belief = tf.split(belief, self.sizesnodes, axis=1)
            msg = tf.split(msg, self.sizesnodes, axis=1)

            nbelief, msg = self.value_factor.LoopyBP([self.value_factor], [belief], [msg])

            return tf.concat(nbelief[0], axis=1), tf.concat(msg[0], axis=1)
        
        if(self.value_factor is not None):
            value_msg = tf.zeros_like(self.init_belief)
            self.value_factor = self.value_factor(self.init_belief)
            ninit_belief, self.value_msg = update_value_msg(self.init_belief, value_msg)
        else:
            self.value_msg = self.init_belief[0:0, :]
            ninit_belief = self.init_belief

            
        self.init_state = tf.concat([ninit_belief, init_msg], axis=1)
        self.zero_state = tf.zeros_like(self.init_state)

        input_padding = tf.zeros([1, self.sub_cell.input_size - self.sub_cell.state_size])
        input_padding = input_padding + self.input_zero_brd

        self.final_step_input = tf.expand_dims(tf.concat([self.init_state, input_padding], axis=1), 1)

        
class InferNetRNN:
    def __init__(self, 
                sizesnodes, sizeanodes, 
                simulate_steps,
                in_state_factor,
                cross_state_factor,
                BPIter=5):
        self.value_factor = None
        self.sub_cell = BasicInferUnit(sizesnodes, sizeanodes, in_state_factor, cross_state_factor)
        self.sizesnodes = sizesnodes
        self.create_init_state()
        
        self.all_input = tf.zeros([1, simulate_steps - 1, self.sub_cell.input_size]) \
                        + tf.expand_dims(self.input_zero_brd, axis=2)
        self.all_input = tf.concat([self.all_input, self.final_step_input], axis=1)

        self.objv = []
        init_state = self.zero_state
        all_input = self.all_input
        for i in range(BPIter):
            transition.Factors.ExpandHOPs = dict()
            outputs, state = tf.nn.dynamic_rnn(
                self.sub_cell, all_input,
                initial_state=init_state
            )

            bprevt_part, others = tf.split(outputs, [self.sub_cell.state_size, self.sub_cell.input_size - self.sub_cell.state_size], axis=2)
            
            beliefs, _ = tf.split(bprevt_part,[np.sum(sizesnodes), np.sum(self.sub_cell.in_state_msg_siz)], axis=2)
            tbeliefs, _ = tf.split(state, [np.sum(sizesnodes), np.sum(self.sub_cell.in_state_msg_siz)], axis=1)

            asize = np.sum(self.sub_cell.sizeanodes)
            beliefa, _ = tf.split(others, [asize, self.sub_cell.input_size - self.sub_cell.state_size - asize], axis=2)
            beliefa = tf.split(beliefa, sizeanodes, axis=2)
            beliefs = tf.split(beliefs, sizesnodes, axis=2)
            objv = 0
            for belief in beliefs:
                objv += tf.reduce_sum(tf.reduce_max(belief, axis=2), axis=1)
            for belief in beliefa:
                objv += tf.reduce_sum(tf.reduce_max(belief, axis=2), axis=1)

            tbeliefs = tf.split(tbeliefs, sizesnodes, axis=1)
            for belief in tbeliefs:
                objv += tf.reduce_max(belief, axis=1)

            self.objv.append(objv)
            
            
            init_state, bprevt_part = tf.split(bprevt_part, [1, simulate_steps - 1], axis=1)
            bprevt_part = tf.concat([bprevt_part, tf.expand_dims(state, 1)], axis=1)

            all_input = tf.concat([bprevt_part, others], axis=2)
            init_state = init_state[:,0,:]

        final_state1, _, final_action, _, _, _ = self.sub_cell.split_input(tf.transpose(outputs, perm=[0,2,1]))

        final_final_state, _, final_final_action, _, _, _ = self.sub_cell.split_input(outputs[:,-1,:])
        self.final_action = final_action
        self.final_state = final_state1
        self.final_output = outputs
        self.final_final_state = final_final_state
        self.final_final_action = final_final_action
            

    def create_init_state(self):
        self.init_belief = tf.placeholder(tf.float32, [None] + [np.sum(self.sub_cell.sizesnodes)])
        self.input_zero_brd =  tf.reduce_sum(tf.zeros_like(self.init_belief), axis=1, keep_dims=True)
        init_msg = tf.zeros([1, np.sum(self.sub_cell.in_state_msg_siz)]) + self.input_zero_brd

        def update_value_msg(belief, msg):
            belief = tf.split(belief, self.sizesnodes, axis=1)
            msg = tf.split(belief, self.sizesnodes, axis=1)

            nbelief, msg = self.value_factor.LoopyBP([self.value_factor], [belief], [msg])

            return tf.concat(nbelief[0], axis=1), tf.concat(msg[0], axis=1)
        
        if(self.value_factor is not None):
            value_msg = tf.zeros_like(self.init_belief)
            self.value_factor = self.value_factor
            ninit_belief, self.value_msg = update_value_msg(self.init_belief, value_msg)
        else:
            self.value_msg = self.init_belief[:, 0:0]
            ninit_belief = self.init_belief
        
            
        self.init_state = tf.concat([ninit_belief, init_msg], axis=1)
        self.zero_state = tf.zeros_like(self.init_state)

        input_padding = tf.zeros([1, self.sub_cell.input_size - self.sub_cell.state_size])
        input_padding = input_padding + self.input_zero_brd

        self.final_step_input = tf.expand_dims(tf.concat([self.init_state, input_padding], axis=1), 1)
