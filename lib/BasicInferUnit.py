import numpy as np
import tensorflow as tf
try:
    from tensorflow.python.ops.rnn_cell_impl import RNNCell
except ImportError:
    from tensorflow.contrib.rnn import RNNCell
import lib.utils as utils
import lib.transition as transition

from collections import defaultdict
import networkx as nx


MERGE_FACTORS = False


class BasicInferUnit(RNNCell):
    """
    The basic inference unit in the network.
    state: b_{s_{tp1}}, m(b_{s_{tp1}}->b_{s_{tp1}})
    input: b_{s_t}, b_{a_t}, m(b_{s_t}->b_{s_t}), m(t->b_{s_t}), m(t->b_{a_t}), m(t->b_{s_tp1})
    output_state: b_{s_t}
    output: b_{s_{tp1}}, b_{a_t}, m(b_{s_t}->b_{s_t}), m(t->b_{s_t}), m(t->b_{a_t}), m(t->b_{s_tp1})
    """

    def __init__(self, sizesnodes,
                 sizeanodes,
                 in_state_factor,
                 cross_state_factor,
                 merge_factors=False):

        self.numsnode = len(sizesnodes)
        self.sizesnodes = sizesnodes
        self.numanode = len(sizeanodes)
        self.sizeanodes = sizeanodes
        self.merge_factors = merge_factors

        self.in_state_factor = in_state_factor
        msg_in_state_siz = []
        for id, factor in enumerate(self.in_state_factor):
            cmsg_siz = []
            included_nodes = factor['nodes']
            for node in included_nodes:
                cmsg_siz.append(self.sizesnodes[node])
            msg_in_state_siz += cmsg_siz
            factor['id'] = id

        self.in_state_msg_siz = msg_in_state_siz

        self.cross_state_msg_t_siz = []
        self.cross_state_msg_tprime_siz = []
        self.cross_state_msg_act_siz = []

        self.cross_state_factor = cross_state_factor
        for id, factor in enumerate(self.cross_state_factor):
            st_node = factor['cnodes']
            at_node = factor['action']
            stprime_node = factor['nextnodes']

            st_node_states = [self.sizesnodes[n] for n in st_node]
            at_node_states = [self.sizeanodes[n] for n in at_node]
            stprime_node_states = [self.sizesnodes[n] for n in stprime_node]

            self.cross_state_msg_t_siz += st_node_states
            self.cross_state_msg_act_siz += at_node_states
            self.cross_state_msg_tprime_siz += stprime_node_states
            factor['id'] = id
        # In this file a->b means message from a to b
        # input: bs_t, bs_t -> bs_t, a_prevt, ft->s_prevt, ft->a_prevt, ft->bs_t
        self.input_parts = [self.sizesnodes, self.in_state_msg_siz, self.sizeanodes, self.cross_state_msg_t_siz,
                            self.cross_state_msg_act_siz, self.cross_state_msg_tprime_siz]
        # state_input: bs_tp, bs_tp->bs_tp
        self.state_parts = [self.sizesnodes, self.in_state_msg_siz]

        self.factor_int_mapping = self.create_factor_int_mapping(
            in_state_factor, cross_state_factor)
        self.colored_factors = self.color_factors(self.factor_int_mapping)
        self.merged_groups = self.create_groups(self.colored_factors)
        self.in_state_factor_groups, self.cross_state_factor_groups = self.separate_group()
        # for idx, cg in enumerate(self.in_state_factor_groups):
        #     print("IState Group {}".format(idx))
        #     for clu in cg:
        #         print("\t name {} nodes {}".format(clu['name'], clu['nodes']))
        #
        # for idx, cg in enumerate(self.cross_state_factor_groups):
        #     print("CState Group {}:".format(idx))
        #     for clu in cg:
        #         print("\t name: {} cnodes {} action {} nnodes {}".format(clu['name'], clu['cnodes'], clu['action'], clu['nextnodes']))
        self.in_state_start_ids = self.map_in_state_factor_start_ids()
        self.cross_state_st_start_ids, self.cross_state_a_start_ids, self.cross_state_stp_start_ids = self.map_cross_state_factor_start_ids()

    def create_factor_int_mapping(self, in_state_factor, cross_state_factor):
        all_factors = in_state_factor + cross_state_factor
        mapping = {}
        for i, factor in enumerate(all_factors):
            mapping[i] = factor
        return mapping


    def map_cross_state_factor_start_ids(self):
        index_st = 0
        index_a = 0
        index_stp = 0
        start_ids_st = {}
        start_ids_a = {}
        start_ids_stp = {}

        offset = len(self.in_state_factor)
        for factor_id in range(len(self.cross_state_factor)):
            start_ids_st[factor_id] = index_st
            start_ids_a[factor_id] = index_a
            start_ids_stp[factor_id] = index_stp
            index_st += len(
                self.factor_int_mapping[factor_id + offset]['cnodes'])
            index_a += len(self.factor_int_mapping[factor_id+offset]['action'])
            index_stp += len(
                self.factor_int_mapping[factor_id+offset]['nextnodes'])

        return start_ids_st, start_ids_a, start_ids_stp

    def map_in_state_factor_start_ids(self):
        index = 0
        start_ids = {}
        for factor_id in range(len(self.in_state_factor)):
            start_ids[factor_id] = index
            index += len(self.factor_int_mapping[factor_id]['nodes'])
        return start_ids


    def color_factors(self, int_factor_mapping):
        graph = self.create_graph(int_factor_mapping)
        colored_factors = nx.coloring.greedy_color(
            graph, strategy=nx.coloring.strategy_largest_first)
        coloring = defaultdict(list)
        for key, value in colored_factors.items():
            coloring[value].append(key)
        return coloring


    def create_graph(self, int_factor_mapping):
        G = nx.Graph()
        G.add_nodes_from(int_factor_mapping.keys())

        for index in range(len(self.in_state_factor)):
            factor_1 = int_factor_mapping[index]
            included_nodes_factor_1 = factor_1['nodes']
            self.add_edge_in_state(
                G, included_nodes_factor_1, int_factor_mapping, index)
            self.add_edge_cross_state(
                G, int_factor_mapping, included_nodes_factor_1, index)

        for index in range(len(self.in_state_factor), len(self.in_state_factor) + len(self.cross_state_factor)):
            factor_1 = int_factor_mapping[index]
            included_nodes_factor_1 = factor_1['cnodes']
            included_actionvar_factor_1 = factor_1['action']
            included_nnodes_factor_l = factor_1['nextnodes']
            self.add_edge_in_state(G, included_nodes_factor_1, int_factor_mapping, index)
            self.add_edge_cross_state(G, int_factor_mapping, included_nodes_factor_1, index)
            self.add_edge_cross_nstate(G, int_factor_mapping, included_nnodes_factor_l, index)
            self.add_edge_action_nodes(G, int_factor_mapping, included_actionvar_factor_1, index)
        return G


    def add_edge_action_nodes(self, G, int_factor_mapping, included_actionvar_factor_1, factor_1_index):
        for index in range(len(self.in_state_factor), len(self.in_state_factor) + len(self.cross_state_factor)):
            factor_2 = int_factor_mapping[index]
            included_actionvar_factor_2 = factor_2['action']
            if (len(list(set(included_actionvar_factor_1) & set(included_actionvar_factor_2))) > 0):
                G.add_edge(factor_1_index, index)

    def add_edge_cross_state(self, G, int_factor_mapping, included_nodes_factor_1, factor_1_index):
        for index in range(len(self.in_state_factor), len(self.in_state_factor) + len(self.cross_state_factor)):
            factor_2 = int_factor_mapping[index]
            included_nodes_factor_2 = factor_2['cnodes']
            if (len(list(set(included_nodes_factor_1) & set(included_nodes_factor_2))) > 0):
                G.add_edge(factor_1_index, index)

    def add_edge_cross_nstate(self, G, int_factor_mapping, included_nodes_factor_1, factor_1_index):
        for index in range(len(self.in_state_factor), len(self.in_state_factor) + len(self.cross_state_factor)):
            factor_2 = int_factor_mapping[index]
            included_nodes_factor_2 = factor_2['nextnodes']
            if (len(list(set(included_nodes_factor_1) & set(included_nodes_factor_2))) > 0):
                G.add_edge(factor_1_index, index)

    def add_edge_in_state(self, G, included_nodes_factor_1, int_factor_mapping, factor_1_index):
        for index in range(len(self.in_state_factor)):
            factor_2 = int_factor_mapping[index]
            included_nodes_factor_2 = factor_2['nodes']
            if (len(list(set(included_nodes_factor_1) & set(included_nodes_factor_2))) > 0):
                G.add_edge(factor_1_index, index)

    def create_groups(self, colored_factors):
        groups = []
        for color, colored_group in colored_factors.items():
            ungrouped_factors = set(colored_group)
            for factor in colored_group:
                if factor not in ungrouped_factors:               # factor already in a group
                    continue
                new_group = self.find_compatible(factor, ungrouped_factors)
                groups.append(list(new_group))
                ungrouped_factors -= new_group
        return groups


    def find_compatible(self, factor_item, ungrouped_factors):
        compatible = set()
        compatible.add(factor_item)
        if not hasattr(self.factor_int_mapping[factor_item]['Factor'][0], 'potential'):
            if not isinstance(self.factor_int_mapping[factor_item]['Factor'][0], transition.ConvFactor):
                return compatible
        for factor in ungrouped_factors:
            if self.is_compatible(factor_item, factor):
                compatible.add(factor)
        return compatible


    def is_compatible(self, factor1, factor2):
        component_factor1 = self.factor_int_mapping[factor1]['Factor'][0]
        component_factor2 = self.factor_int_mapping[factor2]['Factor'][0]
        # if type(component_factor1) == transition.TreeFactor or type(component_factor2) == transition.TreeFactor:
        #     return False
        if type(component_factor1) == transition.ConvFactor and type(component_factor2) == transition.ConvFactor:
            if component_factor1.ksize == component_factor2.ksize and component_factor1.nchannels == component_factor2.nchannels and component_factor1.var_shape == component_factor2.var_shape:
                return True
            else:
                return False
        if not hasattr(component_factor1, 'potential') or not hasattr(component_factor2, 'potential'):
            return False
        return (type(component_factor1) == type(component_factor2)) and (component_factor1.potential.get_shape() == component_factor2.potential.get_shape())

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

    def split_input(self, input_var, axis=1):
        return utils.split_as_slices(input_var, self.input_parts, axis)

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

    def split_state(self, state_var, axis=1):
        return utils.split_as_slices(state_var, self.state_parts, axis)

    @property
    def output_size(self):
        """
        output: b_tprev, a_tprev, b_tprev->b_tprev, f_t -> b_tprev, f_t -> a_tprev, f_t -> b_t 
        """
        return self.input_size


    def separate_group(self):
        in_state_groups = []
        cross_state_groups = []
        for group in self.merged_groups:
            factor_group = []
            if self.factor_int_mapping[group[0]] in self.in_state_factor:
                for group_member in group:
                    factor_group.append(self.factor_int_mapping[group_member])
                in_state_groups.append(factor_group)
            else:
                for group_member in group:
                    factor_group.append(self.factor_int_mapping[group_member])
                cross_state_groups.append(factor_group)
        return in_state_groups, cross_state_groups

    def __call_2__(self, inputs, state, scope=None):

        bs_t, imsgs_t, ba_t, cmsgs_t, cmsga_t, cmsgs_tp = self.split_input(
            inputs)
        bs_tp, imsgs_tp = self.split_state(state)

        # In state msg passing on tp
        for factor_group_is in self.in_state_factor_groups:
            # grab in_state beliefs for t prime
            belief = [[bs_tp[n] for n in factor_is['nodes']]
                      for factor_is in factor_group_is]
            # grab in_state messages for t prime
            msgs = [imsgs_tp[self.in_state_start_ids[factor_is['id']]:(
                    self.in_state_start_ids[factor_is['id']] + len(factor_is['nodes']))] for factor_is in factor_group_is]

            # LoopyBP require three parameters: 1. a list of higher order factors, 2. a list of beliefs, 3. a list of messages
            nbeliefs, nmsg = type(factor_group_is[0]['Factor'][0]).LoopyBP(
                [factor_is['Factor'][0] for factor_is in factor_group_is], belief, msgs, None)
            # print('basic infer unit', nbeliefs, nmsg)

            # write the updated message and belief back
            for factor_id, factor_is in enumerate(factor_group_is):
                for idx, nidx in enumerate(factor_is['nodes']):
                    # print(nbeliefs[factor_id])
                    bs_tp[nidx] = nbeliefs[factor_id][idx]
                    # print(idx, bs_t[nidx])

                imsgs_tp[self.in_state_start_ids[factor_is['id']]:(
                    self.in_state_start_ids[factor_is['id']] + len(factor_is['nodes']))] = nmsg[factor_id]

        for factor_group_cs in self.cross_state_factor_groups:
            belief_t_s = [[bs_t[n] for n in factor_cs['cnodes']]
                          for factor_cs in factor_group_cs]  # belief state t
            belief_action = [[ba_t[n] for n in factor_cs['action']]
                             for factor_cs in factor_group_cs]  # belief action t
            belief_tp_s = [[bs_tp[n] for n in factor_cs['nextnodes']]
                           for factor_cs in factor_group_cs]  # belief state t prime

            # message is a list which have exactly the same shape as belief
            # for time t, message to state is in cmsgs_t
            #             message to action in in cmsga_t
            #     time t_prime, message to state is in cmsgs_tp
            # maybe in the version with merged factor, we should some how re-organize the cmsgs_t and cmsgs_tp
            msg_all = [cmsgs_t[self.cross_state_st_start_ids[factor_cs['id']]:(self.cross_state_st_start_ids[factor_cs['id']] + len(factor_cs['cnodes']))]
                       + cmsga_t[self.cross_state_a_start_ids[factor_cs['id']]:(
                           self.cross_state_a_start_ids[factor_cs['id']] + len(factor_cs['action']))]
                       + cmsgs_tp[self.cross_state_stp_start_ids[factor_cs['id']]:(
                           self.cross_state_stp_start_ids[factor_cs['id']] + len(factor_cs['nextnodes']))]
                       for factor_cs in factor_group_cs]

            beliefs = [belief_t_s[i] + belief_action[i] + belief_tp_s[i]
                       for i in range(len(factor_group_cs))]
            # print('basic infer unit input ', beliefs, msg_all)

            # LoopyBP require three parameters: 1. a list of higher order factors, 2. a list of beliefs, 3. a list of messages
            nbeliefs, nmsg = type(factor_group_cs[0]['Factor'][0]).LoopyBP(
                [factor_cs['Factor'][0] for factor_cs in factor_group_cs], beliefs, msg_all, None)
            # print('basic infer unit', nbeliefs, nmsg)

            # after updating all messgeas and beliefs, update the corresponding variable so that in next
            # iteration we are using the updated belief and message (as we are using coordinate descent, the beliefs and messages must be updated in order)
            for factor_id, factor_cs in enumerate(factor_group_cs):
                for idx, nidx in enumerate(factor_cs['cnodes']):
                    bs_t[nidx] = nbeliefs[factor_id][idx]
                    # print(idx, bs_t[nidx])
                cmsgs_t[self.cross_state_st_start_ids[factor_cs['id']]:(self.cross_state_st_start_ids[factor_cs['id']] + len(factor_cs['cnodes']))] = \
                    nmsg[factor_id][0:len(factor_cs['cnodes'])]

                for ii, nidx in enumerate(factor_cs['action']):
                    idx = ii + len(factor_cs['cnodes'])
                    ba_t[nidx] = nbeliefs[factor_id][idx]
                cmsga_t[self.cross_state_a_start_ids[factor_cs['id']]:(self.cross_state_a_start_ids[factor_cs['id']] + len(factor_cs['action']))] = \
                    nmsg[factor_id][len(factor_cs['cnodes']):(
                        len(factor_cs['cnodes']) + len(factor_cs['action']))]

                for ii, nidx in enumerate(factor_cs['nextnodes']):
                    idx = ii + len(factor_cs['cnodes']) + \
                        len(factor_cs['action'])
                    bs_tp[nidx] = nbeliefs[0][idx]
                cmsgs_tp[self.cross_state_stp_start_ids[factor_cs['id']]:(self.cross_state_stp_start_ids[factor_cs['id']] + len(factor_cs['nextnodes']))] = \
                    nmsg[factor_id][(len(factor_cs['cnodes']) +
                                     len(factor_cs['action'])):]

        # enf of cross_state msg passing

        # update in state message, belief for time t
        # it has exactly the same structure as time t_prime
        for factor_group_is in self.in_state_factor_groups:
            # grab in_state beliefs for t prime
            belief = [[bs_t[n] for n in factor_is['nodes']]
                      for factor_is in factor_group_is]
            # grab in_state messages for t prime
            msgs = [imsgs_t[self.in_state_start_ids[factor_is['id']]:(
                    self.in_state_start_ids[factor_is['id']] + len(factor_is['nodes']))] for factor_is in factor_group_is]

            # LoopyBP require three parameters: 1. a list of higher order factors, 2. a list of beliefs, 3. a list of messages
            nbeliefs, nmsg = type(factor_group_is[0]['Factor'][0]).LoopyBP(
                [factor_is['Factor'][0] for factor_is in factor_group_is], belief, msgs, None)

            # write the updated message and belief back
            for factor_id, factor_is in enumerate(factor_group_is):
                for idx, nidx in enumerate(factor_is['nodes']):
                    bs_t[nidx] = nbeliefs[factor_id][idx]
                imsgs_t[self.in_state_start_ids[factor_is['id']]:(
                        self.in_state_start_ids[factor_is['id']] + len(factor_is['nodes']))] = nmsg[factor_id]

        # Output Organization: bs_prevt, bs_prevt->bs_prevt, a_prevt, ft->bs_prevt, ft->a_prevt, ft->bs_t
        # Organization of output states: bs_t, bs_t -> bs_t

        output = tf.concat(bs_tp + imsgs_tp + ba_t +
                           cmsgs_t + cmsga_t + cmsgs_tp, axis=1)
        output_state = tf.concat(bs_t + imsgs_t, axis=1)

        return output, output_state



    def __call__(self, inputs, state, scope=None):
        """
        See link:
        The basic inference unit in the network.
        state: b_{s_{tp1}}, m(b_{s_{tp1}}->b_{s_{tp1}})
        input: b_{s_t}, b_{a_t}, m(b_{s_t}->b_{s_t}), m(t->b_{s_t}), m(t->b_{a_t}), m(t->b_{s_tp1})
        output_state: b_{s_t}, m(b_{s_{t}}->b_st)
        output: b_{s_{tp1}}, b_{a_t}, m(b_{s_t}->b_{s_t}), m(t->b_{s_t}), m(t->b_{a_t}), m(t->b_{s_tp1})
        """
        if self.merge_factors == True:
            return self.__call_2__(inputs, state, scope)

        bs_t, imsgs_t, ba_t, cmsgs_t, cmsga_t, cmsgs_tp = self.split_input(
            inputs)
        bs_tp, imsgs_tp = self.split_state(state)

        # In state msg passing on tp
        is_msg_start_idx = 0
        for factor in self.in_state_factor:
            # grab in_state beliefs for t prime
            belief = [[bs_tp[n] for n in factor['nodes']]]
            # grab in_state messages for t prime
            msgs = [imsgs_tp[is_msg_start_idx:(
                is_msg_start_idx + len(factor['nodes']))]]

            # LoopyBP require three parameters: 1. a list of higher order factors, 2. a list of beliefs, 3. a list of messages
            nbeliefs, nmsg = type(factor['Factor'][0]).LoopyBP(
                [factor['Factor'][0]], belief, msgs, None)

            # write the updated message and belief back
            for idx, nidx in enumerate(factor['nodes']):
                bs_tp[nidx] = nbeliefs[0][idx]
            imsgs_tp[is_msg_start_idx:(
                is_msg_start_idx + len(factor['nodes']))] = nmsg[0]

            is_msg_start_idx += len(factor['nodes'])

        cs_msg_start_t_idx = 0
        cs_msg_start_prev_t_idx = 0
        cs_msg_action_t_idx = 0

        for factor in self.cross_state_factor:
            t_s_node = factor['cnodes']  # state t node
            a_node = factor['action']  # action t node
            tp_s_node = factor['nextnodes']  # state t_prime node

            belief_t_s = [bs_t[nidx] for nidx in t_s_node]  # belief state t
            belief_action = [ba_t[nidx] for nidx in a_node]  # belief action t
            belief_tp_s = [bs_tp[nidx]
                           for nidx in tp_s_node]  # belief state t prime

            # message is a list which have exactly the same shape as belief
            # for time t, message to state is in cmsgs_t
            #             message to action in in cmsga_t
            #     time t_prime, message to state is in cmsgs_tp
            # maybe in the version with merged factor, we should some how re-organize the cmsgs_t and cmsgs_tp
            msg_all = [cmsgs_t[cs_msg_start_prev_t_idx:(cs_msg_start_prev_t_idx + len(t_s_node))]
                       + cmsga_t[cs_msg_action_t_idx:(cs_msg_action_t_idx + len(a_node))]
                       + cmsgs_tp[cs_msg_start_t_idx:(cs_msg_start_t_idx + len(tp_s_node))]]

            beliefs = [belief_t_s + belief_action + belief_tp_s]

            # LoopyBP require three parameters: 1. a list of higher order factors, 2. a list of beliefs, 3. a list of messages
            nbeliefs, nmsg = type(factor['Factor'][0]).LoopyBP(
                [factor['Factor'][0]], beliefs, msg_all, None)

            # after updating all messgeas and beliefs, update the corresponding variable so that in next
            # iteration we are using the updated belief and message (as we are using coordinate descent, the beliefs and messages must be updated in order)
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

        # enf of cross_state msg passing

        # update in state message, belief for time t
        # it has exactly the same structure as time t_prime
        is_msg_start_idx = 0
        for factor in self.in_state_factor:
            belief = [[bs_t[n] for n in factor['nodes']]]
            msgs = [imsgs_t[is_msg_start_idx:(
                is_msg_start_idx + len(factor['nodes']))]]
            nbeliefs, nmsg = type(factor['Factor'][0]).LoopyBP(
                [factor['Factor'][0]], belief, msgs, None)

            for idx, nidx in enumerate(factor['nodes']):
                bs_t[nidx] = nbeliefs[0][idx]
            imsgs_t[is_msg_start_idx:(
                is_msg_start_idx + len(factor['nodes']))] = nmsg[0]

            is_msg_start_idx += len(factor['nodes'])

        # Output Organization: bs_prevt, bs_prevt->bs_prevt, a_prevt, ft->bs_prevt, ft->a_prevt, ft->bs_t
        # Organization of output states: bs_t, bs_t -> bs_t

        # Output gate in beliefs

        # for idx, b in enumerate(bs_tp):
        #     fixed_b = (b - tf.reduce_max(b, keepdims=True)) * 100
        #     b_normalized = b - tf.reduce_logsumexp(b, keepdims=True)
        #     entropy = - tf.exp(b_normalized) * b_normalized
        #     entropy = tf.reduce_sum(entropy, keepdims=True)

        #     scale = tf.exp(-entropy)

        #     bs_tp[idx] = (1 - scale) * b + scale * fixed_b

        output = tf.concat(bs_tp + imsgs_tp + ba_t +
                           cmsgs_t + cmsga_t + cmsgs_tp, axis=1)
        output_state = tf.concat(bs_t + imsgs_t, axis=1)

        return output, output_state

class InferNetRNN:
    def __init__(self, sizesnodes, sizeanodes, simulate_steps,
                 in_state_factor,
                 cross_state_factor,
                 BPIter=5):
        self.value_factor = None
        self.sub_cell = BasicInferUnit(
            sizesnodes, sizeanodes, in_state_factor, cross_state_factor)
        self.sizesnodes = sizesnodes
        self.sizeanodes = sizeanodes
        self.create_init_state()

        self.all_input = tf.zeros([1, simulate_steps - 1, self.sub_cell.input_size])\
            + tf.expand_dims(self.input_zero_brd, axis=2)
        self.all_input = tf.concat(
            [self.all_input, self.final_step_input], axis=1)

        self.objv = []
        init_state = self.zero_state
        all_input = self.all_input
        for i in range(BPIter):
            transition.Factors.ExpandHOPs = dict()
            outputs, state = tf.nn.dynamic_rnn(self.sub_cell, all_input,
                                               initial_state=init_state)

            bprevt_part, others = tf.split(outputs, [
                                           self.sub_cell.state_size, self.sub_cell.input_size - self.sub_cell.state_size], axis=2)

            beliefs, _ = tf.split(bprevt_part, [np.sum(sizesnodes), np.sum(
                self.sub_cell.in_state_msg_siz)], axis=2)
            tbeliefs, _ = tf.split(state, [np.sum(sizesnodes), np.sum(
                self.sub_cell.in_state_msg_siz)], axis=1)

            asize = np.sum(self.sub_cell.sizeanodes)
            beliefa, _ = tf.split(others, [
                                  asize, self.sub_cell.input_size - self.sub_cell.state_size - asize], axis=2)
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

            init_state, bprevt_part = tf.split(
                bprevt_part, [1, simulate_steps - 1], axis=1)
            bprevt_part = tf.concat(
                [bprevt_part, tf.expand_dims(state, 1)], axis=1)

            all_input = tf.concat([bprevt_part, others], axis=2)
            init_state = init_state[:, 0, :]

        final_state1, _, final_action, _, _, _ = self.sub_cell.split_input(
            tf.transpose(outputs, perm=[0, 2, 1]))

        final_final_state, _, final_final_action, _, _, _ = self.sub_cell.split_input(
            outputs[:, -1, :])
        self.final_action = final_action
        self.final_state = final_state1
        self.final_output = outputs
        self.belief_state_0 = state
        self.final_final_state = final_final_state
        self.final_final_action = final_final_action

    def create_init_state(self):
        self.init_belief = tf.placeholder(
            tf.float32, [None] + [np.sum(self.sub_cell.sizesnodes)])
        self.input_zero_brd = tf.reduce_sum(
            tf.zeros_like(self.init_belief), axis=1, keep_dims=True)
        init_msg = tf.zeros([1, np.sum(self.sub_cell.in_state_msg_siz)]) + \
            self.input_zero_brd

        def update_value_msg(belief, msg):
            belief = tf.split(belief, self.sizesnodes, axis=1)
            msg = tf.split(belief, self.sizesnodes, axis=1)

            nbelief, msg = self.value_factor.LoopyBP(
                [self.value_factor], [belief], [msg])

            return tf.concat(nbelief[0], axis=1), tf.concat(msg[0], axis=1)

        if(self.value_factor is not None):
            value_msg = tf.zeros_like(self.init_belief)
            self.value_factor = self.value_factor
            ninit_belief, self.value_msg = update_value_msg(
                self.init_belief, value_msg)
        else:
            self.value_msg = self.init_belief[:, 0:0]
            ninit_belief = self.init_belief

        self.init_state = tf.concat([ninit_belief, init_msg], axis=1)
        self.zero_state = tf.zeros_like(self.init_state)

        input_padding = tf.zeros(
            [1, self.sub_cell.input_size - self.sub_cell.state_size])
        input_padding = input_padding + self.input_zero_brd

        self.final_step_input = tf.expand_dims(
            tf.concat([self.init_state, input_padding], axis=1), 1)


class InferNetNoRepeatComputeRNN:
    def __init__(self, sizesnodes, sizeanodes, simulate_steps,
                 in_state_factor,
                 cross_state_factor,
                 BPIter=5):
        self.value_factor = None
        self.sub_cell = BasicInferUnitNoRepeat(
            sizesnodes, sizeanodes, in_state_factor, cross_state_factor)

        #self.sub_cell_reverse = BasicInferUnitReverse(sizesnodes, sizeanodes, in_state_factor, cross_state_factor)
        self.sizesnodes = sizesnodes
        self.sizeanodes = sizeanodes

        self.create_init_state()
        self.all_input = tf.zeros([1, simulate_steps - 1, self.sub_cell.input_size])\
            + tf.expand_dims(self.input_zero_brd, axis=2)
        self.all_input = tf.concat(
            [self.all_input, self.final_step_input], axis=1)
        self.objv = []
        all_input = self.all_input
        in_state_input = self.zero_state
        for i in range(BPIter):
            transition.Factors.ExpandHOPs = dict()
            bs_T, imsgs_T = self.sub_cell.doFirstInStateFactorInference(in_state_input)
            transition.Factors.ExpandHOPs = dict()
            outputs, state = tf.nn.dynamic_rnn(self.sub_cell, all_input,
                                               initial_state=bs_T)
            bprevt_part, others = tf.split(outputs, [
                                           self.sub_cell.state_size, self.sub_cell.input_size - self.sub_cell.state_size], axis=2)

            beliefs, _ = tf.split(bprevt_part, [np.sum(sizesnodes),0], axis=2)
            tbeliefs, _ = tf.split(state, [np.sum(sizesnodes), 0], axis=1)

            asize = np.sum(self.sub_cell.sizeanodes)
            _, beliefa, _ = tf.split(others, [np.sum(self.sub_cell.in_state_msg_siz),
                                  asize, self.sub_cell.input_size - self.sub_cell.state_size - asize - np.sum(self.sub_cell.in_state_msg_siz)], axis=2)
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

            in_state_input, bprevt_part = tf.split(
                bprevt_part, [1, simulate_steps - 1], axis=1)
            bprevt_part = tf.concat(
                [bprevt_part, tf.expand_dims(state, 1)], axis=1)
            in_state_input = in_state_input[:, 0, :]
            in_state_input = tf.concat([in_state_input, imsgs_T], axis=1)
            all_input = tf.concat([bprevt_part, others], axis=2)
        final_state1, _, final_action, _, _, _ = self.sub_cell.split_input(tf.transpose(outputs, perm=[0, 2, 1]))
        final_final_state, _, final_final_action, _, _, _ = self.sub_cell.split_input(outputs[:, -1, :])
        self.final_action = final_action
        self.final_state = final_state1
        self.final_output = outputs
        self.imsgs_T = imsgs_T
        self.final_final_state = final_final_state
        self.final_final_action = final_final_action
        self.belief_state_0 = state


    def create_init_state(self):
        self.init_belief = tf.placeholder(
            tf.float32, [None] + [np.sum(self.sub_cell.sizesnodes)])
        self.input_zero_brd = tf.reduce_sum(
            tf.zeros_like(self.init_belief), axis=1, keep_dims=True)
        init_msg = tf.zeros([1, np.sum(self.sub_cell.in_state_msg_siz)]) + \
            self.input_zero_brd

        def update_value_msg(belief, msg):
            belief = tf.split(belief, self.sizesnodes, axis=1)
            msg = tf.split(belief, self.sizesnodes, axis=1)

            nbelief, msg = self.value_factor.LoopyBP(
                [self.value_factor], [belief], [msg])

            return tf.concat(nbelief[0], axis=1), tf.concat(msg[0], axis=1)

        if(self.value_factor is not None):
            value_msg = tf.zeros_like(self.init_belief)
            self.value_factor = self.value_factor
            self.ninit_belief, self.value_msg = update_value_msg(
                self.init_belief, value_msg)
        else:
            self.value_msg = self.init_belief[:, 0:0]
            self.ninit_belief = self.init_belief

        self.init_state = tf.concat([self.ninit_belief, init_msg], axis=1)
        self.zero_state = tf.zeros_like(self.init_state)

        input_padding = tf.zeros(
            [1, self.sub_cell.input_size - self.sub_cell.state_size])
        input_padding = input_padding + self.input_zero_brd

        self.final_step_input = tf.expand_dims(
            tf.concat([self.ninit_belief, input_padding], axis=1), 1)


class InferNetPipeLine :
    def __init__(self):
        print("Removed : Dummy object for backward compatibility")


class InferNetNoRepeatComputeRNNBothWaysBP:
    def __init__(self, sizesnodes, sizeanodes, simulate_steps,
                 in_state_factor,
                 cross_state_factor,
                 BPIter=5):
        self.value_factor = None
        self.sub_cell = BasicInferUnitNoRepeat(
            sizesnodes, sizeanodes, in_state_factor, cross_state_factor)

        self.sub_cell_reverse = BasicInferUnitReverse(sizesnodes, sizeanodes, in_state_factor, cross_state_factor)
        self.sizesnodes = sizesnodes
        self.sizeanodes = sizeanodes

        self.create_init_state()
        self.all_input = tf.zeros([1, simulate_steps - 1, self.sub_cell.input_size])\
            + tf.expand_dims(self.input_zero_brd, axis=2)
        self.all_input = tf.concat(
            [self.all_input, self.final_step_input], axis=1)
        self.objv = []
        all_input = self.all_input
        in_state_input = self.zero_state
        for i in range(BPIter):
            transition.Factors.ExpandHOPs = dict()
            bs_T, imsgs_T = self.sub_cell.doFirstInStateFactorInference(in_state_input)
            outputs, state = tf.nn.dynamic_rnn(self.sub_cell, all_input,
                                               initial_state=bs_T)
            inputs_reverse = tf.reverse(outputs, axis=[1])
            transition.Factors.ExpandHOPs = dict()
            outputs_reverse, state_T = tf.nn.dynamic_rnn(self.sub_cell_reverse, inputs=inputs_reverse,initial_state=state)
            last_IS_factor_input = tf.concat([state_T, imsgs_T], axis=1)
            transition.Factors.ExpandHOPs = dict()
            bs_T, imsgs_T = self.sub_cell_reverse.doFirstInStateFactorInference(last_IS_factor_input)

            bprevt_part, others = tf.split(outputs_reverse, [
                                           self.sub_cell.state_size, self.sub_cell.input_size - self.sub_cell.state_size], axis=2)

            beliefs, _ = tf.split(bprevt_part, [np.sum(sizesnodes),0], axis=2)
            tbeliefs, _ = tf.split(state_T, [np.sum(sizesnodes), 0], axis=1)

            asize = np.sum(self.sub_cell.sizeanodes)
            _, beliefa, _ = tf.split(others, [np.sum(self.sub_cell.in_state_msg_siz),
                                  asize, self.sub_cell.input_size - self.sub_cell.state_size - asize - np.sum(self.sub_cell.in_state_msg_siz)], axis=2)
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

            # in_state_input, bprevt_part = tf.split(
            #     bprevt_part, [1, simulate_steps - 1], axis=1)
            # bprevt_part = tf.concat(
            #     [bprevt_part, tf.expand_dims(state, 1)], axis=1)
            # in_state_input = in_state_input[:, 0, :]
            # in_state_input = tf.concat([in_state_input, imsgs_T], axis=1)
            # all_input = tf.concat([bprevt_part, others], axis=2)

            in_state_input = tf.concat([bs_T, imsgs_T], axis=1)
            all_input = tf.reverse(outputs_reverse, axis=[1])

        outputs = tf.reverse(outputs_reverse, axis=[1])
        final_state1, _, final_action, _, _, _ = self.sub_cell.split_input(
            tf.transpose(outputs, perm=[0, 2, 1]))

        final_final_state, _, final_final_action, _, _, _ = self.sub_cell.split_input(
            outputs[:, -1, :])
        self.final_action = final_action
        self.final_state = final_state1
        self.final_output = outputs
        self.final_final_state = final_final_state
        self.final_final_action = final_final_action

    def create_init_state(self):
        self.init_belief = tf.placeholder(
            tf.float32, [None] + [np.sum(self.sub_cell.sizesnodes)])
        self.input_zero_brd = tf.reduce_sum(
            tf.zeros_like(self.init_belief), axis=1, keep_dims=True)
        init_msg = tf.zeros([1, np.sum(self.sub_cell.in_state_msg_siz)]) + \
            self.input_zero_brd

        def update_value_msg(belief, msg):
            belief = tf.split(belief, self.sizesnodes, axis=1)
            msg = tf.split(belief, self.sizesnodes, axis=1)

            nbelief, msg = self.value_factor.LoopyBP(
                [self.value_factor], [belief], [msg])

            return tf.concat(nbelief[0], axis=1), tf.concat(msg[0], axis=1)

        if(self.value_factor is not None):
            value_msg = tf.zeros_like(self.init_belief)
            self.value_factor = self.value_factor
            self.ninit_belief, self.value_msg = update_value_msg(
                self.init_belief, value_msg)
        else:
            self.value_msg = self.init_belief[:, 0:0]
            self.ninit_belief = self.init_belief

        self.init_state = tf.concat([self.ninit_belief, init_msg], axis=1)
        self.zero_state = tf.zeros_like(self.init_state)

        input_padding = tf.zeros(
            [1, self.sub_cell.input_size - self.sub_cell.state_size])
        input_padding = input_padding + self.input_zero_brd

        self.final_step_input = tf.expand_dims(
            tf.concat([self.ninit_belief, input_padding], axis=1), 1)


class BasicInferUnitNoRepeat(BasicInferUnit):
    """
    The basic inference unit in the network.
    state: b_{s_{tp1}}, m(b_{s_{tp1}}->b_{s_{tp1}})
    input: b_{s_t}, b_{a_t}, m(b_{s_t}->b_{s_t}), m(t->b_{s_t}), m(t->b_{a_t}), m(t->b_{s_tp1})
    output_state: b_{s_t}
    output: b_{s_{tp1}}, b_{a_t}, m(b_{s_t}->b_{s_t}), m(t->b_{s_t}), m(t->b_{a_t}), m(t->b_{s_tp1})
    """

    def __init__(self, sizesnodes, sizeanodes,
                 in_state_factor, cross_state_factor):
        super(BasicInferUnitNoRepeat, self).__init__(sizesnodes, sizeanodes, in_state_factor, cross_state_factor)
        self.state_parts = [self.sizesnodes]

    @property
    def state_size(self):
        total_dim = np.sum(self.sizesnodes)
        return total_dim

    def doFirstInStateFactorInference(self, inputs, scope=None):
        #In state msg passing on tp
        bs_t, imsgs_t = utils.split_as_slices(inputs, [self.sizesnodes, self.in_state_msg_siz])
        for factor_group_is in self.in_state_factor_groups:
            # grab in_state beliefs for t prime
            belief = [[bs_t[n] for n in factor_is['nodes']] for factor_is in factor_group_is]
            # grab in_state messages for t prime
            msgs = [imsgs_t[self.in_state_start_ids[factor_is['id']]:(
                    self.in_state_start_ids[factor_is['id']] + len(factor_is['nodes']))] for factor_is in
                    factor_group_is]

            # LoopyBP require three parameters: 1. a list of higher order factors, 2. a list of beliefs, 3. a list of messages
            nbeliefs, nmsg = type(factor_group_is[0]['Factor'][0]).LoopyBP(
                [factor_is['Factor'][0] for factor_is in factor_group_is], belief, msgs, None)

            # write the updated message and belief back
            for factor_id, factor_is in enumerate(factor_group_is):
                for idx, nidx in enumerate(factor_is['nodes']):
                    bs_t[nidx] = nbeliefs[factor_id][idx]
                imsgs_t[self.in_state_start_ids[factor_is['id']]:(
                        self.in_state_start_ids[factor_is['id']] + len(factor_is['nodes']))] = nmsg[factor_id]
        return tf.concat(bs_t, axis=1), tf.concat(imsgs_t, axis=1)


    def __call__(self, inputs, state, scope=None):
        bs_t, imsgs_t, ba_t, cmsgs_t, cmsga_t, cmsgs_tp = self.split_input(inputs)
        bs_tp = self.split_state(state)[0]

        for factor_group_cs in self.cross_state_factor_groups:
            belief_t_s = [[bs_t[n] for n in factor_cs['cnodes']] for factor_cs in factor_group_cs]  # belief state t
            belief_action = [[ba_t[n] for n in factor_cs['action']] for factor_cs in factor_group_cs]  # belief action t
            belief_tp_s = [[bs_tp[n] for n in factor_cs['nextnodes']] for factor_cs in
                           factor_group_cs]  # belief state t prime
            msg_all = [cmsgs_t[self.cross_state_st_start_ids[factor_cs['id']]:(
                    self.cross_state_st_start_ids[factor_cs['id']] + len(factor_cs['cnodes']))]
                       + cmsga_t[self.cross_state_a_start_ids[factor_cs['id']]:(
                    self.cross_state_a_start_ids[factor_cs['id']] + len(factor_cs['action']))]
                       + cmsgs_tp[self.cross_state_stp_start_ids[factor_cs['id']]:(
                    self.cross_state_stp_start_ids[factor_cs['id']] + len(factor_cs['nextnodes']))]
                       for factor_cs in factor_group_cs]

            beliefs = [belief_t_s[i] + belief_action[i] + belief_tp_s[i] for i in range(len(factor_group_cs))]

            nbeliefs, nmsg = type(factor_group_cs[0]['Factor'][0]).LoopyBP(
                [factor_cs['Factor'][0] for factor_cs in factor_group_cs], beliefs, msg_all, None)
            for factor_id, factor_cs in enumerate(factor_group_cs):
                for idx, nidx in enumerate(factor_cs['cnodes']):
                    bs_t[nidx] = nbeliefs[factor_id][idx]
                cmsgs_t[self.cross_state_st_start_ids[factor_cs['id']]:(
                        self.cross_state_st_start_ids[factor_cs['id']] + len(factor_cs['cnodes']))] = \
                    nmsg[factor_id][0:len(factor_cs['cnodes'])]

                for ii, nidx in enumerate(factor_cs['action']):
                    idx = ii + len(factor_cs['cnodes'])
                    ba_t[nidx] = nbeliefs[factor_id][idx]
                cmsga_t[self.cross_state_a_start_ids[factor_cs['id']]:(
                        self.cross_state_a_start_ids[factor_cs['id']] + len(factor_cs['action']))] = \
                    nmsg[factor_id][len(factor_cs['cnodes']):(len(factor_cs['cnodes']) + len(factor_cs['action']))]

                for ii, nidx in enumerate(factor_cs['nextnodes']):
                    idx = ii + len(factor_cs['cnodes']) + len(factor_cs['action'])
                    bs_tp[nidx] = nbeliefs[0][idx]
                cmsgs_tp[self.cross_state_stp_start_ids[factor_cs['id']]:(
                        self.cross_state_stp_start_ids[factor_cs['id']] + len(factor_cs['nextnodes']))] = \
                    nmsg[factor_id][(len(factor_cs['cnodes']) + len(factor_cs['action'])):]


        for factor_group_is in self.in_state_factor_groups:
            belief = [[bs_t[n] for n in factor_is['nodes']] for factor_is in factor_group_is]
            msgs = [imsgs_t[self.in_state_start_ids[factor_is['id']]:(
                    self.in_state_start_ids[factor_is['id']] + len(factor_is['nodes']))] for factor_is in
                    factor_group_is]
            nbeliefs, nmsg = type(factor_group_is[0]['Factor'][0]).LoopyBP(
                [factor_is['Factor'][0] for factor_is in factor_group_is], belief, msgs, None)
            for factor_id, factor_is in enumerate(factor_group_is):
                for idx, nidx in enumerate(factor_is['nodes']):
                    bs_t[nidx] = nbeliefs[factor_id][idx]
                imsgs_t[self.in_state_start_ids[factor_is['id']]:(
                        self.in_state_start_ids[factor_is['id']] + len(factor_is['nodes']))] = nmsg[factor_id]

        output = tf.concat(bs_tp + imsgs_t + ba_t + cmsgs_t + cmsga_t + cmsgs_tp, axis = 1)
        output_state =  tf.concat(bs_t, axis=1)
        return output, output_state


class BasicInferUnitReverse(BasicInferUnitNoRepeat) :
    def __call__(self, inputs, state, scope=None):
        bs_t, imsgs_tminus, ba_tminus, cmsgs_tminus, cmsga_tminus, cmsgs_t = self.split_input(inputs)
        bs_tminus = self.split_state(state)[0]

        for factor_group_is in self.in_state_factor_groups:
            belief = [[bs_tminus[n] for n in factor_is['nodes']] for factor_is in factor_group_is]
            msgs = [imsgs_tminus[self.in_state_start_ids[factor_is['id']]:(
                    self.in_state_start_ids[factor_is['id']] + len(factor_is['nodes']))] for factor_is in
                    factor_group_is]

            nbeliefs, nmsg = type(factor_group_is[0]['Factor'][0]).LoopyBP(
                [factor_is['Factor'][0] for factor_is in factor_group_is], belief, msgs, None)

            for factor_id, factor_is in enumerate(factor_group_is):
                for idx, nidx in enumerate(factor_is['nodes']):
                    bs_tminus[nidx] = nbeliefs[factor_id][idx]
                imsgs_tminus[self.in_state_start_ids[factor_is['id']]:(
                        self.in_state_start_ids[factor_is['id']] + len(factor_is['nodes']))] = nmsg[factor_id]

        for factor_group_cs in self.cross_state_factor_groups:
            belief_t_s = [[bs_t[n] for n in factor_cs['nextnodes']] for factor_cs in
                          factor_group_cs]
            belief_actiontminus = [[ba_tminus[n] for n in factor_cs['action']] for factor_cs in factor_group_cs]
            belief_tminus_s = [[bs_tminus[n] for n in factor_cs['cnodes']] for factor_cs in factor_group_cs]

            msg_all = [cmsgs_tminus[self.cross_state_st_start_ids[factor_cs['id']]:(
                    self.cross_state_st_start_ids[factor_cs['id']] + len(factor_cs['cnodes']))]
                       + cmsga_tminus[self.cross_state_a_start_ids[factor_cs['id']]:(
                    self.cross_state_a_start_ids[factor_cs['id']] + len(factor_cs['action']))]
                       + cmsgs_t[self.cross_state_stp_start_ids[factor_cs['id']]:(
                    self.cross_state_stp_start_ids[factor_cs['id']] + len(factor_cs['nextnodes']))]
                       for factor_cs in factor_group_cs]

            beliefs = [belief_tminus_s[i] + belief_actiontminus[i] + belief_t_s[i] for i in range(len(factor_group_cs))]
            nbeliefs, nmsg = type(factor_group_cs[0]['Factor'][0]).LoopyBP(
                [factor_cs['Factor'][0] for factor_cs in factor_group_cs], beliefs, msg_all, None)
            for factor_id, factor_cs in enumerate(factor_group_cs):
                for idx, nidx in enumerate(factor_cs['cnodes']):
                    bs_tminus[nidx] = nbeliefs[factor_id][idx]
                cmsgs_tminus[self.cross_state_st_start_ids[factor_cs['id']]:(
                        self.cross_state_st_start_ids[factor_cs['id']] + len(factor_cs['cnodes']))] = \
                    nmsg[factor_id][0:len(factor_cs['cnodes'])]

                for ii, nidx in enumerate(factor_cs['action']):
                    idx = ii + len(factor_cs['cnodes'])
                    ba_tminus[nidx] = nbeliefs[factor_id][idx]
                cmsga_tminus[self.cross_state_a_start_ids[factor_cs['id']]:(
                        self.cross_state_a_start_ids[factor_cs['id']] + len(factor_cs['action']))] = \
                    nmsg[factor_id][len(factor_cs['cnodes']):(len(factor_cs['cnodes']) + len(factor_cs['action']))]

                for ii, nidx in enumerate(factor_cs['nextnodes']):
                    idx = ii + len(factor_cs['cnodes']) + len(factor_cs['action'])
                    bs_t[nidx] = nbeliefs[0][idx]
                cmsgs_t[self.cross_state_stp_start_ids[factor_cs['id']]:(
                        self.cross_state_stp_start_ids[factor_cs['id']] + len(factor_cs['nextnodes']))] = \
                    nmsg[factor_id][(len(factor_cs['cnodes']) + len(factor_cs['action'])):]

        output = tf.concat(bs_tminus + imsgs_tminus + ba_tminus + cmsgs_tminus + cmsga_tminus + cmsgs_t, axis = 1)
        output_state =  tf.concat(bs_t, axis=1)
        return output, output_state
