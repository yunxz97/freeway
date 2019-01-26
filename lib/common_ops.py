import tensorflow as tf
import numpy as np


def to_log_scale(potential):
    if isinstance(potential, tf.Tensor):
        res = tf.log(tf.clip_by_value(potential, 1e-38, 1000))
    else:
        res = np.log(np.clip(potential, 1e-38, 10000)).astype(np.float32)
        idx = np.where(res < -10)
        res[idx] = -1e20
        idx = np.where(res > 5)
        res[idx] = 100
    return res


def cart_prod(a, b):
    """
    Generate cart_prod of 2 d tensor a and b
    """
    final_length = int(a.shape[1] + b.shape[1])
    tile_a = tf.tile(tf.expand_dims(a, 1), [1, tf.shape(b)[0], 1])
    tile_b = tf.tile(tf.expand_dims(b, 0), [tf.shape(a)[0], 1, 1])

    cart_ab = tf.concat([tile_a, tile_b], axis=2)
    return tf.reshape(cart_ab, [-1, final_length])


def cart_prod_assign(a, b):
    final_length = int(a.shape[1] + b.shape[1])
    tile_a = tf.tile(tf.expand_dims(a, 3), [1, 1, 1, tf.shape(b)[2]])
    tile_b = tf.tile(tf.expand_dims(b, 2), [1, 1, tf.shape(a)[2], 1])

    cart_ab = tf.concat([tile_a, tile_b], axis=1)
    cart_ab = tf.reshape(
        cart_ab,
        [-1, final_length, tf.shape(b)[2] * tf.shape(a)[2]])
    return cart_ab


def create_candidates(batch_size, vsize):
    candi = tf.constant([[list(range(vsize))]], dtype=tf.int32)
    return tf.tile(candi, [batch_size, 1, 1])


def gen_partial_assign(nodes_list, assigned_nodes, offsets, cid_in_factor,
                       c_node_idx, candidates, partial_assign, partial_idx,
                       nexisting_value):
    for idx_, nid_ in enumerate(nodes_list):
        if (nid_ < c_node_idx):
            partial_idx.append(idx_ + offsets)
            partial_assign.append(assigned_nodes[nid_])
    partial_idx.append(cid_in_factor + offsets)
    # expand on assignments
    if len(partial_assign) > 0:
        partial_assign = tf.concat(partial_assign, axis=1)
    else:
        batch_size = tf.shape(candidates)[0]
        partial_assign = tf.fill([batch_size, 0, nexisting_value], 0)
        partial_assign = tf.cast(partial_assign, tf.int32)

    partial_assign = cart_prod_assign(partial_assign, candidates)

    # the partial_primal api requires the partial_assign
    # to have shape None x (KxM)
    npartial_assign = [
        partial_assign[:, k, :] for k, _t in enumerate(partial_idx)
    ]
    partial_assign = npartial_assign

    # adjust the order of partial assignments
    passign_reorder = partial_assign[0:offsets]
    pidx_reorder = partial_idx[0:offsets]
    baseidx = offsets
    for idx_, nid_ in enumerate(nodes_list):
        if nid_ < c_node_idx:
            passign_reorder.append(partial_assign[baseidx])
            pidx_reorder.append(partial_idx[baseidx])
            baseidx += 1
        elif nid_ == c_node_idx:
            passign_reorder.append(partial_assign[-1])
            pidx_reorder.append(partial_idx[-1])
    return passign_reorder, pidx_reorder


def update_value(partial_primal, hops, partial_assign, partial_idx, beliefs,
                 msgs, lmsg, ncandidates, vsize):
    hvalue = partial_primal(hops, partial_assign, partial_idx, beliefs, msgs)

    # lvalue = tf.tile(tf.expand_dims(lmsg, 1),
    #                  [1, ncandidates, 1])
    # lvalue = tf.reshape(lvalue, [-1, ncandidates * vsize])
    return hvalue


def null_candidiates(batch_size, size, ncandidates):
    res = tf.fill([batch_size, 0, size, ncandidates], 0)
    return tf.cast(res, tf.int32)


def batch_gather_int64(tensor, indices):
    return tf.cast(
        tf.batch_gather(tf.cast(tensor, tf.int64), indices), tf.int32)


def shrink_candidates(current_state, current_action, prev_state, pcassigned,
                      sizesnodes, sizeanodes, cassigned, cvalue, shk_sz,
                      acandidates, vsize, step):
    # get top_k assignments
    ncvalue, nassign_idx = tf.nn.top_k(cvalue, shk_sz)  # shrink

    # shape of nassign_idx None x BeamWidth
    prev_idx = tf.div(nassign_idx, vsize, name='compute_prev_idx')
    prev_idx_offset = -tf.multiply(
        prev_idx, vsize, name='compute_prev_idx_off')
    next_idx = tf.add(nassign_idx, prev_idx_offset, name='compute_next_idx')

    prev_idx = tf.cast(prev_idx, tf.int32)
    next_idx = tf.cast(next_idx, tf.int32)

    prev_idx_gather_s = tf.tile(
        tf.expand_dims(tf.expand_dims(prev_idx, 1), 1),
        [1, step, len(sizesnodes), 1])
    prev_idx_gather_a = tf.tile(
        tf.expand_dims(tf.expand_dims(prev_idx, 1), 1),
        [1, step, len(sizeanodes), 1])
    # update candidates
    batch_size = tf.shape(current_state)[0]
    current_state = tf.cond(
        tf.equal(step, 0),
        lambda: null_candidiates(batch_size, len(sizesnodes), shk_sz),
        lambda: batch_gather_int64(current_state, prev_idx_gather_s))
    current_action = tf.cond(
        tf.equal(step, 0),
        lambda: null_candidiates(batch_size, len(sizeanodes), shk_sz),
        lambda: batch_gather_int64(current_action, prev_idx_gather_a))

    prev_state = batch_gather_int64(
        prev_state,
        tf.tile(tf.expand_dims(prev_idx, 1), [1, len(sizesnodes), 1]))

    nassigned = []
    for ca in cassigned:
        nassigned.append(
            batch_gather_int64(ca, tf.expand_dims(prev_idx, axis=1)))
    nassigned.append(
        batch_gather_int64(acandidates, tf.expand_dims(next_idx, axis=1)))
    npcassigned = []
    for ca in pcassigned:
        npcassigned.append(
            batch_gather_int64(ca, tf.expand_dims(prev_idx, axis=1)))

    return current_state, current_action, prev_state, npcassigned, nassigned, ncvalue, nassign_idx, prev_idx


def expand_cvalue(cvalue, vsize, beam_width, local_belief):
    exp_sz = tf.shape(cvalue)[1] * vsize  # expansion size
    shk_sz = tf.cond(
        tf.less(beam_width, exp_sz), lambda: beam_width,
        lambda: exp_sz)  # shrinking size
    cvalue = tf.expand_dims(cvalue, axis=2)
    cavalue = tf.expand_dims(local_belief, 1)
    cvalue = tf.add(cvalue, cavalue, name='expand_value')
    cvalue = tf.reshape(cvalue, [-1, exp_sz])

    return cvalue, shk_sz


def padding_assignment(assignments):
    """
    """
    batch_size = tf.shape(assignments[0][0])[0]
    batch_idx = [
        tf.tile(tf.constant([i], dtype=tf.int32), [batch_size])
        for i, _ in enumerate(assignments)
    ]
    return [[bidx] + assign for bidx, assign in zip(batch_idx, assignments)]


if __name__ == '__main__':
    assignment_cand = tf.constant([[1, 3, 4], [2, 3, 5], [4, 3, 0]],
                                  dtype=tf.int32)
    assignment_expa = tf.constant([[3, 4], [4, 5], [2, 3], [6, 7]],
                                  dtype=tf.int32)

    res = cart_prod(assignment_cand, assignment_expa)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(res))
