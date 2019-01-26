import platform
import os
import tensorflow as tf
from tensorflow.python.framework import ops


uname = platform.uname().system
MRFInferGPU_path = os.path.dirname(os.path.realpath(__file__))
if uname == 'Windows':
    MRFInferGPU_path = os.path.join(
        MRFInferGPU_path, '../MRFInferGPU/build/Release/MRFInference.dll')
else:
    MRFInferGPU_path = os.path.join(
        MRFInferGPU_path, '../MRFInferGPU/build/libMRFInference.so')
MRFInferModule = tf.load_op_library(MRFInferGPU_path)


@ops.RegisterGradient("MsgGather")
def _msggather_grad(op, grad):
    hop = op.inputs[0]
    ndims = len(hop.shape) - 1
    lops = [op.inputs[idx + 1] for idx in range(ndims)]
    msgs = [op.inputs[idx + ndims + 1] for idx in range(ndims)]
    hop_grad, lops_grads, msgs_grads = MRFInferModule.grad_gather_msg(
        grad, hop, lops, msgs)
    res = [hop_grad] + lops_grads + msgs_grads
    return res


@ops.RegisterGradient("UpdMsg")
def _updmsg_grad(op, *grad):
    return MRFInferModule.grad_upd_msg(op.inputs[0], grad)


@ops.RegisterGradient("UpdMsgBelief")
def _updmsg_belief_grad(op, grad_nbeliefs, grad_nmsgs):
    return MRFInferModule.grad_upd_msg_belief(grad_nbeliefs,
                                              grad_nmsgs,
                                              op.inputs[0],
                                              op.inputs[1])


@ops.RegisterGradient("MsgGatherShift2D")
def _msg_gather_shift2d_grad(op, *grad):

    lop_next_avar_grad = grad[0]
    lop_prev_avar_grad = grad[1]

    offsets = op.inputs[0]
    lop_prev = op.inputs[1]
    lop_avar = op.inputs[2]
    lop_next = op.inputs[3]

    lop_prev_grad, lop_avar_grad, lop_next_grad = \
        MRFInferModule.grad_msg_gather_shift2d(lop_next_avar_grad,
                                               lop_prev_avar_grad,
                                               lop_prev,
                                               lop_avar,
                                               lop_next,
                                               offsets)
    return None, lop_prev_grad, lop_avar_grad, lop_next_grad


@ops.RegisterGradient("AddMsg")
def _grad_add_msg(op, grad):
    lop_grad, msg_grad = MRFInferModule.grad_add_msg(grad, op.inputs[0])
    return lop_grad, msg_grad
