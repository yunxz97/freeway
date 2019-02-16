import argparse
import time
import threading
import tensorflow as tf
from preprocessors.freeway_env import FreewayEnvironment


import numpy as np

import ac_net_freeway as ac_net
import worker_freeway as worker

import constants


PARSER = argparse.ArgumentParser(description=None)
PARSER.add_argument(
    '-d', '--device', default=constants.DEVICE, type=str, help='choose device: cpu/gpu')
PARSER.add_argument('-ngpu', '--NUM_GPU', default=constants.N_GPU, type=int, help='Number of available GPUs')
PARSER.add_argument(
    '-e', '--episodes', default=constants.EPISODES, type=int, help='number of episodes')
PARSER.add_argument(
    '-w', '--workers', default=constants.WORKERS, type=int, help='number of workers')
PARSER.add_argument(
    '-l', '--log_dir', default=constants.LOG_DIR, type=str, help='log directory')
PARSER.add_argument('-t', '--T_MAX', default=constants.T_MAX, type=int, help='Buffer Size')
PARSER.add_argument(
    '-m',
    '--MULT_FAC',
    default=constants.MULTI_FAC,
    type=float,
    help='Reward multiplication factor')
PARSER.add_argument(
    '-rl_lr',
    '--RL_LEARN_RATE',
    default=constants.RL_LR,
    type=float,
    help='Learning rate for reinforcement learning')
PARSER.add_argument(
    '-sl_lr',
    '--SL_LEARN_RATE',
    default=constants.SL_LR,
    type=float,
    help='Learning rate for supervised learning')
PARSER.add_argument(
    '-s', '--SIM_STEPS', default=constants.SIM_STEPS, type=int, help='Simulation steps')
PARSER.add_argument(
    '-b',
    '--BP_STEPS',
    default=constants.BP_STEPS,
    type=int,
    help='Belief Propagation iterations')
PARSER.add_argument(
    '-beta', '--BETA', default=constants.BETA, type=float, help='Entropy Loss Fraction')
PARSER.add_argument(
    '-gamma', '--GAMMA', default=constants.GAMMA, type=float, help='Discount Factor')
PARSER.add_argument(
    '-uid',
    '--UNIQUE_ID',
    default=constants.UNIQUE_ID,
    type=float,
    help='ID to differentiate runs')
PARSER.add_argument(
    '-lm', '--LOAD_MODEL', default=constants.LOAD_MODEL, type=bool, help='To load new model')
PARSER.add_argument(
    '-seq',
    '--Sequential',
    default=constants.SEQUENTIAL,
    type=bool,
    help='To use sequential BP')

model_path = "./model"
ARGS = PARSER.parse_args()
print(ARGS)

DEVICE = ARGS.device
NUM_GPU = ARGS.NUM_GPU
LEARNING_RATE_RL = ARGS.RL_LEARN_RATE
LEARNING_RATE_SL = ARGS.SL_LEARN_RATE

GAMMA = ARGS.GAMMA

T_MAX = ARGS.T_MAX
BETA = ARGS.BETA

# NUM_WORKERS = multiprocessing.cpu_count()
NUM_WORKERS = ARGS.workers
NUM_EPISODES = ARGS.episodes
LOG_DIR = ARGS.log_dir

MULT_FAC = ARGS.MULT_FAC
SIM_STEPS = ARGS.SIM_STEPS
BP_STEPS = ARGS.BP_STEPS

load_model = ARGS.LOAD_MODEL
ENV_ARGS = {"gamma": GAMMA}

ISSeq = ARGS.Sequential


def main():
    tf.reset_default_graph()

    history = []

    # dummy_env = FreewayEnvironment(args=ENV_ARGS, BP=True)
    # agent_args = populate_args(dummy_env)
    with tf.device('/{}:{}'.format(DEVICE, max(NUM_GPU - 1, 0))):
        global_model = ac_net.ACNetFreeway(
            LEARNING_RATE_RL,
            LEARNING_RATE_SL,
            SIM_STEPS=SIM_STEPS,
            BP_STEPS=BP_STEPS,
            MULT_FAC=MULT_FAC,
            # env_args=agent_args,
            BETA=BETA,
            sequential=ISSeq,
            name='global')

    workers = []

    if DEVICE == "cpu":
        device_id = [0] * NUM_WORKERS
    else:
        device_id = [i % NUM_GPU for i in range(NUM_WORKERS)]

    for i in range(NUM_WORKERS):
        # env = FreewayEnvironment(args=ENV_ARGS, BP=True)
        env = FreewayEnvironment()
        # agent_args = populate_args(env)
        # env._max_episode_steps = 3000
        workers.append(
            worker.Worker(
                env,
                worker_name='worker_{}'.format(i),
                global_name='global',
                rl_lr=LEARNING_RATE_RL,
                supervised_lr=LEARNING_RATE_SL,
                gamma=GAMMA,
                t_max=T_MAX,
                sess=None,
                history=history,
                BETA=BETA,
                SIM_STEPS=SIM_STEPS,
                BP_STEPS=BP_STEPS,
                model_path=model_path,
                MULT_FAC=MULT_FAC,
                sequential=ISSeq,
                logdir=LOG_DIR,
                devs='/{}:{}'.format(DEVICE, device_id[i])))
    saver = tf.train.Saver(max_to_keep=5)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    for w in workers:
        w.set_sess(sess)


    if load_model:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        print(ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    print("Before Training")
    # with tf.variable_scope("global/sl_params", reuse=True):
    #     w = tf.get_variable("reward_transition")
    #     print(w.eval(session=sess))
    #
    # with tf.variable_scope("global/rl_params", reuse=True):
    #     w = tf.get_variable("RL_RewardFactor")
    #     print(w.eval(session=sess))

    thread_list = []
    for workeri in workers:

        def worker_work():
            return workeri.work(n_episodes=NUM_EPISODES, saver=saver)

        thread = threading.Thread(target=worker_work)
        thread.start()
        thread_list.append(thread)

    for threads in thread_list:
        threads.join()

    print("After Training")
    # with tf.variable_scope("global/sl_params", reuse=True):
    #     w = tf.get_variable("reward_transition")
    #     print(w.eval(session=sess))
    #
    # with tf.variable_scope("global/rl_params", reuse=True):
    #     w = tf.get_variable("RL_RewardFactor")
    #     print(w.eval(session=sess))


if __name__ == "__main__":
    main()
