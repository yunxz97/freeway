import random
from collections import deque, namedtuple

import numpy as np
import tensorflow as tf

import ac_net_freeway as ac_net
import gym
import tf_utils

import sys
import os

from gym.wrappers.monitoring.video_recorder import VideoRecorder

if len(sys.argv) > 1:
    MODEL_TYPE = sys.argv[1]
else:
    MODEL_TYPE = "Base" # ["Base", "Decomposed"]
print("Using model type: {}".format(MODEL_TYPE))

if MODEL_TYPE == "Base":
    from preprocessors.base import DDNBasePreprocessor as Processor
    from agents.base import FreewayBaseAgent as Agent
    from agents.base import variable_range
elif MODEL_TYPE == "Decomposed":
    from preprocessors.decomposed import DDNDecomposedPreprocessor as Processor
    from agents.decomposed import FreewayDecomposedAgent as Agent
    from agents.decomposed import variable_range
else:
    raise "Model Type Not Implemented"

MAX_STEPS_PER_EPISODE = 1000
Step = namedtuple('Step', 'cur_step action next_step reward done')

preprocessor_for_ddn = Processor()

BASE_VIDEO_PATH = 'videos'
os.makedirs(BASE_VIDEO_PATH, exist_ok=True)


class Worker:
    def __init__(
            self, env, state_size, action_size,
            worker_name, global_name, lr, gamma, t_max, sess,
            model_path, logdir,
            history, BETA,
            agent_args,
            SIM_STEPS, BP_STEPS,
            MULT_FAC, sequential=False, ):

        self.env = env
        self.name = worker_name
        self.gamma = gamma
        self.sess = sess
        self.t_max = t_max
        self.history = history
        self.model_path = model_path

        self.local_model = ac_net.ACNetFreeway(
            Agent,
            state_size, action_size, lr,
            name=worker_name,
            env_args=agent_args,
            SIM_STEPS=SIM_STEPS,
            BP_STEPS=BP_STEPS,
            MULT_FAC=MULT_FAC,
            BETA=BETA,
            sequential=sequential,
            global_name=global_name
        )
        self.copy_to_local_op = tf_utils.update_target_graph(global_name, worker_name)
        self.summary_writer = tf.summary.FileWriter("{}/train_{}".format(logdir, worker_name))

    def work(self):
        self.summary_writer.add_graph(self.sess.graph)
        n_episodes = 1000

        episode_i = 0
        episode_len = 0
        cur_state = preprocessor_for_ddn.obs_to_state(self.env.reset())
        count = 1
        cum_reward = 0
        # start_life = 5
        # need_restart = True

        while episode_i < n_episodes:
            # setup video recorder
            video_path = os.path.join(BASE_VIDEO_PATH, f"{episode_i}.mp4")
            video_recorder = VideoRecorder(self.env, video_path, enabled=video_path is not None)

            # 1) sync from global model to local model
            # self._copy_to_local()

            # 2) collect t_max steps (if terminated then i++)
            steps = []
            # print(self.local_model.predict_policy(cur_state, self.sess), int(np.argmax(cur_state))% (6), int(np.argmax(cur_state)/(6)))
            for _ in range(self.t_max):
                # if need_restart:
                #     action = 0
                #     need_restart = False
                #     print('start using life: ' + str(start_life))
                # else:
                action = self.local_model.get_action(cur_state, self.sess)
                    # print(action)

                next_state, reward, done, info = self.env.step(action)
                next_state = preprocessor_for_ddn.obs_to_state(next_state)

                # if start_life > info['ale.lives']:
                #     need_restart = True
                #     start_life = info['ale.lives']

                # capture video
                video_recorder.capture_frame()

                # reward *= MULT_FAC
                cum_reward += np.power(self.gamma, episode_len) * reward

                if reward != 0:
                    print('cum_reward: ' + str(cum_reward))
                # print(episode_len)

                episode_len = episode_len + 1
                # steps.append(
                #     Step(
                #         cur_step=cur_state,
                #         action=action,
                #         next_step=next_state,
                #         reward=reward,
                #         done=done
                #     )
                # )
                if done or episode_len >= MAX_STEPS_PER_EPISODE:
                    self.history.append(episode_len)
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/episode_len', simple_value=float(episode_len))
                    summary.value.add(tag='Perf/episode_reward', simple_value=float(cum_reward))
                    self.summary_writer.add_summary(summary, episode_i)
                    print(episode_i)
                    print(summary)
                    print(
                        'worker {}: episode {} finished in {} steps, cumulative reward: {}'.format(self.name, episode_i,
                                                                                                   episode_len,
                                                                                                   cum_reward))
                    # print(action)
                    if episode_i % 100 == 0 and episode_i != 0:
                        saver.save(self.sess, self.model_path + '/model-' + str(episode_i) + '.cptk')
                        print("Saved Model")
                    cum_reward = 0
                    episode_i = episode_i + 1
                    episode_len = 0
                    # start_life = 5
                    # need_restart = True
                    cur_state = preprocessor_for_ddn.obs_to_state(self.env.reset())
                    break
                cur_state = next_state

            # save video
            print(f"Saving video to {video_path}")
            video_recorder.close()
            video_recorder.enabled = False
            print(f"Video saved")

            # 3) convert the t_max steps into a batch
            # if steps[-1].done:
            #     R = 0
            # else:
            #     R = self.local_model.predict_value(cur_state, self.sess)
            # R_batch = np.zeros(len(steps))
            # advantage_batch = np.zeros(len(steps))
            # target_v_batch = np.zeros(len(steps))
            # for i in reversed(range(len(steps))):
            #     step = steps[i]
            #     R = step.reward + self.gamma * R
            #     R_batch[i] = R
            # cur_state_batch = [step.cur_step for step in steps]
            # pred_v_batch = self.local_model.predict_value(cur_state_batch, self.sess)
            # action_batch = [step.action for step in steps]
            # advantage_batch = [R_batch[i] - pred_v_batch[i] for i in range(len(steps))]
            # # 4) compute the gradient and update the global model
            # action_batch = np.reshape(action_batch, [-1])
            # advantage_batch = np.reshape(advantage_batch, [-1])
            # R_batch = np.reshape(R_batch, [-1])
            # feed_dict = {
            #     self.local_model.input_s: cur_state_batch,
            #     self.local_model.input_a: action_batch,
            #     self.local_model.advantage: advantage_batch,
            #     self.local_model.target_v: R_batch,
            # }

            # if type(self.local_model.agent.infer_net) == InferNetPipeLine:
            #     feed_dict[self.local_model.agent.infer_net.simulate_steps] = self.local_model.SIM_STEPS
            #     feed_dict[self.local_model.agent.infer_net.max_steps] = self.local_model.SIM_STEPS + (self.local_model.BP_STEPS - 1) * 2

            # v_l, p_l, e_l, loss, _, _, v_n = self.sess.run(
            #     [
            #         self.local_model.value_loss,
            #         self.local_model.policy_loss,
            #         self.local_model.entropy_loss,
            #         self.local_model.loss,
            #         self.local_model.gradients,
            #         self.local_model.apply_gradients,
            #         self.local_model.var_norms
            #     ],
            #     feed_dict,
            # )

            # mean_reward = np.mean([step.reward for step in steps])
            # mean_value = np.mean(R_batch)

            # if count % 1 == 0:
            #     summary = tf.Summary()
            #     summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
            #     summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
            #     summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
            #     summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
            #     summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
            #     summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
            #     self.summary_writer.add_summary(summary, count)
            # # print(summary)
            # print(count)
            # count += 1


all_state_dim = list(variable_range.values())

STATE_SIZE = sum(all_state_dim)
ACTION_SIZE = 3

LEARNING_RATE = 0.00025
GAMMA = .99  # 0.99
T_MAX = MAX_STEPS_PER_EPISODE
BETA = 0.01  # ARGS.BETA
MULT_FAC = 3  # set some random value here, not sure what is the approriate value
SIM_STEPS = 10
BP_STEPS = 50

# load_model = ARGS.LOAD_MODEL
# ENV_ARGS = {"gamma": GAMMA}

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
# from tensorflow.python import debug as tf_debug
# sess = tf_debug.TensorBoardDebugWrapperSession(sess, "Yuyangs-MacBook-Pro.local:6064")

worker = Worker(
    env=gym.make("FreewayDeterministic-v4"),
    state_size=STATE_SIZE, action_size=ACTION_SIZE,
    lr=LEARNING_RATE,
    gamma=GAMMA, t_max=T_MAX, sess=sess,
    history=[],
    BETA=BETA,
    agent_args={},
    SIM_STEPS=SIM_STEPS,
    BP_STEPS=BP_STEPS,
    model_path="./model", logdir='logs',
    global_name='global',
    worker_name='the_only_worker',
    MULT_FAC=MULT_FAC,
    sequential=True,
)
sess.run(tf.global_variables_initializer())
print('initialization finishes')
worker.work()
