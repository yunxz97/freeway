from collections import namedtuple
from preprocessors.base import DDNBasePreprocessor
import numpy as np
import tensorflow as tf
from lib.BasicInferUnit import InferNetPipeLine
import ac_net_freeway as ac_net
import tf_utils
from MultiEnv import MultiEnv
from copy import deepcopy
import constants

MAX_STEPS = 10000

Step = namedtuple('Step', 'cur_step action next_step reward done')

ARGS = {}
SIM_STEPS = constants.SIM_STEPS
BP_STEPS = constants.BP_STEPS
MULT_FAC = constants.MULTI_FAC
NUM_MULTI_ENV = constants.NUM_MULTI_ENV


class Worker(object):
    def __init__(self,
                 env,
                 worker_name,
                 global_name,
                 rl_lr,
                 supervised_lr,
                 gamma,
                 t_max,
                 sess,
                 model_path,
                 history,
                 BETA,
                 agent_args=ARGS,
                 SIM_STEPS=SIM_STEPS,
                 BP_STEPS=BP_STEPS,
                 MULT_FAC=MULT_FAC,
                 num_env = NUM_MULTI_ENV,
                 sequential=False,
                 logdir='logs',
                 devs='/device:GPU:0'):
        self.num_env = num_env

        if constants.USE_MULTIENV:
            self.env = MultiEnv(env, self.num_env)
        else:
            self.env = env

        self.name = worker_name
        self.gamma = gamma
        self.sess = sess
        self.t_max = t_max
        self.history = history
        self.model_path = model_path

        with tf.device(devs):
            self.local_model = ac_net.ACNetFreeway(
                rl_lr,
                supervised_lr,
                worker_name,
                # env_args=agent_args,
                SIM_STEPS=SIM_STEPS,
                BP_STEPS=BP_STEPS,
                MULT_FAC=MULT_FAC,
                BETA=BETA,
                sequential=sequential,
                global_name=global_name)
        self.copy_to_local_op = tf_utils.update_target_graph(
            global_name, worker_name)

        self.summary_writer = tf.summary.FileWriter("{}/train_{}".format(
            logdir, worker_name))

    def set_sess(self, sess):
        self.sess = sess

    def _copy_to_local(self):
        self.sess.run(self.copy_to_local_op)

    def work(self, n_episodes, saver):
        episode_i = 0
        episode_len = [0 for _ in range(self.num_env)]
        cur_state, _, _ = deepcopy(self.env.reset())
        if not constants.USE_MULTIENV:
            cur_state = [[cur_state]]
        count = 1
        cum_reward = [0 for _ in range(self.num_env)]
        while episode_i < n_episodes:
            # 1) sync from global model to local model
            self._copy_to_local()

            # 2) collect t_max steps (if terminated then i++)
            steps_batch = [[] for _ in range(self.num_env)]
            step_env_map = [i for i in range(self.num_env)]
            step_env_inv_map = [i for i in range(self.num_env)]
            done_env = [False for _ in range(self.num_env)]
            info_batch = [[] for _ in range(self.num_env)]

            for _ in range(self.t_max):
                print('state     :', cur_state[0][0].tolist())
                action = self.local_model.get_action(
                    [self.process_observation(state) for state in cur_state], self.sess)
                # action = self.local_model.get_action(cur_state, self.sess)
                next_state, reward, done = deepcopy(self.env.step(np.argmax(action, axis=1)))
                if not constants.USE_MULTIENV:
                    next_state = [[next_state]]
                    reward = [reward]
                    done = [done]

                # reward = [r*MULT_FAC for r in reward] #REMOVE for non-IPPC envs
                info = self.env.get_info()

                is_reset_needed = any(done)
                for agent_id in range(self.num_env) :
                    # TODO WARNING:: Hack for shortening the horizon of IPPC NOT needed for other domains USE ONLY ELSE PART FOR OTHER ENVS

                    # if done[agent_id] and reward[agent_id] != 0 and info[agent_id][0] != info[agent_id][1]:
                    #     # Done but not at goal position or at the horizon --> agent gone invisible
                    #     steps_remaining = info[agent_id][1] - info[agent_id][0]
                    #     reward_for_cumulative = -1 * (1 - np.power(self.gamma, steps_remaining + 1)) / (1 - self.gamma) * MULT_FAC
                    # else:
                    #     reward_for_cumulative = reward[agent_id]
                    reward_for_cumulative = reward[agent_id]
                    done_env[step_env_map[agent_id]] = done[agent_id]
                    info_batch[step_env_map[agent_id]] = info[agent_id]

                    cum_reward[agent_id] += np.power(self.gamma, episode_len[agent_id]) * reward_for_cumulative
                    episode_len[agent_id] = episode_len[agent_id] + 1

                    steps_batch[step_env_map[agent_id]].append(Step(cur_step=cur_state[agent_id],
                                                              action=action[agent_id],
                                                              next_step=next_state[agent_id],
                                                              reward=reward[agent_id],
                                                              done=done[agent_id]))
                    if episode_len[agent_id] >= MAX_STEPS:
                        is_reset_needed = True

                if is_reset_needed:
                    agents_to_reset = []
                    for agent_id in range(self.num_env) :
                        if episode_len[agent_id] >= MAX_STEPS or done[agent_id] :
                            agents_to_reset.append(agent_id)
                    cur_state, _, _ = deepcopy(self.env.reset_partial(agents_to_reset))
                    for agent_id in agents_to_reset :
                        self.history.append(episode_len[agent_id])
                        summary = tf.Summary()
                        summary.value.add(tag='Perf/episode_len',simple_value=float(episode_len[agent_id]))
                        summary.value.add(tag='Perf/episode_reward',simple_value=float(cum_reward[agent_id]))
                        self.summary_writer.add_summary(summary, episode_i)
                        print('worker {}: episode {} finished in {} steps, cumulative reward: {}'.format(self.name,
                                                                                                         episode_i,
                                                                                                         episode_len[agent_id],
                                                                                                         cum_reward[agent_id]))

                        if episode_i % 10 == 0 and episode_i != 0:
                            saver.save(self.sess, self.model_path + '/model-' +str(episode_i) + '.cptk')
                            print("Saved Model")
                        steps_batch.append([])
                        info_batch.append([])
                        cum_reward[agent_id] = 0
                        episode_len[agent_id] = 0
                        done_env.append(False)
                        step_env_map[agent_id] = len(steps_batch)-1
                        step_env_inv_map.append(agent_id)
                        episode_i = episode_i + 1
                else :
                    cur_state = next_state

            R_batch_consolidated = []
            cur_state_batch_consolidated = []
            next_state_batch_consolidated = []
            action_batch_consolidated = []
            advantage_batch_consolidated = []
            reward_batch_consolidated = []

            # 3) convert the t_max steps into a batch
            # for idx, steps in enumerate(steps_batch):
            #     if not steps :
            #         continue
            #     if steps[-1].done:
            #         R = 0
            #     else:
            #         R = self.local_model.predict_value(self.process_observation(cur_state[step_env_inv_map[idx]]), self.sess)
            #         # R = self.local_model.predict_value(cur_state[step_env_inv_map[idx]], self.sess)
            #     R_batch = [0 for _ in range(len(steps))]
            #     for i in reversed(range(len(steps))):
            #         #TODO WARNING:: Hack for shortening the horizon of IPPC NOT needed for other domains USE ONLY ELSE PART FOR OTHER ENVS
            #         # if i == len(steps) - 1 and \
            #         #         steps[-1].done and \
            #         #         steps[-1].reward !=0 and \
            #         #         info_batch[idx][0] != info_batch[idx][1]:
            #         #     #Done but not at goal position or at the horizon --> agent gone invisible
            #         #     steps_remaining = info_batch[idx][1] - info_batch[idx][0]
            #         #     reward = -1 * (1 - np.power(self.gamma, steps_remaining + 1)) / (1 - self.gamma) * MULT_FAC
            #         #     R = reward  # as done, no need to add gamma*R
            #         # else:
            #         #     step = steps[i]
            #         #     R = step.reward + self.gamma * R
            #         step = steps[i]
            #         R = step.reward + self.gamma * R
            #
            #         R_batch[i] = R
            #     cur_state_batch = [np.reshape(self.process_observation(step.cur_step), [-1])for step in steps]
            #     next_state_batch = [np.reshape(self.process_observation(step.next_step), [-1])for step in steps]
            #     # cur_state_batch = [np.reshape(step.cur_step, [-1])for step in steps]
            #     # next_state_batch = [np.reshape(step.next_step, [-1])for step in steps]
            #     reward_batch = [step.reward for step in steps]
            #     pred_v_batch = self.local_model.predict_value(cur_state_batch, self.sess)
            #     action_batch = [step.action for step in steps]
            #
            #     advantage_batch = list(np.reshape([R_batch[i] - pred_v_batch[i] for i in range(len(steps))], [-1]))
            #     R_batch = list(np.reshape(R_batch, [-1]))
            #     # 4) compute the gradient and update the global model
            #     # action_batch = np.reshape(action_batch, [-1])
            #     R_batch_consolidated += R_batch
            #     cur_state_batch_consolidated += cur_state_batch
            #     next_state_batch_consolidated += next_state_batch
            #     action_batch_consolidated += action_batch
            #     advantage_batch_consolidated += advantage_batch
            #     reward_batch_consolidated += reward_batch
            #
            # advantage_batch_consolidated = np.reshape(advantage_batch_consolidated, [-1])
            # R_batch_consolidated = np.reshape(R_batch_consolidated, [-1])
            #
            # feed_dict = {
            #     self.local_model.input_s: cur_state_batch_consolidated,
            #     self.local_model.input_sprime: next_state_batch_consolidated,
            #     self.local_model.input_a: action_batch_consolidated,
            #     self.local_model.advantage: advantage_batch_consolidated,
            #     self.local_model.target_v: R_batch_consolidated,
            #     self.local_model.agent.init_state_pl: cur_state_batch_consolidated,
            #     self.local_model.input_r: reward_batch_consolidated
            # }
            #
            # batch_size = len(cur_state_batch_consolidated)
            #
            # if (type(self.local_model.agent.infer_net) == InferNetPipeLine):
            #     feed_dict[self.local_model.agent.infer_net.init_action] = np.zeros(
            #         [len(action_batch_consolidated), 3, np.sum(self.local_model.agent.action_dim)])
            #     zero_placeholder = np.zeros([batch_size, 2, np.sum(self.local_model.agent.all_state_dim)])
            #     cur_state_batch_consolidated = np.expand_dims(cur_state_batch_consolidated, axis=1)
            #     feed_dict[self.local_model.agent.init_state_pl] = np.concatenate(
            #         [zero_placeholder, cur_state_batch_consolidated], axis=1)
            #     # cur_state_batch_consolidated = np.repeat(
            #     #     np.expand_dims(cur_state_batch_consolidated, axis=1), 3, axis=1)
            #     # feed_dict[self.local_model.agent.init_state_pl] = cur_state_batch_consolidated
            #
            # v_l, p_l, e_l, rl_loss, sl_loss, \
            # _, _, _, _, \
            # v_n_rl, v_n_sl, v_n_total, \
            # rew_loss, trans_loss, sn, nsn, an, rf = self.sess.run(
            # # v_l, p_l, e_l, rl_loss, sl_loss, \
            # # v_n_rl, v_n_sl, v_n_total, \
            # # rew_loss, trans_loss, sn, nsn, an = self.sess.run(
            #     [
            #         self.local_model.value_loss, self.local_model.policy_loss,
            #         self.local_model.entropy_loss, self.local_model.rl_loss,
            #         self.local_model.supervised_loss,
            #         self.local_model.gradients_rl,
            #         self.local_model.apply_gradients_rl,
            #         self.local_model.gradients_sl,
            #         self.local_model.apply_gradients_sl,
            #         self.local_model.var_norms_rl,
            #         self.local_model.var_norms_sl,
            #         self.local_model.var_norms_total,
            #         self.local_model.reward_loss,
            #         self.local_model.transition_loss,
            #         self.local_model.state_nodes,
            #         self.local_model.next_state_nodes,
            #         self.local_model.action_nodes,
            #         self.local_model.reward_factor_value
            #     ], feed_dict)
            #
            # if is_reset_needed:
            #     print('reward factor:', rf)
            #
            # mean_reward = np.mean(reward_batch_consolidated)
            # mean_value = np.mean(R_batch_consolidated)
            #
            # summary = tf.Summary()
            # summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
            # summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
            # summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
            # summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
            # summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
            #
            # summary.value.add(tag='Losses/Supervised Loss', simple_value=float(sl_loss))
            # summary.value.add(tag='Losses/RL Loss', simple_value=float(rl_loss))
            #
            # summary.value.add(tag='Losses/Var Norm RL', simple_value=float(v_n_rl))
            # summary.value.add(tag='Losses/Var Norm SL', simple_value=float(v_n_rl))
            # summary.value.add(tag='Losses/Var Norm Total', simple_value=float(v_n_rl))
            #
            # self.summary_writer.add_summary(summary, count)
            # count += 1
        self.env.stop()

    def process_observation(self, cur_state):
        # return self.local_model.agent.UpdatePositionVisibility(cur_state[0], cur_state[1])
        return DDNBasePreprocessor.obs_to_state(cur_state[0].astype(np.int32))
