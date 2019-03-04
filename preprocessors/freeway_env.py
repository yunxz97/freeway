import numpy as np
import gym
from preprocessors.base import DDNBasePreprocessor
from constants import SCREEN_HEIGHT, SCREEN_WIDTH, VIDEO_DIR, ENV_MAX_STEPS, GAMMA, \
    REWARD_SHAPING, N_ACTION_REPEAT, SKIP_LANE
import os
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from datetime import datetime
from re import sub


class FreewayEnvironment:
    steps, max_steps = 0, ENV_MAX_STEPS
    gamma = GAMMA

    def __init__(self, args = {}, env="Freeway-v0"):
        self.artificial_reward = np.arange(2, -2, -4 / SCREEN_HEIGHT)[-SCREEN_HEIGHT:]
        # print(self.artificial_reward)
        self.max_steps = ENV_MAX_STEPS
        self.gamma = GAMMA
        self.specs = {
            'obs': [[SCREEN_HEIGHT,
                     SCREEN_WIDTH, SCREEN_WIDTH, SCREEN_WIDTH, SCREEN_WIDTH, SCREEN_WIDTH,
                     SCREEN_WIDTH, SCREEN_WIDTH, SCREEN_WIDTH, SCREEN_WIDTH, SCREEN_WIDTH,
                     2,   2,   2,   2,   2,
                     2,   2,   2,   2,   2,
                     2]],
            'act': [3]}
        self.env = gym.make(env)
        self.extractor = DDNBasePreprocessor()
        self.R = 0
        os.makedirs(VIDEO_DIR, exist_ok=True)

    def obs_spec(self):
        return self.specs['obs']

    def act_specs(self):
        return self.specs['act']

    def start(self):
        return

    def stop(self):
        return

    def reset(self):
        print('resetting...')
        self.R = 0
        self.steps = 0
        obs = self.env.reset()
        obs = self.extractor.get_obs(obs)
        id = sub('[-: ]', '', str(datetime.today()).split('.')[0])
        video_path = os.path.join(VIDEO_DIR, id + '.mp4')
        self.video_recorder = VideoRecorder(self.env, video_path)
        return obs, 0, 0

    def step(self, action):
        self.video_recorder.capture_frame()
        for _ in range(N_ACTION_REPEAT):
            obs, r, done, info = self.env.step(action)
            if done:
                break
        # if r != 0:
        #     print('Success!')
        self.R += r * np.power(GAMMA, self.steps)
        if self.steps == self.max_steps:
            done = True
        obs = self.extractor.get_obs(obs)

        if REWARD_SHAPING:
            if r > 0:
                r = 100
            else:
                r = self.artificial_reward[obs[0]]

        self.steps += 1
        if done:
            print('total return: ', self.R)
            print(f"Saving video to {VIDEO_DIR}")
            self.video_recorder.close()
            # self.video_recorder.enabled = False
            print(f"Video saved")
        # self.env.render()

        return obs, r, done

    def get_info(self):
        return self.steps, self.max_steps


if __name__ == "__main__":
    env = FreewayEnvironment(env="Freeway-v0")