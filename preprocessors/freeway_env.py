import numpy as np
import gym
from preprocessors.base import DDNBasePreprocessor
from constants import SCREEN_HEIGHT, SCREEN_WIDTH, ENV_MAX_STEPS, GAMMA, VIDEO_DIR, ENV_MAX_STEPS, GAMMA
import os
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from datetime import datetime
from re import sub


class FreewayEnvironment:
    steps, max_steps = 0, ENV_MAX_STEPS
    gamma = GAMMA

    def __init__(self, args = {}, env="Freeway-v0"):
        self.artificial_reward = np.zeros((SCREEN_HEIGHT, ))
        reward_step = SCREEN_HEIGHT // 10
        self.artificial_reward[np.arange(SCREEN_HEIGHT-1, -1, -reward_step)] = np.arange(11)
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
        if not self.specs:
            self.make_specs()
        return self.specs['obs']

    def act_specs(self):
        if not self.specs:
            self.make_specs()
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
        return obs

    def step(self, action):
        self.video_recorder.capture_frame()
        obs, r, done, info = self.env.step(action)
        # if r != 0:
        #     print('Success!')
        self.R += r * np.power(GAMMA, self.steps)
        if self.steps == self.max_steps:
            done = True
        obs = self.extractor.get_obs(obs)

        r = self.artificial_reward[obs[0]]

        self.steps += 1
        if done:
            print('total return: ', self.R)
            print(f"Saving video to {VIDEO_DIR}")
            self.video_recorder.close()
            # self.video_recorder.enabled = False
            print(f"Video saved")
        # self.env.render()
        return obs, r, done, info

    def get_info(self):
        return self.steps, self.max_steps

