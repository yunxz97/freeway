import numpy as np
import gym
from preprocessors.base import DDNBasePreprocessor
from constants import SCREEN_HEIGHT, SCREEN_WIDTH, ENV_MAX_STEPS, GAMMA


class FreewayEnvironment:
    steps, max_steps = 0, ENV_MAX_STEPS
    gamma = GAMMA

    def __init__(self, args = {}, env="Freeway-v0"):
        self.max_steps = args.get("max_steps", self.max_steps)
        self.gamma = args.get("gamma", self.gamma)
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
        print('total return: ', self.R)
        print('resetting...')
        self.R = 0
        self.steps = 0
        obs = self.env.reset()
        obs = self.extractor.get_obs(obs)
        return obs

    def step(self, action):
        obs, r, done, info = self.env.step(action)
        if r != 0:
            print('Success!')
            self.R += r * np.power(GAMMA, self.steps)
        if self.steps == self.max_steps:
            done = True
        obs = self.extractor.get_obs(obs)
        self.steps += 1
        # self.env.render()
        return obs, r, done, info

    def get_info(self):
        return self.steps, self.max_steps

