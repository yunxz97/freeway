import numpy as np
from utils import one_hot

from constants import (
    TRAIN_FACTOR_WEIGHTS,
    SCREEN_WIDTH, SCREEN_HEIGHT, INLINE_LEFT, INLINE_RIGHT, MAX_LANE, LANES
)
from factors.base import FactorWrapper, CarMovementFactor, ChickenMovementFactor, DestinationRewardFactor, YRewardFactor


class CarInlineFactor(FactorWrapper):
    def __init__(self, car, train = TRAIN_FACTOR_WEIGHTS):
        super(CarInlineFactor, self).__init__()
        transitionMat = np.zeros(
            shape=[
                SCREEN_WIDTH, 2
            ],
            dtype=np.float32
        )
        transitionMat[slice(INLINE_LEFT, INLINE_RIGHT), 1] = 1

        self.build(transitionMat, train, name=car)


class PlayerLaneFactor(FactorWrapper):
    def __init__(self, train = TRAIN_FACTOR_WEIGHTS):
        super(PlayerLaneFactor, self).__init__()
        transitionMat = np.zeros(
            shape=[
                SCREEN_HEIGHT, MAX_LANE
            ],
            dtype=np.float32
        )
        for player_y in range(SCREEN_HEIGHT):
            lane = 0
            for lane_top, lane_bottom in LANES:
                if lane_top <= player_y and player_y <= lane_bottom:
                    transitionMat[player_y, lane] = 1
                lane += 1
        self.build(transitionMat, train)


class PlayerCollisionFactor(FactorWrapper):
    def __init__(self, train = TRAIN_FACTOR_WEIGHTS):
        super(PlayerCollisionFactor, self).__init__()
        transitionMat = np.zeros(
            shape=[2]*10 + [MAX_LANE, 2],
            dtype=np.float32
        )
        for car in range(MAX_LANE):
            for lane in range(MAX_LANE):
                for collision in range(2):
                    if car == lane and collision:
                        index = tuple(np.concatenate([one_hot(car, MAX_LANE), [lane], [collision]]))
                        transitionMat[index] = 1
        self.build(transitionMat, train)
