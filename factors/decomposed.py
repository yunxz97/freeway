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
        lane = 0
        for lane_top, lane_bottom in LANES:
            transitionMat[slice(lane_top, lane_bottom), lane] = 1
            lane += 1
        self.build(transitionMat, train)


class PlayerCollisionFactor(FactorWrapper):
    def __init__(self, train = TRAIN_FACTOR_WEIGHTS):
        super(PlayerCollisionFactor, self).__init__()
        transitionMat = np.zeros(
            shape=[2]*10 + [MAX_LANE, 2],
            dtype=np.float32
        )
        transitionMat[1, :, :, :, :, :, :, :, :, :, 0, 1] = 1
        transitionMat[:, 1, :, :, :, :, :, :, :, :, 1, 1] = 1
        transitionMat[:, :, 1, :, :, :, :, :, :, :, 2, 1] = 1
        transitionMat[:, :, :, 1, :, :, :, :, :, :, 3, 1] = 1
        transitionMat[:, :, :, :, 1, :, :, :, :, :, 4, 1] = 1
        transitionMat[:, :, :, :, :, 1, :, :, :, :, 5, 1] = 1
        transitionMat[:, :, :, :, :, :, 1, :, :, :, 6, 1] = 1
        transitionMat[:, :, :, :, :, :, :, 1, :, :, 7, 1] = 1
        transitionMat[:, :, :, :, :, :, :, :, 1, :, 8, 1] = 1
        transitionMat[:, :, :, :, :, :, :, :, :, 1, 9, 1] = 1
        self.build(transitionMat, train)
