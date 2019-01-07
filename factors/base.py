import numpy as np
import tensorflow as tf
from lib.transition import Factors
from constants import SMALL_NON_ZERO, TRAIN_FACTOR_WEIGHTS, SCREEN_WIDTH, SCREEN_HEIGHT, HIT_IMPACT, PLAYER_MOVE_SPEED
from utils import to_log_probability


class FactorWrapper(Factors):
    def build(self, transitionMat, train, name='', max_clip_value=1):
        transitionMat = to_log_probability(transitionMat, SMALL_NON_ZERO, max_clip_value)
        potential = tf.constant(transitionMat, dtype=tf.float32)
        self.transMatrix = transitionMat
        self.potential = tf.get_variable(self.__class__.__name__+str(name) , initializer=potential, trainable=train)
        self.beliefs = self.potential


class CarMovementFactor(FactorWrapper):
    def __init__(self, car, dist=SCREEN_WIDTH, train=TRAIN_FACTOR_WEIGHTS):
        super().__init__()
        assert type(car) is int and 1 <= car <= 10
        speed = [1, 1, 2, 2, 4, -4, -2, -2, -1, -1][car-1]

        transition_mtx = np.zeros([dist, dist])
        mtx_range = np.arange(dist)
        if car in [1, 10]:
            transition_mtx[mtx_range, (mtx_range + speed) % dist] = 4/5
            transition_mtx[mtx_range, mtx_range] = 1/5
        elif car in [3, 8]:
            transition_mtx[mtx_range, (mtx_range + speed) % dist] = 2/3
            transition_mtx[mtx_range, mtx_range] = 1/3
        else:
            transition_mtx[mtx_range, (mtx_range+speed) % dist] = 1

        self.build(transition_mtx, train, name=car, max_clip_value=1)


class ChickenMovementFactor(FactorWrapper):
    def __init__(self, dist=SCREEN_HEIGHT, train=TRAIN_FACTOR_WEIGHTS):
        super().__init__()
        transition_mtx = np.zeros([dist, 2, 3, dist])
        mtx_range = np.arange(dist)

        transition_mtx[mtx_range, 0, 0, mtx_range] = 1

        transition_mtx[mtx_range, 0, 1, np.clip(mtx_range-PLAYER_MOVE_SPEED, 0, dist-1)] = 1

        transition_mtx[mtx_range, 0, 2, np.clip(mtx_range+PLAYER_MOVE_SPEED, 0, dist-1)] = 1

        transition_mtx[mtx_range, 1, 0, np.clip(mtx_range+HIT_IMPACT, 0, dist-1)] = 1

        transition_mtx[mtx_range, 1, 1, np.clip(mtx_range-PLAYER_MOVE_SPEED+HIT_IMPACT, 0, dist-1)] = 1

        transition_mtx[mtx_range, 1, 2, np.clip(mtx_range+PLAYER_MOVE_SPEED+HIT_IMPACT, 0, dist-1)] = 1

        # print(transition_mtx[191, 0, 0])
        # print(transition_mtx[191, 0, 1])
        # print(transition_mtx[191, 0, 2])

        self.build(transition_mtx, train, max_clip_value=1)


class CarHitFactor(FactorWrapper):
    def __init__(self, car, train=TRAIN_FACTOR_WEIGHTS):
        super().__init__()
        assert type(car) is int and 1 <= car <= 10

        car1_y = slice(167, 187)
        car2_y = slice(151, 169)
        car3_y = slice(135, 153)
        car4_y = slice(119, 137)
        car5_y = slice(104, 121)
        car6_y = slice(87, 104)
        car7_y = slice(71, 89)
        car8_y = slice(55, 73)
        car9_y = slice(39, 57)
        car10_y = slice(23, 41)
        chicken_x = slice(39, 55)

        car_slice = [car1_y, car2_y, car3_y, car4_y, car5_y,
                     car6_y, car7_y, car8_y, car9_y, car10_y][car-1]

        transition_mtx = np.zeros([210, 160, 2])
        transition_mtx[car_slice, chicken_x, 1] = 1
        transition_mtx[:, :, 0] = 1
        transition_mtx[car_slice, chicken_x, 0] = 0

        self.build(transition_mtx, train, name=car, max_clip_value=1)


class HitFactor(FactorWrapper):
    def __init__(self, train=TRAIN_FACTOR_WEIGHTS):
        super().__init__()

        transition_mtx = np.zeros([2] * 11)
        transition_mtx[:, :, :, :, :, :, :, :, :, :, 1] = 1
        transition_mtx[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] = 0
        transition_mtx[:, :, :, :, :, :, :, :, :, :, 0] = 0
        transition_mtx[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] = 1

        self.build(transition_mtx, train, max_clip_value=1)


class DestinationRewardFactor(FactorWrapper):
    def __init__(self, dist=SCREEN_HEIGHT, train=TRAIN_FACTOR_WEIGHTS):
        super().__init__()

        transition_mtx = np.ones(dist)

        transition_mtx[:23] = 10000

        self.build(transition_mtx, train, max_clip_value=10000)


class YRewardFactor(FactorWrapper):
    def __init__(self, dist=SCREEN_HEIGHT, train=TRAIN_FACTOR_WEIGHTS):
        super().__init__()

        transition_mtx = np.arange(1, .99, -.01/dist)[:dist]
        # print(transition_mtx)
        self.build(transition_mtx, train, max_clip_value=dist)


