import numpy as np
import tensorflow as tf
from lib.transition import Factors

TRAIN_FACTOR_WEIGHTS = True
INFINITY = 10
SPEED_RANGE = 21  # -10 ~ 10
max_speed = 10

X = 160
Y = 210


class CarMovementFactor(Factors):
    def __init__(self, car, dist=X, train=TRAIN_FACTOR_WEIGHTS):
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

        transition_mtx = np.log(np.clip(transition_mtx, 1e-6, 1))

        potential = tf.constant(transition_mtx, dtype=tf.float32)
        self.transMatrix = transition_mtx
        self.potential = tf.get_variable(self.__class__.__name__+str(car), initializer=potential, trainable=train)
        self.beliefs = self.potential


class ChickenMovementFactor(Factors):
    def __init__(self, dist=Y, train=TRAIN_FACTOR_WEIGHTS):
        super().__init__()
        speed = 4
        hit_impact = 25

        transition_mtx = np.zeros([dist, 2, 3, dist])
        mtx_range = np.arange(dist)

        transition_mtx[mtx_range, 0, 0, mtx_range] = 1

        transition_mtx[mtx_range, 0, 1, np.clip(mtx_range-speed, dist-1, 0)] = 1

        transition_mtx[mtx_range, 0, 2, np.clip(mtx_range+speed, dist-1, 0)] = 1

        transition_mtx[mtx_range, 1, 0, np.clip(mtx_range+hit_impact, dist-1, 0)] = 1

        transition_mtx[mtx_range, 1, 1, np.clip(mtx_range-speed+hit_impact, dist-1, 0)] = 1

        transition_mtx[mtx_range, 1, 2, np.clip(mtx_range+speed+hit_impact, dist-1, 0)] = 1

        potential = tf.constant(transition_mtx, dtype=tf.float32)
        self.transMatrix = transition_mtx
        self.potential = tf.get_variable(self.__class__.__name__, initializer=potential, trainable=train)
        self.beliefs = self.potential


class CarHitFactor(Factors):
    def __init__(self, car, train=TRAIN_FACTOR_WEIGHTS):
        super().__init__()
        assert type(car) is int and 1 <= car <= 10

        car1_y = slice(170, 185)
        car2_y = slice(150, 170)
        car3_y = slice(135, 150)
        car4_y = slice(120, 135)
        car5_y = slice(105, 120)
        car6_y = slice(85, 105)
        car7_y = slice(70, 85)
        car8_y = slice(55, 70)
        car9_y = slice(40, 55)
        car10_y = slice(25, 40)
        chicken_x = slice(44, 48)

        car_slice = [car1_y, car2_y, car3_y, car4_y, car5_y,
                     car6_y, car7_y, car8_y, car9_y, car10_y][car-1]

        transition_mtx = np.zeros([210, 160, 2])
        transition_mtx[car_slice, chicken_x, 1] = 1

        transition_mtx = np.log(np.clip(transition_mtx, 1e-6, 1))

        potential = tf.constant(transition_mtx, dtype=tf.float32)
        self.transMatrix = transition_mtx
        self.potential = tf.get_variable(self.__class__.__name__+str(car), initializer=potential, trainable=train)
        self.beliefs = self.potential


class HitFactor(Factors):
    def __init__(self, train=TRAIN_FACTOR_WEIGHTS):
        super().__init__()

        transition_mtx = np.zeros([2] * 11)
        transition_mtx[:, :, :, :, :, :, :, :, :, :, 1] = 1
        transition_mtx[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] = 0
        transition_mtx[:, :, :, :, :, :, :, :, :, :, 0] = 0
        transition_mtx[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] = 1

        potential = tf.constant(transition_mtx, dtype=tf.float32)
        self.transMatrix = transition_mtx
        self.potential = tf.get_variable(self.__class__.__name__, initializer=potential, trainable=train)
        self.beliefs = self.potential


class DestinationRewardFactor(Factors):
    def __init__(self, dist=Y, train=TRAIN_FACTOR_WEIGHTS):
        super().__init__()

        transition_mtx = np.array([0] * dist)

        transition_mtx[:23] = 10000

        potential = tf.constant(transition_mtx, dtype=tf.float32)
        self.transMatrix = transition_mtx
        self.potential = tf.get_variable(self.__class__.__name__, initializer=potential, trainable=train)
        self.beliefs = self.potential



