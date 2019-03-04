import numpy as np
import tensorflow as tf
from lib.transition import Factors, ConvFactor1D
from constants import SMALL_NON_ZERO, TRAIN_FACTOR_WEIGHTS,\
    SCREEN_WIDTH, SCREEN_HEIGHT, HIT_IMPACT, PLAYER_MOVE_SPEED, Y_DOWNSAMPLING, X_DOWNSAMPLING, SL_ENABLE, RANDOM_INIT,\
    REPEAT_ACTION, N_ACTION_REPEAT
from utils import to_log_probability


class FactorWrapper(Factors):
    def build(self, transitionMat, train, SL, RL, name='', max_clip_value=1):
        transitionMat = to_log_probability(transitionMat, SMALL_NON_ZERO, max_clip_value)
        potential = tf.constant(transitionMat, dtype=tf.float32)
        self.transMatrix = transitionMat
        if SL:
            with tf.variable_scope("sl_params"):
                self.sl_params = tf.get_variable('sl' + self.__class__.__name__ + str(name),
                                                 initializer=potential, trainable=train)
            if RL:
                with tf.variable_scope("rl_params"):
                    self.rl_params = tf.get_variable('rl' + self.__class__.__name__ + str(name),
                                                     initializer=tf.zeros_like(self.sl_params), trainable=train)
                self.potential = self.rl_params + self.sl_params
            else:
                self.potential = self.sl_params
        elif RL:
            with tf.variable_scope("rl_params"):
                self.potential = tf.get_variable('rl' + self.__class__.__name__ + str(name),
                                                 initializer=potential, trainable=train)
                # print(self.potential.name)
        else:
            self.potential = tf.get_variable(self.__class__.__name__ + str(name),
                                             initializer=potential, trainable=False)
        # self.potential = tf.get_variable(self.__class__.__name__+str(name), initializer=potential, trainable=train)
        self.beliefs = self.potential


class CarMovementFactor(FactorWrapper):
    # TODO: add support for REPEAT_ACTION
    def __init__(self, car, dist=SCREEN_WIDTH, train=TRAIN_FACTOR_WEIGHTS):
        super().__init__()
        assert type(car) is int and 1 <= car <= 10

        if RANDOM_INIT:
            transition_mtx = np.ones([dist, dist])
        else:
            if X_DOWNSAMPLING:
                speed = [1, 1, 1, 1, 2, -2, -1, -1, -1, -1][car - 1]
                transition_mtx = np.zeros([dist, dist])
                mtx_range = np.arange(dist)
                if car in [1, 10]:
                    transition_mtx[mtx_range, (mtx_range + speed) % dist] = 2 / 5
                    transition_mtx[mtx_range, mtx_range] = 3 / 5
                elif car in [2, 9]:
                    transition_mtx[mtx_range, (mtx_range + speed) % dist] = 1 / 2
                    transition_mtx[mtx_range, mtx_range] = 1 / 2
                elif car in [3, 8]:
                    transition_mtx[mtx_range, (mtx_range + speed) % dist] = 2 / 3
                    transition_mtx[mtx_range, mtx_range] = 1 / 3
                else:
                    transition_mtx[mtx_range, (mtx_range + speed) % dist] = 1
            else:
                speed = [1, 1, 2, 2, 4, -4, -2, -2, -1, -1][car - 1]
                transition_mtx = np.zeros([dist, dist])
                mtx_range = np.arange(dist)
                if car in [1, 10]:
                    transition_mtx[mtx_range, (mtx_range + speed) % dist] = 4 / 5
                    transition_mtx[mtx_range, mtx_range] = 1 / 5
                elif car in [3, 8]:
                    transition_mtx[mtx_range, (mtx_range + speed) % dist] = 2 / 3
                    transition_mtx[mtx_range, mtx_range] = 1 / 3
                else:
                    transition_mtx[mtx_range, (mtx_range + speed) % dist] = 1

        self.build(transition_mtx, train, SL=SL_ENABLE, RL=True, name=car, max_clip_value=1)


class CarMovementConvFactor(ConvFactor1D):
    def __init__(self, car, train=TRAIN_FACTOR_WEIGHTS):
        assert type(car) is int and 1 <= car <= 10

        nchannels = 3

        hksize = 2 * N_ACTION_REPEAT if X_DOWNSAMPLING else 4 * N_ACTION_REPEAT
        ksize = 2 * hksize + 1

        if RANDOM_INIT:
            kernels = np.ones((nchannels, ksize), dtype=np.float32)
        else:
            kernels = np.zeros((nchannels, ksize), dtype=np.float32)
            if REPEAT_ACTION:
                if X_DOWNSAMPLING:
                    speed = (np.array(
                        [.4, .5, .666, 1, 2, -2, -1, -.666, -.5, -.4], dtype=np.float32) * N_ACTION_REPEAT)[car-1]
                else:
                    speed = (np.array(
                        [.8, 1, 1.333, 2, 4, -4, -2, -1.333, -1, -.8], dtype=np.float32) * N_ACTION_REPEAT)[car - 1]
                speed = int(round(speed))
                kernels[:, -speed+hksize] = 1
            else:
                if X_DOWNSAMPLING:
                    speed = [1, 1, 1, 1, 2, -2, -1, -1, -1, -1][car - 1]

                    if car in [1, 10]:
                        kernels[:, hksize] = 3 / 5
                        kernels[:, -speed+hksize] = 2 / 5
                    elif car in [2, 9]:
                        kernels[:, hksize] = 1 / 2
                        kernels[:, -speed+hksize] = 1 / 2
                    elif car in [3, 8]:
                        kernels[:, hksize] = 1 / 3
                        kernels[:, -speed+hksize] = 2 / 3
                    else:
                        kernels[:, -speed+hksize] = 1

                else:
                    speed = [1, 1, 2, 2, 4, -4, -2, -2, -1, -1][car - 1]

                    if car in [1, 10]:
                        kernels[:, hksize] = 1 / 5
                        kernels[:, -speed+hksize] = 4 / 5
                    elif car in [3, 8]:
                        kernels[:, hksize] = 1 / 3
                        kernels[:, -speed+hksize] = 2 / 3
                    else:
                        kernels[:, -speed+hksize] = 1

        kernels = np.transpose(kernels, [1, 0])
        kernels = np.reshape(kernels, [ksize, 1, nchannels])

        super(CarMovementConvFactor, self).__init__(
            nchannels=nchannels, ksize=ksize,
            nlabels=SCREEN_WIDTH,
            kernel=kernels,
            circular_padding=True, trainable=train,
            name=self.__class__.__name__ + str(car),
            with_rl_parmas=True
        )


class ChickenMovementFactor(FactorWrapper): 
    def __init__(self, dist=SCREEN_HEIGHT, train=TRAIN_FACTOR_WEIGHTS):
        super().__init__()
        if RANDOM_INIT:
            transition_mtx = np.ones([dist, 2, 3, dist])
        else:
            transition_mtx = np.zeros([dist, 2, 3, dist])
            mtx_range = np.arange(dist)

            transition_mtx[mtx_range, 0, 0, mtx_range] = 1

            transition_mtx[mtx_range, 0, 1, np.clip(mtx_range - PLAYER_MOVE_SPEED, 0, dist - 1)] = 1

            transition_mtx[mtx_range, 0, 2, np.clip(mtx_range + PLAYER_MOVE_SPEED, 0, dist - 1)] = 1

            transition_mtx[mtx_range, 1, 0, np.clip(mtx_range + HIT_IMPACT, 0, dist - 1)] = 1

            transition_mtx[mtx_range, 1, 1, np.clip(mtx_range - PLAYER_MOVE_SPEED + HIT_IMPACT, 0, dist - 1)] = 1

            transition_mtx[mtx_range, 1, 2, np.clip(mtx_range + PLAYER_MOVE_SPEED + HIT_IMPACT, 0, dist - 1)] = 1

        # print(transition_mtx[191, 0, 0])
        # print(transition_mtx[191, 0, 1])
        # print(transition_mtx[191, 0, 2])

        self.build(transition_mtx, train, SL=SL_ENABLE, RL=True, max_clip_value=1)


class CarHitFactor(FactorWrapper):
    def __init__(self, car, y_range=SCREEN_HEIGHT, x_range = SCREEN_WIDTH, train=TRAIN_FACTOR_WEIGHTS):
        super().__init__()
        assert type(car) is int and 1 <= car <= 10

        if RANDOM_INIT:
            transition_mtx = np.ones([y_range, x_range, 2])
        else:
            if Y_DOWNSAMPLING:
                car1_y = slice(int(round(167/4)), int(round(187/4)))
                car2_y = slice(int(round(151/4)), int(round(169/4)))
                car3_y = slice(int(round(135/4)), int(round(153/4)))
                car4_y = slice(int(round(119/4)), int(round(137/4)))
                car5_y = slice(int(round(104/4)), int(round(121/4)))
                car6_y = slice(int(round(87/4)), int(round(104/4)))
                car7_y = slice(int(round(71/4)), int(round(89/4)))
                car8_y = slice(int(round(55/4)), int(round(73/4)))
                car9_y = slice(int(round(39/4)), int(round(57/4)))
                car10_y = slice(int(round(23/4)), int(round(41/4)))
            else:
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

            if X_DOWNSAMPLING:
                chicken_x = slice(19, 28)
            else:
                chicken_x = slice(39, 55)

            car_slice = [car1_y, car2_y, car3_y, car4_y, car5_y,
                         car6_y, car7_y, car8_y, car9_y, car10_y][car - 1]

            transition_mtx = np.zeros([y_range, x_range, 2])
            transition_mtx[car_slice, chicken_x, 1] = 1
            transition_mtx[:, :, 0] = 1
            transition_mtx[car_slice, chicken_x, 0] = 0

        self.build(transition_mtx, train, SL=SL_ENABLE, RL=True, name=car, max_clip_value=1)


class HitFactor(FactorWrapper):
    def __init__(self, train=TRAIN_FACTOR_WEIGHTS):
        super().__init__()

        # no random init for this factor
        transition_mtx = np.zeros([2] * 11)
        transition_mtx[:, :, :, :, :, :, :, :, :, :, 1] = 1
        transition_mtx[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] = 0
        transition_mtx[:, :, :, :, :, :, :, :, :, :, 0] = 0
        transition_mtx[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] = 1

        self.build(transition_mtx, train, SL=False, RL=True, max_clip_value=1)


class DestinationRewardFactor(FactorWrapper):
    def __init__(self, dist=SCREEN_HEIGHT, train=TRAIN_FACTOR_WEIGHTS):
        super().__init__()

        transition_mtx = np.ones(dist)

        if not RANDOM_INIT:
            transition_mtx[:int(round(SCREEN_HEIGHT/8))] = 1000000000

        self.build(transition_mtx, train, SL=SL_ENABLE, RL=True, max_clip_value=int(1e9))


class YRewardFactor(FactorWrapper):
    def __init__(self, dist=SCREEN_HEIGHT, train=TRAIN_FACTOR_WEIGHTS):
        super().__init__()

        transition_mtx = np.arange(20, 1, -19 / dist)[:dist]
        # print(transition_mtx)
        self.build(transition_mtx, train, SL=False, RL=True, max_clip_value=dist)
