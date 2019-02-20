import numpy as np
import tensorflow as tf
from lib.transition import Factors, ConvFactor1D
from constants import SMALL_NON_ZERO, TRAIN_FACTOR_WEIGHTS, \
    SCREEN_WIDTH, SCREEN_HEIGHT, HIT_IMPACT, PLAYER_MOVE_SPEED, DOWNSAMPLING, SL_ENABLE, LOG_ZERO
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
    def __init__(self, car, dist=SCREEN_WIDTH, train=TRAIN_FACTOR_WEIGHTS):
        super().__init__()
        assert type(car) is int and 1 <= car <= 10
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
        self.speed = [1, 1, 2, 2, 4, -4, -2, -2, -1, -1][car - 1]

        nchannels = 3
        ksize = abs(self.speed)+1
        kernels = np.zeros((nchannels, ksize), dtype=np.float32)
        if self.speed > 0:
            if car == 1:
                kernels[:] = 800, 200
            elif car == 3:
                kernels[:] = 666, 0, 333
            else:
                kernels[:, 0] = 1000
        else:
            if car == 10:
                kernels[:] = 200, 800
            elif car == 8:
                kernels[:] = 333, 0, 666
            else:
                kernels[:, -1] = 1000

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

    def padding_inputs(self, pos, pad_val=LOG_ZERO):
        if self.speed > 0:
            left_pad = pos[:, -self.speed:, :]
            padded_val = tf.concat([left_pad, pos], axis=1)
        else:
            right_pad = pos[:, self.speed:, :]
            padded_val = tf.concat([pos, right_pad], axis=1)
        # else:
        #     hksize = int(self.ksize / 2)
        #     left_pad = pos[:, -hksize:, :]
        #     right_pad = pos[:, :hksize, :]
        #     padded_val = tf.concat([left_pad, pos, right_pad], axis=1)
        return padded_val


class ChickenMovementFactor(FactorWrapper): 
    def __init__(self, dist=SCREEN_HEIGHT, train=TRAIN_FACTOR_WEIGHTS):
        super().__init__()
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


# class ChickenMovementDownsampledFactor(FactorWrapper):
#     def __init__(self, dist=SCREEN_HEIGHT//4, train=TRAIN_FACTOR_WEIGHTS):
#         super().__init__()
#         transition_mtx = np.zeros([dist, 2, 3, dist])
#         mtx_range = np.arange(dist)
#
#         transition_mtx[mtx_range, 0, 0, mtx_range] = 1
#
#         transition_mtx[mtx_range, 0, 1, np.clip(mtx_range - PLAYER_MOVE_SPEED//4, 0, dist - 1)] = 1
#
#         transition_mtx[mtx_range, 0, 2, np.clip(mtx_range + PLAYER_MOVE_SPEED//4, 0, dist - 1)] = 1
#
#         transition_mtx[mtx_range, 1, 0, np.clip(mtx_range + HIT_IMPACT//4, 0, dist - 1)] = 1
#
#         transition_mtx[mtx_range, 1, 1, np.clip(mtx_range - PLAYER_MOVE_SPEED//4 + HIT_IMPACT//4, 0, dist - 1)] = 1
#
#         transition_mtx[mtx_range, 1, 2, np.clip(mtx_range + PLAYER_MOVE_SPEED//4 + HIT_IMPACT//4, 0, dist - 1)] = 1
#
#         # print(transition_mtx[191, 0, 0])
#         # print(transition_mtx[191, 0, 1])
#         # print(transition_mtx[191, 0, 2])
#
#         self.build(transition_mtx, train, SL=True, RL=True, max_clip_value=1)


class CarHitFactor(FactorWrapper):
    def __init__(self, car, dist=SCREEN_HEIGHT, train=TRAIN_FACTOR_WEIGHTS):
        super().__init__()
        assert type(car) is int and 1 <= car <= 10

        if DOWNSAMPLING:
            car1_y = slice(167//4, 187//4)
            car2_y = slice(151//4, 169//4)
            car3_y = slice(135//4, 153//4)
            car4_y = slice(119//4, 137//4)
            car5_y = slice(104//4, 121//4)
            car6_y = slice(87//4, 104//4)
            car7_y = slice(71//4, 89//4)
            car8_y = slice(55//4, 73//4)
            car9_y = slice(39//4, 57//4)
            car10_y = slice(23//4, 41//4)
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

        chicken_x = slice(39, 55)

        car_slice = [car1_y, car2_y, car3_y, car4_y, car5_y,
                     car6_y, car7_y, car8_y, car9_y, car10_y][car - 1]

        transition_mtx = np.zeros([dist, 160, 2])
        transition_mtx[car_slice, chicken_x, 1] = 1
        transition_mtx[:, :, 0] = 1
        transition_mtx[car_slice, chicken_x, 0] = 0

        self.build(transition_mtx, train, SL=SL_ENABLE, RL=True, name=car, max_clip_value=1)

#
# class CarHitDownsampledFactor(FactorWrapper):
#     def __init__(self, car, dist=SCREEN_HEIGHT, train=TRAIN_FACTOR_WEIGHTS):
#         super().__init__()
#         assert type(car) is int and 1 <= car <= 10
#
#         car1_y = slice(167//4, 187//4)
#         car2_y = slice(151//4, 169//4)
#         car3_y = slice(135//4, 153//4)
#         car4_y = slice(119//4, 137//4)
#         car5_y = slice(104//4, 121//4)
#         car6_y = slice(87//4, 104//4)
#         car7_y = slice(71//4, 89//4)
#         car8_y = slice(55//4, 73//4)
#         car9_y = slice(39//4, 57//4)
#         car10_y = slice(23//4, 41//4)
#         chicken_x = slice(39, 55)
#
#         car_slice = [car1_y, car2_y, car3_y, car4_y, car5_y,
#                      car6_y, car7_y, car8_y, car9_y, car10_y][car - 1]
#
#         transition_mtx = np.zeros([dist, 160, 2])
#         transition_mtx[car_slice, chicken_x, 1] = 1
#         transition_mtx[:, :, 0] = 1
#         transition_mtx[car_slice, chicken_x, 0] = 0
#
#         self.build(transition_mtx, train, SL=True, RL=True, name=car, max_clip_value=1)


class HitFactor(FactorWrapper):
    def __init__(self, train=TRAIN_FACTOR_WEIGHTS):
        super().__init__()

        transition_mtx = np.zeros([2] * 11)
        transition_mtx[:, :, :, :, :, :, :, :, :, :, 1] = 1
        transition_mtx[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] = 0
        transition_mtx[:, :, :, :, :, :, :, :, :, :, 0] = 0
        transition_mtx[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] = 1

        self.build(transition_mtx, train, SL=SL_ENABLE, RL=False, max_clip_value=1)


class DestinationRewardFactor(FactorWrapper):
    def __init__(self, dist=SCREEN_HEIGHT, train=TRAIN_FACTOR_WEIGHTS):
        super().__init__()

        transition_mtx = np.ones(dist)

        transition_mtx[:SCREEN_HEIGHT//8] = 5

        self.build(transition_mtx, train, SL=SL_ENABLE, RL=True, max_clip_value=100)


# class DestinationRewardDownsampledFactor(FactorWrapper):
#     def __init__(self, dist=SCREEN_HEIGHT, train=TRAIN_FACTOR_WEIGHTS):
#         super().__init__()
#
#         transition_mtx = np.ones(dist//4)
#
#         transition_mtx[:6] = 10
#
#         self.build(transition_mtx, train, SL=True, RL=True, max_clip_value=10000)


class YRewardFactor(FactorWrapper):
    def __init__(self, dist=SCREEN_HEIGHT, train=TRAIN_FACTOR_WEIGHTS):
        super().__init__()

        transition_mtx = np.arange(1, .99, -.01 / dist)[:dist]
        # print(transition_mtx)
        self.build(transition_mtx, train, SL=SL_ENABLE, RL=True, max_clip_value=dist)
