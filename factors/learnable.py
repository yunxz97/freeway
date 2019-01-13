import numpy as np
from random import randint

from constants import TRAIN_FACTOR_WEIGHTS, SCREEN_WIDTH, SCREEN_HEIGHT, HIT_IMPACT, PLAYER_MOVE_SPEED

from factors.base import FactorWrapper, YRewardFactor, HitFactor, DestinationRewardFactor

class CarMovementFactor(FactorWrapper):
    def __init__(self, car, dist=SCREEN_WIDTH, train=TRAIN_FACTOR_WEIGHTS):
        super().__init__()

        transition_mtx = np.zeros([dist, dist])
        mtx_range = np.arange(dist)
        possible_speeds = range(-4, 5)
        for speed in possible_speeds:
            transition_mtx[mtx_range, (mtx_range + speed) % dist] = 1/float(len(possible_speeds))

        self.build(transition_mtx, train, name=car, max_clip_value=1)


class ChickenMovementFactor(FactorWrapper):
    def __init__(self, dist=SCREEN_HEIGHT, train=TRAIN_FACTOR_WEIGHTS):
        super().__init__()
        transition_mtx = np.zeros([dist, 2, 3, dist])
        mtx_range = np.arange(dist)

        possible_player_speeds = range(1, PLAYER_MOVE_SPEED*2)
        possible_hit_impacts = range(1, HIT_IMPACT*2)

        proba_no_hit = 1/float(len(possible_player_speeds))
        proba_hit = 1/float(len(possible_player_speeds)) * 1/float(len(possible_hit_impacts))

        transition_mtx[mtx_range, 0, 0, mtx_range] = 1

        for p_speed in possible_player_speeds:
            transition_mtx[mtx_range, 0, 1, np.clip(mtx_range-p_speed, 0, dist-1)] = proba_no_hit
            transition_mtx[mtx_range, 0, 2, np.clip(mtx_range+p_speed, 0, dist-1)] = proba_no_hit

            for p_hit_impact in possible_hit_impacts:
                transition_mtx[mtx_range, 1, 0, np.clip(mtx_range+p_hit_impact, 0, dist-1)] = proba_hit
                transition_mtx[mtx_range, 1, 1, np.clip(mtx_range-p_speed+p_hit_impact, 0, dist-1)] = proba_hit
                transition_mtx[mtx_range, 1, 2, np.clip(mtx_range+p_speed+p_hit_impact, 0, dist-1)] = proba_hit

        self.build(transition_mtx, train, max_clip_value=1)


class CarHitFactor(FactorWrapper):
    def __init__(self, car, train=TRAIN_FACTOR_WEIGHTS):
        super().__init__()
        assert type(car) is int and 1 <= car <= 10

        car_idx = [(167,187), (151,169), (135, 153), (119, 137), (104, 121), (87, 104), (71, 89), (55, 73), (39, 57), (23, 41)][car-1]

        # random perturb
        PERTURB = 20
        car_slice = slice(car_idx[0] + randint(-PERTURB, PERTURB), car_idx[1] + randint(-PERTURB, PERTURB))
        chicken_x = slice(39 + randint(-PERTURB, PERTURB), 55 + randint(-PERTURB, PERTURB))

        transition_mtx = np.zeros([210, 160, 2])
        transition_mtx[car_slice, chicken_x, 1] = 1
        transition_mtx[:, :, 0] = 1
        transition_mtx[car_slice, chicken_x, 0] = 0

        self.build(transition_mtx, train, name=car, max_clip_value=1)