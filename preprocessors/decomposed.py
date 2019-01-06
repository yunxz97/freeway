from preprocessors.base import DDNBasePreprocessor
from constants import INLINE_LEFT, INLINE_RIGHT, SCREEN_WIDTH, SCREEN_HEIGHT, LANES
import numpy as np
from utils import one_hot

class DDNDecomposedPreprocessor(DDNBasePreprocessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def obs_to_state(self, im):
        unprocessed = self.extract_state(im)
        player_y = unprocessed[0]
        car_x = unprocessed[1:11]
        car_inline = ((INLINE_LEFT <= car_x) & (car_x <= INLINE_RIGHT)).astype(np.int)
        player_lane = [1 if lane_top <= player_y and player_y <= lane_bottom else 0 for lane_top, lane_bottom in LANES]
        player_collide = unprocessed[11]

        # print(player_y, car_x, car_inline, player_lane, player_collide)

        all_state = np.concatenate([
            one_hot(player_y, SCREEN_HEIGHT),
            np.concatenate([one_hot(c_x, SCREEN_WIDTH) for c_x in car_x]),
            np.concatenate([one_hot(c_i, 2) for c_i in car_inline]),
            np.array(player_lane),
            one_hot(player_collide, 2)
        ])

        current_state = np.log(np.clip(all_state, 1e-6, 1))
        return current_state
