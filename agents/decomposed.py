from constants import SCREEN_HEIGHT, SCREEN_WIDTH, MAX_LANE
from agents.base import FreewayBaseAgent

import factors.decomposed as freeway_factors

variable_mapping = {
    "player_y": 0,
    "car1_x": 1,
    "car2_x": 2,
    "car3_x": 3,
    "car4_x": 4,
    "car5_x": 5,
    "car6_x": 6,
    "car7_x": 7,
    "car8_x": 8,
    "car9_x": 9,
    "car10_x": 10,
    "car1_inline": 11,
    "car2_inline": 12,
    "car3_inline": 13,
    "car4_inline": 14,
    "car5_inline": 15,
    "car6_inline": 16,
    "car7_inline": 17,
    "car8_inline": 18,
    "car9_inline": 19,
    "car10_inline": 20,
    "player_lane": 21,
    "player_collide": 22
}

variable_range = {
    "player_y": SCREEN_HEIGHT,
    "car1_x": SCREEN_WIDTH,
    "car2_x": SCREEN_WIDTH,
    "car3_x": SCREEN_WIDTH,
    "car4_x": SCREEN_WIDTH,
    "car5_x": SCREEN_WIDTH,
    "car6_x": SCREEN_WIDTH,
    "car7_x": SCREEN_WIDTH,
    "car8_x": SCREEN_WIDTH,
    "car9_x": SCREEN_WIDTH,
    "car10_x": SCREEN_WIDTH,
    "car1_inline": 2,
    "car2_inline": 2,
    "car3_inline": 2,
    "car4_inline": 2,
    "car5_inline": 2,
    "car6_inline": 2,
    "car7_inline": 2,
    "car8_inline": 2,
    "car9_inline": 2,
    "car10_inline": 2,
    "player_lane": MAX_LANE,
    "player_collide": 2
}


class FreewayDecomposedAgent(FreewayBaseAgent):
    def __init__(self, **kwargs):
        super().__init__(variable_range=variable_range, factors=freeway_factors, **kwargs)

    def add_in_state_factor(self):
        self.in_state_factor = [
            self.create_in_state_factor(
                [variable_mapping[f'player_y'], variable_mapping['player_lane']],
                self.factors.PlayerLaneFactor(train=False),
            ),
            # self.create_in_state_factor(
            #     [variable_mapping[f'player_y']],
            #     DestinationRewardFactor(train=False),
            # ),
            self.create_in_state_factor(
                [variable_mapping["player_y"]],
                self.factors.YRewardFactor(train=False)
            )
        ]
        for car in range(1, MAX_LANE+1):
            self.in_state_factor.append(
                self.create_in_state_factor(
                    [variable_mapping[f'car{car}_x'], variable_mapping[f'car{car}_inline']],
                    self.factors.CarInlineFactor(car, train=False),
                )
            )
        self.in_state_factor.append(
            self.create_in_state_factor(
                [variable_mapping[f'car{car}_inline'] for car in range(1, MAX_LANE+1)] 
                + [variable_mapping['player_lane'], variable_mapping['player_collide']],
                self.factors.PlayerCollisionFactor(train=False),
            )
        )

    def add_cross_state_factor(self):
        self.cross_state_factor = [
            self.create_cross_state_factor(
                [variable_mapping['player_y'], variable_mapping['player_collide']],
                [0], # TODO: CHECK LATER
                [variable_mapping['player_y']],
                self.factors.ChickenMovementFactor(train=False),
            )
        ]
        for car in range(1, MAX_LANE+1):
            self.cross_state_factor.append(
                self.create_cross_state_factor(
                    [variable_mapping[f'car{car}_x']],
                    [],
                    [variable_mapping[f'car{car}_x']],
                    self.factors.CarMovementFactor(car, train=False),
                )
            )
