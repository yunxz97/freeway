import gym
from gym.utils.play import play
import numpy as np
from scipy import signal

"""
    "chicken_y": 0,
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
    "car1_hit": 11,
    "car2_hit": 12,
    "car3_hit": 13,
    "car4_hit": 14,
    "car5_hit": 15,
    "car6_hit": 16,
    "car7_hit": 17,
    "car8_hit": 18,
    "car9_hit": 19,
    "car10_hit": 20,
    "hit": 21
"""


class StateExtractor:
    def __init__(self, im):
        self.im = im
        self.positions_t = [0]
        self.positions_tp1 = self.positions_t
        self.hit_counter = [0] * 10
        self.extract_positions(im)

    def extract_positions(self, im):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # chicken position
            sub_im = im[:, 40:60]
            mask_r = (sub_im[:, :, 0] > 240)
            mask_g = (sub_im[:, :, 1] > 240)
            mask_b = (sub_im[:, :, 2] > 80) & (sub_im[:, :, 2] < 90)
            mask = (mask_r & mask_g & mask_b).astype(np.int16)
            mask[mask == 0] = -1
            chicken_body = np.array(
                            [[1, 0, 1, 1, 1],
                            [1, 1, 1, 1, 1]])
            match_value = 9
            mask_conv = signal.fftconvolve(mask, chicken_body[::-1][:, ::-1], 'same')
            match_map = (mask_conv > match_value-.1)
            match_indices = np.where(match_map)[0]
            if match_indices.size == 0:
                # print("==================================\n" * 2)
                chicken_pos = self.positions_t[0]
            else:
                chicken_pos = match_indices[0]

            # car1
            sub_im = im[170:185]
            mask_r = (sub_im[:, :, 0] > 200) & (sub_im[:, :, 0] < 220)
            mask_g = (sub_im[:, :, 1] > 200) & (sub_im[:, :, 1] < 220)
            mask_b = (sub_im[:, :, 2] > 60) & (sub_im[:, :, 2] < 70)
            mask = (mask_r & mask_g & mask_b)
            car1_pos = np.mean(np.where(mask)[1], dtype=np.int16)

            # car2
            sub_im = im[150:170]
            mask_r = (sub_im[:, :, 0] > 130) & (sub_im[:, :, 0] < 140)
            mask_g = (sub_im[:, :, 1] > 180) & (sub_im[:, :, 1] < 190)
            mask_b = (sub_im[:, :, 2] > 80) & (sub_im[:, :, 2] < 90)
            mask = (mask_r & mask_g & mask_b)
            car2_pos = np.mean(np.where(mask)[1], dtype=np.int16)

            # car3
            sub_im = im[135:150]
            mask_r = (sub_im[:, :, 0] > 180) & (sub_im[:, :, 0] < 190)
            mask_g = (sub_im[:, :, 1] > 45) & (sub_im[:, :, 1] < 55)
            mask_b = (sub_im[:, :, 2] > 45) & (sub_im[:, :, 2] < 55)
            mask = (mask_r & mask_g & mask_b)
            car3_pos = np.mean(np.where(mask)[1], dtype=np.int16)

            # car4
            sub_im = im[120:135]
            mask_r = (sub_im[:, :, 0] > 80) & (sub_im[:, :, 0] < 90)
            mask_g = (sub_im[:, :, 1] > 85) & (sub_im[:, :, 1] < 95)
            mask_b = (sub_im[:, :, 2] > 210) & (sub_im[:, :, 2] < 220)
            mask = (mask_r & mask_g & mask_b)
            car4_pos = np.mean(np.where(mask)[1], dtype=np.int16)

            # car5
            sub_im = im[105:120]
            mask_r = (sub_im[:, :, 0] > 160) & (sub_im[:, :, 0] < 170)
            mask_g = (sub_im[:, :, 1] > 95) & (sub_im[:, :, 1] < 105)
            mask_b = (sub_im[:, :, 2] > 30) & (sub_im[:, :, 2] < 40)
            mask = (mask_r & mask_g & mask_b)
            car5_pos = np.mean(np.where(mask)[1], dtype=np.int16)

            # car6
            sub_im = im[85:105]
            mask_r = (sub_im[:, :, 0] > 20) & (sub_im[:, :, 0] < 30)
            mask_g = (sub_im[:, :, 1] > 20) & (sub_im[:, :, 1] < 30)
            mask_b = (sub_im[:, :, 2] > 160) & (sub_im[:, :, 2] < 170)
            mask = (mask_r & mask_g & mask_b)
            car6_pos = np.mean(np.where(mask)[1], dtype=np.int16)

            # car7
            sub_im = im[70:85]
            mask_r = (sub_im[:, :, 0] > 225) & (sub_im[:, :, 0] < 235)
            mask_g = (sub_im[:, :, 1] > 105) & (sub_im[:, :, 1] < 115)
            mask_b = (sub_im[:, :, 2] > 105) & (sub_im[:, :, 2] < 115)
            mask = (mask_r & mask_g & mask_b)
            car7_pos = np.mean(np.where(mask)[1], dtype=np.int16)

            # car8
            sub_im = im[55:70]
            mask_r = (sub_im[:, :, 0] > 100) & (sub_im[:, :, 0] < 110)
            mask_g = (sub_im[:, :, 1] > 100) & (sub_im[:, :, 1] < 110)
            mask_b = (sub_im[:, :, 2] > 10) & (sub_im[:, :, 2] < 20)
            mask = (mask_r & mask_g & mask_b)
            car8_pos = np.mean(np.where(mask)[1], dtype=np.int16)

            # car9
            sub_im = im[40:55]
            mask_r = (sub_im[:, :, 0] > 175) & (sub_im[:, :, 0] < 185)
            mask_g = (sub_im[:, :, 1] > 225) & (sub_im[:, :, 1] < 235)
            mask_b = (sub_im[:, :, 2] > 110) & (sub_im[:, :, 2] < 120)
            mask = (mask_r & mask_g & mask_b)
            car9_pos = np.mean(np.where(mask)[1], dtype=np.int16)

            # car10
            sub_im = im[25:40]
            mask_r = (sub_im[:, :, 0] > 160) & (sub_im[:, :, 0] < 170)
            mask_g = (sub_im[:, :, 1] > 20) & (sub_im[:, :, 1] < 30)
            mask_b = (sub_im[:, :, 2] > 20) & (sub_im[:, :, 2] < 30)
            mask = (mask_r & mask_g & mask_b)
            car10_pos = np.mean(np.where(mask)[1], dtype=np.int16)

        result = np.array([chicken_pos, car1_pos, car2_pos, car3_pos, car4_pos,
                            car5_pos, car6_pos, car7_pos, car8_pos, car9_pos, car10_pos], dtype=np.int16)
        self.positions_t = self.positions_tp1
        self.positions_tp1 = result
        return result

    def extract_hit(self):
        hit = [False] * 10
        hit[0] = (170 <= self.positions_tp1[0] <= 185) and (40 <= self.positions_tp1[1] <= 50)
        hit[1] = (150 <= self.positions_tp1[0] <= 170) and (40 <= self.positions_tp1[2] <= 50)
        hit[2] = (135 <= self.positions_tp1[0] <= 150) and (40 <= self.positions_tp1[3] <= 50)
        hit[3] = (120 <= self.positions_tp1[0] <= 135) and (40 <= self.positions_tp1[4] <= 50)
        hit[4] = (105 <= self.positions_tp1[0] <= 120) and (40 <= self.positions_tp1[5] <= 50)
        hit[5] = (85 <= self.positions_tp1[0] <= 105) and (40 <= self.positions_tp1[6] <= 50)
        hit[6] = (70 <= self.positions_tp1[0] <= 85) and (40 <= self.positions_tp1[7] <= 50)
        hit[7] = (55 <= self.positions_tp1[0] <= 70) and (40 <= self.positions_tp1[8] <= 50)
        hit[8] = (40 <= self.positions_tp1[0] <= 55) and (40 <= self.positions_tp1[9] <= 50)
        hit[9] = (25 <= self.positions_tp1[0] <= 40) and (40 <= self.positions_tp1[10] <= 50)

        for i in range(10):
            if self.hit_counter[i] == 0:
                if hit[i]:
                    self.hit_counter[i] = 3
            else:
                hit[i] = False
                self.hit_counter[i] -= 1

        hit_combined = any(hit)

        return [*hit, hit_combined]
        
    def extract_state(self, im):
        self.extract_positions(im)
        hit = [self.extract_hit()[-1]]
        speed = self.positions_tp1 - self.positions_t
        missing_val_mask = np.logical_or(self.positions_t == 0, self.positions_tp1 == 0)
        speed[missing_val_mask] = 0
        return np.concatenate([self.positions_tp1, hit])


env = gym.make('FreewayDeterministic-v4')
im = env.reset()
extractor = StateExtractor(im)


def on_frame(obs, obs_tp1, action, reward, *_):
    from time import sleep
    on_frame.counter += 1
    state = extractor.extract_state(obs)
    if on_frame.counter % 1 == 0:
        print(state)
        print(action)
    if reward != 0:
        print(state)
        print(reward)
        sleep(5)


on_frame.counter = 0
play(env, zoom=6, callback=on_frame)
#
# env = gym.make('FreewayDeterministic-v4')
# im = env.reset()
# # pil_im = Image.fromarray(im)
# # pil_im.show()
# extractor = StateExtractor(im)
#
# # pil_im.save('step0.png')
#
# rewards = []
# for i in range(200):
#     im, r, done, _ = env.step(1)
#     if done:
#         break
#     # if i % 3 == 0:
#     print(extractor.extract_state(im))
#     rewards.append(r)
#     # if i % 4 == 0:
#     #     pil_im = Image.fromarray(im)
#     #     pil_im.save('step'+str(i//4)+'.png')
# # print(rewards)



# pil_im = Image.open('step13.png')
# pil_im.load()
# im = np.asarray(pil_im)
# print(im[82, 76])
