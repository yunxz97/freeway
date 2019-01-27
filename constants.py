from datetime import datetime
from re import sub

TRAIN_FACTOR_WEIGHTS = True
INFINITY = 1000
SMALL_NON_ZERO = 1e-100
LOG_ZERO = -1e2

# env constants
ENV_MAX_STEPS = 2000
SCREEN_WIDTH = 160
SCREEN_HEIGHT = 210
INLINE_LEFT = 40
INLINE_RIGHT = 50
MAX_LANE = 10
PLAYER_MOVE_SPEED = 4
HIT_IMPACT = 25
FINISHED_LINE = 22
LANES = [(167, 187), (151, 169), (135, 153), (119, 137), (104, 121), (87, 104), (71, 89), (55, 73), (39, 57), (23, 41)]

# params
WORKER_MAX_STEPS = 10000
SIM_STEPS = 10
BP_STEPS = 5
NUM_MULTI_ENV = 2
DEVICE = 'gpu'
EPISODES = 10000
WORKERS = 8
LOG_DIR = 'freeway_logs'
T_MAX = 8
RL_LR = 1e-2
SL_LR = 1e-2
BETA = .01
GAMMA = .99
UNIQUE_ID = int(sub('[-: ]', '', str(datetime.today()).split('.')[0]))
LOAD_MODEL = False
SEQUENTIAL = True
MULTI_FAC = 1