import ctypes
from copy import deepcopy
import numpy as np
from multiprocessing import Pipe, Process
from multiprocessing.sharedctypes import RawArray

START, STEP, RESET, STOP, DONE, INFO = range(6)


class Space:
    """
    Holds information about any generic space
    In essence is a simplification of gym.spaces module into a single endpoint
    """
    def __init__(self, shape=(), dtype=np.int32, domain=(0, 1), categorical=False, name=None):
        self.name = name
        self.shape, self.dtype = shape, dtype
        self.categorical, (self.lo, self.hi) = categorical, domain


class ProcEnv :
    def __init__(self, env, id, shm):
        self._env = env
        self.idx = id
        self.shm = shm
        self.conn = self.w_conn = self.proc = None

    def start(self):
        self.conn, self.w_conn = Pipe()
        self.proc = Process(target=self._run)
        self.proc.start()
        self.conn.send((START, None))

    def step(self, act):
        self.conn.send((STEP, act))

    def reset(self):
        self.conn.send((RESET, None))

    def stop(self):
        self.conn.send((STOP, None))

    def wait(self):
        return self.conn.recv()

    def obs_spec(self):
        return self._env.obs_spec()

    def act_spec(self):
        return self._env.act_spec()

    def get_info(self):
        self.conn.send((INFO, None))

    def _run(self):
        try:
            while True:
                msg, data = self.w_conn.recv()
                if msg == START:
                    self._env.start()
                    self.w_conn.send(DONE)
                elif msg == STEP:
                    obs, rew, done, info = self._env.step(data)
                    # print(obs, rew, done)
                    for shm, ob in zip(self.shm, [obs] + [rew, done]):
                        np.copyto(dst=shm[self.idx], src=ob)
                    # print('shm', self.shm)
                    self.w_conn.send(DONE)
                elif msg == RESET:
                    obs = self._env.reset()
                    for shm, ob in zip(self.shm, [obs] + [0, 0]):
                        np.copyto(dst=shm[self.idx], src=ob)
                    self.w_conn.send(DONE)
                elif msg == STOP:
                    self._env.stop()
                    self.w_conn.close()
                    break
                elif msg == INFO:
                    info = self._env.get_info()
                    self.w_conn.send((info))
        except KeyboardInterrupt:
            self._env.stop()
            self.w_conn.close()


class MultiEnv:
    def __init__(self, env, num_envs=1):
        self.num_envs = num_envs
        self.shm = [make_shared(
            num_envs, Space(shape=(len(s),), name='obs_{}'.format(idx))) for idx, s in enumerate(env.obs_spec())]
        self.shm.append(make_shared(num_envs, Space((1,), name="reward", dtype=np.float32)))
        self.shm.append(make_shared(num_envs, Space((1,), name="done")))
        self.envs = [ProcEnv(environment, id, self.shm)
                     for id, environment in enumerate([env] + [deepcopy(env)
                                                               for _ in range(self.num_envs-1)])]

        for environment in self.envs :
            environment.start()
        self.wait()

    def wait(self):
        return [e.wait() for e in self.envs]

    def _observe(self):
        self.wait()
        # print(x)
        # obs, reward, done = zip(*x)
        obs = list(map(lambda x: list(x), zip(*self.shm[:-2])))
        reward = np.squeeze(self.shm[-2], axis=-1)
        done = np.squeeze(self.shm[-1], axis=-1)
        return obs, reward, done

    def step(self, actions):
        for idx, env in enumerate(self.envs):
            env.step(actions[idx])
        return self._observe()

    def reset(self):
        for e in self.envs:
            e.reset()
        return self._observe()

    def reset_partial(self, id_list):
        for id in id_list:
            self.envs[id].reset()
        return self._observe_partial(id_list)

    def _observe_partial(self, id_list) :
        [self.envs[id].wait() for id in id_list]

        obs = list(map(lambda x: list(x), zip(*self.shm[:-2])))
        reward = np.squeeze(self.shm[-2], axis=-1)
        done = np.squeeze(self.shm[-1], axis=-1)
        return obs, reward, done

    def stop(self):
        for environment in self.envs :
            environment.stop()
        return

    def get_info(self):
        for e in self.envs :
            e.get_info()
        return self.wait()


def make_shared(n_envs, obs_space):
    shape = (n_envs, ) + obs_space.shape
    raw = RawArray(to_ctype(obs_space.dtype), int(np.prod(shape)))
    return np.frombuffer(raw, dtype=obs_space.dtype).reshape(shape)


def to_ctype(_type):
    types = {
        np.bool: ctypes.c_bool,
        np.int8: ctypes.c_byte,
        np.uint8: ctypes.c_ubyte,
        np.int32: ctypes.c_int32,
        np.int64: ctypes.c_longlong,
        np.uint64: ctypes.c_ulonglong,
        np.float32: ctypes.c_float,
        np.float64: ctypes.c_double,
    }
    if isinstance(_type, np.dtype):
        _type = _type.type
    return types[_type]
