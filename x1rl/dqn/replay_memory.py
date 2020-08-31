import random
import numpy as np
import tensorflow as tf

class ReplayMemory(object):
    def __init__(self, size):
        self.size = size
        
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def append(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        data = self._storage[0]
        ob_dtype = data[0].dtype
        ac_dtype = data[1].dtype
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        
        return np.array(obses_t, dtype=ob_dtype),   \
            np.array(actions, dtype=ac_dtype),      \
            np.array(rewards, dtype=np.float32),    \
            np.array(obses_tp1, dtype=ob_dtype),    \
            np.array(dones, dtype=np.float32)        
    
    def sample(self, batch_size):
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)
