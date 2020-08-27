from collections import deque
import numpy as np
import tensorflow as tf

class RingBuffer(object):
    def __init__(self, size):
        self.size = size
        self.data = deque(maxlen=size)

    def __len__(self):
        return self.length()

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length():
            raise KeyError()
        return self.data[idx]

    def append(self, v):
        self.data.append(v)

    def length(self):
        return len(self.data)

class ReplayMemory(object):
    def __init__(self, size):
        self.size = size
        
        self.states = RingBuffer(size)
        self.actions = RingBuffer(size)
        self.rewards = RingBuffer(size)
        self.next_states = RingBuffer(size)
        self.dones = RingBuffer(size)

    def append(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def sample(self, sample_size):
        # Get indices of samples for replay buffers
        indices = np.random.choice(range(len(self.dones)), size=sample_size)

        # Using list comprehension to sample from replay buffer
        state_sample = np.array([self.states[i] for i in indices])
        action_sample = [self.actions[i] for i in indices]
        reward_sample = [self.rewards[i] for i in indices]
        next_state_sample = np.array([self.next_states[i] for i in indices])
        done_sample = tf.convert_to_tensor(
            [float(self.dones[i]) for i in indices]
        )

        return state_sample, action_sample, reward_sample, next_state_sample, done_sample
