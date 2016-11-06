import numpy as np
import random


class ReplayMemory:
    def __init__(self, config):
        self.memory_size = config['memory_size']
        self.actions = np.empty(self.memory_size, dtype = np.uint8)
        self.rewards = np.empty(self.memory_size, dtype = np.integer)
        self.screens = np.empty((self.memory_size, config['screen_height'], config['screen_width'], config['channel_length']), dtype=np.float16)
        self.terminals = np.empty(self.memory_size, dtype=np.bool)

        self.dims = (config['screen_height'].config['screen_width'],config['channel_length'])
        self.batch_size = config['batch_size']

        self.current = 0
        self.total_count = 0

    def add(self, screen, reward, action, terminal):
        assert screen.shape == self.dims
        self.screens[self.current, ...] = screen
        self.rewards[self.current] = reward
        self.actions[self.current] = action
        self.terminals[self.terminals] = terminal

        self.total_count = max(self.total_count, self.current+1)
        self.current = (self.current+1) % self.memory_size

    def sample(self):
        assert self.total_count == self.memory_size

        indexes = []
        while len(indexes) < self.batch_size:
            ### to do what is history_length???
            index = random.randint(0, self.total_count-1)



