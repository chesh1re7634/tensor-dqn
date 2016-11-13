import numpy as np
import random
from config import SimpleConfig
from environment import SimpleGymEnvironment


class ReplayMemory:
    def __init__(self, config):
        self.memory_size = config.memory_size
        self.actions = np.empty(self.memory_size, dtype = np.uint8)
        self.rewards = np.empty(self.memory_size, dtype = np.integer)
        self.screens = np.empty((self.memory_size, config.screen_height, config.screen_width), dtype=np.float16)
        self.terminals = np.empty(self.memory_size, dtype=np.bool)

        self.history_length = config.history_length
        self.dims = (config.screen_height, config.screen_width)
        self.batch_size = config.batch_size

        self.current = 0
        self.total_count = 0

        self.prestates  = np.empty((self.batch_size, self.history_length) + self.dims, dtype = np.float16)
        self.poststates = np.empty((self.batch_size, self.history_length) + self.dims, dtype = np.float16)

    def add(self, screen, reward, action, terminal):
        assert screen.shape == self.dims
        self.screens[self.current, ...] = screen
        self.rewards[self.current] = reward
        self.actions[self.current] = action
        self.terminals[self.terminals] = terminal

        self.total_count = max(self.total_count, self.current+1)
        self.current = (self.current+1) % self.memory_size

    def getState(self, index):
        assert self.total_count > 0, 'replay memory is empty'
        assert 0 <= index and index < self.memory_size, 'not valid index'

        if index >= self.history_length -1:
            return self.screens[(index - (self.history_length - 1)):(index+1),...]
        else:
            indexes = [(index-i) % self.total_count for i in reversed(range(self.history_length))]
            return self.screens[indexes, ...]


    def sample(self):
        assert self.total_count > self.history_length

        indexes = []
        while len(indexes) < self.batch_size:
            ## sample one index
            while True:
                ### to do what is history_length???
                index = random.randint(self.history_length, self.total_count-1)
                ##  if current pointer in history
                if index - self.history_length < self.current and self.current <= index:
                    continue
                ## if history have episode end, then get new on
                if self.terminals[(index - self.history_length):index].any():
                    continue
                break

            ## len(indexs) <= 1, 2, 3 ...
            self.prestates[len(indexes), ...]   = self.getState(index-1)
            self.poststates[len(indexes), ...]  = self.getState(index)
            indexes.append(index)

        actions     = self.actions[indexes]
        rewards     = self.rewards[indexes]
        terminals   = self.terminals[indexes]

        return np.transpose(self.prestates, (0, 2, 3, 1)), actions, rewards, \
               np.transpose(self.poststates,  (0, 2, 3,1)), terminals

if __name__ == "__main__":
    config = SimpleConfig()

    env = SimpleGymEnvironment(config)

    env.new_game(bRandom=True)

    memory = ReplayMemory(config)

    for _ in range(2 * config.memory_size):
        action = random.randint(0, env.action_size)

        screen, reward, terminal = env.act(action)

        memory.add(screen, reward, action, reward)

    data = memory.sample()

    print len(data)
    print data[0].shape
    print data[1].shape
    print data[2].shape
    print data[3].shape
    print data[4].shape





