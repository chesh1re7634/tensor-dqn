import numpy as np

from config import SimpleConfig
from environment import SimpleGymEnvironment

class History:
    def __init__(self, config):
        self.batch_size, self.history_length, self.screen_height, self.screen_width = \
            config.batch_size, config.history_length, config.screen_height, config.screen_width

        self.history = np.zeros([self.history_length, self.screen_height, self.screen_width], dtype = np.float32)

    def add(self, screen):
        assert type(screen) is np.ndarray, "screen is not numpy.ndarray"
        assert screen.shape == (self.screen_height, self.screen_width), "%s != (%d, %d)" % (str(screen.shape), self.screen_height, self.screen_width)

        self.history[:-1] = self.history[1:]
        self.history[-1] = screen

    def get(self):
        return np.transpose(self.history, (1, 2, 0))

if __name__ == "__main__":
    config = SimpleConfig()

    env = SimpleGymEnvironment(config)
    env.new_game()

    history = History(config)

    for _ in range(10):
        screen, reward, terminal = env.act(0)
        history.add(screen)

    print history.get()




