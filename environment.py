import gym
import random
import cv2
from config import SimpleConfig

class Environment(object):
    def __init__(self, config):
        self.env = gym.make(config.env_name)

        self.screen_width, self.screen_height = config.screen_width, config.screen_height

        self.display = config.display
        self.dims = (self.screen_width, self.screen_height)

        self._screen = None
        self.reward = 0.0
        self.terminal = True

        self.random_start = config.random_start

    def new_game(self, bRandom = False):
        self._screen = self.env.reset()
        if bRandom is True:
            for _ in xrange(random.randint(0, self.random_start-2)):
                self._step(0)
        self._step(0)
        self.render()
        return self.screen, self.reward, self.terminal

    def _step(self, action):
        self._screen, self.reward, self.terminal, _ = self.env.step(action)

    @property
    def screen(self):
        return cv2.resize(cv2.cvtColor(self._screen, cv2.COLOR_RGB2GRAY) / 255., self.dims)

    @property
    def action_size(self):
        return self.env.action_space.n

    @property
    def state(self):
        return self.screen, self.reward, self.terminal

    def render(self):
        if self.display:
            self.env.render()

class SimpleGymEnvironment(Environment):
    def __init__(self, config):
        super(SimpleGymEnvironment, self).__init__(config)

    def act(self, action):
        self._step(action)
        self.render()
        return self.state

if __name__ == "__main__":

    config = SimpleConfig()

    env = SimpleGymEnvironment(config)

    env.new_game()

    print env.act()

