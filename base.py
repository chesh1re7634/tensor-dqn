from config import SimpleConfig

class BaseModel(object):
    def __init__(self, config):
        self.config = config
        attrs = [name for name in self.config.__dict__ if not name.startswith('_')]

        for attr_name in attrs:
            setattr(self, attr_name, getattr(self.config, attr_name))

class Agent(BaseModel):
    def __init__(self, config):
        super(Agent, self).__init__(config)

if __name__ == "__main__":

    config = SimpleConfig

    agent = Agent(config)