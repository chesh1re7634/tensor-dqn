from config import SimpleConfig

class BaseModel(object):
    def __init__(self, config):
        self.config = config

        attrs = [name for name in self.config.__dict__ if not name.startswith('_')]
        for attr_name in attrs:
            setattr(self, attr_name, getattr(self.config, attr_name))



if __name__ == "__main__":

    base = BaseModel(SimpleConfig)

    attrs = [name for name in base.__dict__ if not name.startswith('_')]

    for attr in attrs:
        print attr, "  " ,getattr(base, attr)
