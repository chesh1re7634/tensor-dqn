import ConfigParser

def readConfig(path):
    config_parser = ConfigParser.ConfigParser()
    config_parser.read(path)

    config = {}
    for section in config_parser.sections():
        for option in config_parser.options(section):
            config[option] = config_parser.get(section, option)

    return config

class SimpleConfig:
    memory_size         = 100
    screen_height       = 210
    screen_width        = 160
    channel_length      = 3
    batch_size          = 32
    history_length      = 4
    env_name            = 'SpaceInvaders-v0'

    display             = False

if __name__ == "__main__":
    print readConfig("config.ini")





