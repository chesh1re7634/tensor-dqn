import ConfigParser

class SimpleConfig:

    screen_height       = 210
    screen_width        = 160
    channel_length      = 3
    batch_size          = 32
    history_length      = 4
    env_name            = 'Breakout-v0'

    display             = False

    random_start        = 30

    learn_start         = 20



    discount            = 0.99

    min_delta           = -1
    max_delta           = 1

    train_frequency = 10

    scale = 1000

    target_q_update_step    = 1 * scale
    train_epoch             = 5000 * scale
    memory_size             = 10 * scale






