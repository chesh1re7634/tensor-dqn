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

    target_q_update_step    = 100
    train_epoch             = 50000 * scale
    memory_size             = 20 * scale

    test_frequency = 10

    epsilon = 0.05





