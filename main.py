import gym
from ops import conv2d, linear
import tensorflow as tf
import numpy as np
from config import SimpleConfig
from environment import SimpleGymEnvironment

if __name__ == "__main__":

    config = SimpleConfig

    screen_height, screen_width, history_length = config.screen_height, config.screen_widht, config.history_length

    env = SimpleGymEnvironment(config)

    # gym constant
    action_size = env.action_size

    # memory
    memory_size = config.history_length
    screens = np.empty((memory_size, screen_height, screen_width, history_length), dtype=np.float16)

    # simulation & memory collectiong
    for i in range(memory_size):
        observation, reward, terminal, info = env.step(0)

        screens[i, ...] = observation

    # build graph & ops
    initializer = tf.truncated_normal_initializer(0, 0.02)
    activation_fn = tf.nn.relu
    w = {}
    t_w = {}

    with tf.variable_scope('prediction'):
        x = tf.placeholder('float32', [None, screen_height, screen_width, history_length], name='state')

        l1, w['l1_w'], w['l1_b'] = conv2d(x, 32, [8,8], [4,4], initializer, activation_fn, name='l1')
        l2, w['l2_w'], w['l2_b'] = conv2d(l1, 64, [4,4], [2,2], initializer, activation_fn, name='l2')
        l3, w['l3_w'], w['l3_b'] = conv2d(l2, 64, [3, 3], [1, 1], initializer, activation_fn, name='l3')

        l3_shape = l3.get_shape().as_list()
        l3_flat = tf.reshape(l3, [-1, reduce(lambda x, y: x*y, l3_shape[1:])])

        l4, w['l4_w'], w['l4_b'] = linear(l3_flat, 512, activation_fn=activation_fn,name='l4')
        q, w['q_w'], w['q_b'] = linear(l4, action_size, name='q')

        q_action = tf.argmax(q, dimension=1)

    with tf.variable_scope('target'):
        target_x = tf.placeholder('float32', [None, screen_height, screen_width, history_length], name='target_state')

        target_l1, t_w['l1_w'], t_w['l1_b'] = conv2d(target_x, 32, [8, 8], [4, 4], initializer, activation_fn, name='t_l1')
        target_l2, t_w['l2_w'], t_w['l2_b'] = conv2d(target_l1, 64, [4, 4], [2, 2], initializer, activation_fn, name='t_l2')
        target_l3, t_w['l3_w'], t_w['l3_b'] = conv2d(target_l2, 64, [3, 3], [1, 1], initializer, activation_fn, name='t_l3')

        l3_shape = target_l3.get_shape().as_list()
        target_l3_flat = tf.reshape(target_l3, [-1, reduce(lambda x, y: x+y, l3_shape[1:])])

        target_l4, t_w['l4_w'], t_w['l4_b'] = linear(target_l3_flat, 512, activation_fn=activation_fn, name='t_l4')
        target_q, t_w['q_w'], t_w['q_b'] = linear(target_l4, action_size, name='t_q')

        q_action = tf.argmax(target_q, dimension=1)

    sess = tf.Session()

    sess.run(tf.initialize_all_variables())

    output = sess.run(q_action,feed_dict={x:screens})

    print action_size
    print output



