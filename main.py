import gym
from ops import conv2d, linear
import tensorflow as tf
import numpy as np

from base import BaseModel
from config import SimpleConfig
from environment import SimpleGymEnvironment
from replay_memory import ReplayMemory
from history import History

class Agent(BaseModel):
    def __init__(self, config, environment, sess):
        super(Agent, self).__init__(config)

        # environment
        self.env = environment
        self.action_size = self.env.action_size

        # memory & history
        self.memory = ReplayMemory(self.config)
        self.history = History(self.config)

        # Session
        self.sess = sess

        self.build_dqn()

    def build_dqn(self):
        self.w = {}
        self.t_w = {}

        # build graph & ops
        initializer = tf.truncated_normal_initializer(0, 0.02)
        activation_fn = tf.nn.relu

        with tf.variable_scope('prediction'):
            self.s_t = tf.placeholder('float32', [None, self.screen_height, self.screen_width, self.history_length], name='s_t')

            self.l1, self.w['l1_w'], self.w['l1_b'] = conv2d(self.s_t, 32, [8, 8], [4, 4], initializer, activation_fn, name='l1')
            self.l2, self.w['l2_w'], self.w['l2_b'] = conv2d(self.l1, 64, [4, 4], [2, 2], initializer, activation_fn, name='l2')
            self.l3, self.w['l3_w'], self.w['l3_b'] = conv2d(self.l2, 64, [3, 3], [1, 1], initializer, activation_fn, name='l3')

            l3_shape = self.l3.get_shape().as_list()
            self.l3_flat = tf.reshape(self.l3, [-1, reduce(lambda x, y: x * y, l3_shape[1:])])

            self.l4, self.w['l4_w'], self.w['l4_b'] = linear(self.l3_flat, 512, activation_fn=activation_fn, name='l4')
            self.q, self.w['q_w'], self.w['q_b'] = linear(self.l4, self.action_size, name='q')

            self.q_action = tf.argmax(self.q, dimension=1)
        '''
        with tf.variable_scope('target'):
            self.target_s_t = tf.placeholder('float32', \
                [None, self.screen_height, self.screen_width, self.history_length], name='target_s_t')

            self.target_l1, self.t_w['l1_w'], self.t_w['l1_b'] = \
                conv2d(self.target_s_t, 32, [8, 8], [4, 4], initializer, activation_fn, name='t_l1')
            self.target_l2, self.t_w['l2_w'], self.t_w['l2_b'] = \
                conv2d(self.target_l1, 64, [4, 4], [2, 2], initializer, activation_fn, name='t_l2')
            self.target_l3, self.t_w['l3_w'], self.t_w['l3_b'] = \
                conv2d(self.target_l2, 64, [3, 3], [1, 1], initializer, activation_fn, name='t_l3')

            l3_shape = self.target_l3.get_shape().as_list()
            self.target_l3_flat = tf.reshape(self.target_l3, [-1, reduce(lambda x, y: x + y, l3_shape[1:])])

            self.target_l4, self.t_w['l4_w'], self.t_w['l4_b'] = \
                linear(self.target_l3_flat, 512, activation_fn=activation_fn, name='t_l4')
            self.target_q, self.t_w['q_w'], self.t_w['q_b'] = \
                linear(self.target_l4, self.action_size, name='t_q')

            self.q_action = tf.argmax(self.target_q, dimension=1)
        '''
        self.sess.run(tf.initialize_all_variables())

    def predict(self, s_t):
        # todo

        action = self.sess.run(self.q_action, feed_dict={self.s_t: [s_t]})

        return action


    def train(self):

        screen, reward, terminal = self.env.new_game()

        for _ in range(self.history_length):
            self.history.add(screen)

        predicted_action = self.predict(self.history.get())

        return predicted_action



if __name__ == "__main__":
    config= SimpleConfig
    env = SimpleGymEnvironment(config)
    sess = tf.Session()

    agent = Agent(config, env, sess)

    print agent.train()









