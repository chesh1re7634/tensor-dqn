import gym
from ops import conv2d, linear
import tensorflow as tf
import numpy as np
import random

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

        with tf.variable_scope('target'):
            self.target_s_t = tf.placeholder('float32', \
                [None, self.screen_height, self.screen_width, self.history_length], name='target_s_t')

            self.target_l1, self.t_w['l1_w'], self.t_w['l1_b'] = \
                conv2d(self.target_s_t, 32, [8, 8], [4, 4], initializer, activation_fn, name='t_l1')
            self.target_l2, self.t_w['l2_w'], self.t_w['l2_b'] = \
                conv2d(self.target_l1, 64, [4, 4], [2, 2], initializer, activation_fn, name='t_l2')
            self.target_l3, self.t_w['l3_w'], self.t_w['l3_b'] = \
                conv2d(self.target_l2, 64, [3, 3], [1, 1], initializer, activation_fn, name='t_l3')

            target_l3_shape = self.target_l3.get_shape().as_list()
            self.target_l3_flat = tf.reshape(self.target_l3, [-1, reduce(lambda x, y: x * y, target_l3_shape[1:])])

            self.target_l4, self.t_w['l4_w'], self.t_w['l4_b'] = \
                linear(self.target_l3_flat, 512, activation_fn=activation_fn, name='t_l4')
            self.target_q, self.t_w['q_w'], self.t_w['q_b'] = \
                linear(self.target_l4, self.action_size, name='t_q')

        with tf.variable_scope('optimizer'):
            self.target_q_t = tf.placeholder('float32', [None], name='target_q_t')
            self.action = tf.placeholder('int64', [None], name='action')

            action_one_hot = tf.one_hot(self.action, self.action_size, 1.0, 0.0, name='action_one_hot')
            q_acted = tf.reduce_sum(self.q * action_one_hot, reduction_indices=1, name='q_acted')

            self.delta = self.target_q_t - q_acted
            clipped_delta = tf.clip_by_value(self.delta, self.min_delta, self.max_delta, name='clipped_delta')

            self.loss = tf.reduce_mean(tf.square(clipped_delta), name='loss')
            self.optm = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

        with tf.variable_scope("update_target"):
            self.t_w_input = {}
            self.t_w_assign_op = {}

            for name in self.w.keys():
                self.t_w_input[name] = tf.placeholder('float32', self.t_w[name].get_shape().as_list(), name=name)
                self.t_w_assign_op[name] = self.t_w[name].assign(self.t_w_input[name])

        self.sess.run(tf.initialize_all_variables())

    def predict(self, s_t):
        # todo

        action = self.sess.run(self.q_action, feed_dict={self.s_t: [s_t]})

        return action

    def observe(self, screen, reward, action, terminal):

        self.history.add(screen)
        self.memory.add(screen, reward, action, terminal)

        if self.step > self.learn_start:
            if self.step % self.train_frequency == self.train_frequency - 1:
                self.q_learning()

            if self.step % self.target_q_update_step == self.target_q_update_step - 1:
                self.update_target_q_network()

    def q_learning(self):
        if self.memory.total_count < self.history_length:
            return

        s_t, action, reward, s_t_plus1, terminal = self.memory.sample()

        t_q_plus_1 = self.sess.run(self.target_q, feed_dict={self.target_s_t: s_t})

        terminal = terminal + 0.
        max_q_t_plus_1 = np.max(t_q_plus_1, axis=1)
        target_q_t = (1. - terminal) * self.discount * max_q_t_plus_1 + reward

        _, q_t, loss = self.sess.run([self.optm, self.q, self.loss], feed_dict={
            self.target_q_t : target_q_t,
            self.action : action,
            self.s_t : s_t
        })
        #print loss
        return loss

    def update_target_q_network(self):
        for name in self.w.keys():
            self.t_w_assign_op[name].eval({self.t_w_input[name] : self.w[name].eval(session=self.sess)}, session=self.sess)

    def train(self):

        screen, reward, terminal = self.env.new_game(bRandom=True)

        for _ in range(self.history_length):
            self.history.add(screen)

        ep_rewards = []
        for self.step in range(self.train_epoch):

            action = self.predict(self.history.get())

            screen, reward, terminal = self.env.act(action)

            self.observe(screen, reward, action, terminal)

            ep_rewards.append(reward)

            if terminal:
                screen, reward, terminal = self.env.new_game(bRandom=True)

                print np.mean(ep_rewards)

                ep_rewards = []



if __name__ == "__main__":
    config= SimpleConfig
    env = SimpleGymEnvironment(config)
    sess = tf.Session()

    agent = Agent(config, env, sess)

    agent.train()









