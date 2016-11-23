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

from utils import tmpLogDir

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

        with tf.variable_scope('summary'):
            scalar_summary_tags = ['episode_max_reward', 'episode_min_reward', 'episode_avg_reward', \
                                   'average_reward', 'average_loss', 'average_q']

            self.summary_placeholders = {}
            self.summary_ops = {}

            for tag in scalar_summary_tags:
                self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag)
                self.summary_ops[tag] = tf.scalar_summary("%s/%s" % (self.env_name, tag), self.summary_placeholders[tag])

            self.writer = tf.train.SummaryWriter(tmpLogDir(), self.sess.graph)

        self.sess.run(tf.initialize_all_variables())
        self.update_target_q_network()

    def predict(self, s_t):
        if random.random() <  self.epsilon:
            action = random.randrange(self.env.action_size)
        else:
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

        self.total_loss += loss
        self.total_q += q_t.mean()
        self.update_count += 1

    def update_target_q_network(self):
        for name in self.w.keys():
            self.t_w_assign_op[name].eval({self.t_w_input[name] : self.w[name].eval(session=self.sess)}, session=self.sess)

    def train(self):
        num_game, ep_reward = 0, 0.
        total_reward, self.total_loss, self.total_q = 0., 0., 0.
        ep_rewards = []

        screen, reward, terminal = self.env.new_game(bRandom=True)

        for _ in range(self.history_length):
            self.history.add(screen)

        for self.step in range(self.train_epoch):
            if self.step == self.learn_start:
                num_game, ep_reward = 0, 0.
                total_reward, self.total_loss, self.total_q = 0., 0., 0.
                ep_rewards = []

            action = self.predict(self.history.get())

            screen, reward, terminal = self.env.act(action)

            self.observe(screen, reward, action, terminal)

            if terminal:
                screen, reward, terminal = self.env.new_game(bRandom=True)

                num_game += 1
                ep_rewards.append(ep_reward)
                ep_reward = 0.
            else:
                ep_reward += reward

            total_reward += reward

            if self.step >= self.learn_start and \
                self.step % self.test_frequency == self.test_frequency - 1:

                avg_reward = total_reward / self.test_frequency
                avg_loss = self.total_loss / self.update_count
                avg_q = self.total_q / self.update_count

                try:
                    max_ep_reward = np.max(ep_rewards)
                    min_ep_reward = np.min(ep_rewards)
                    avg_ep_reward = np.mean(ep_rewards)
                except:
                    max_ep_reward, min_ep_reward, avg_ep_reward = 0, 0, 0

                print "ep_max_reward %.4f, ep_min_reward %.4f, ep_avg_reward %.4f, avg_reward, avg_loss, avg_q " % \
                      (max_ep_reward, min_ep_reward, avg_ep_reward, avg_reward, avg_loss, avg_q)

                self.inject_summary({
                    'episode_max_reward' : max_ep_reward,
                    'episode_min_reward' : min_ep_reward,
                    'episode_avg_reward' : avg_ep_reward,
                    'average_reward' : avg_reward,
                    'average_loss' : avg_loss,
                    'average_q' : avg_q
                }, self.step)

                num_game = 0
                total_reward = 0.
                self.total_loss = 0.
                self.total_q = 0.
                self.update_count = 0
                ep_rewards = []

    def inject_summary(self, tag_dict, step):
        summary_lists = self.sess.run([self.summary_ops[tag] for tag in self.tag_dict.keys()], \
            {self.summary_placeholders[tag]: value for tag, value in tag_dict.items()
        })

        for summary_str in summary_lists:
            self.wrtier.add_summary(summary_str, step)

if __name__ == "__main__":
    config= SimpleConfig
    env = SimpleGymEnvironment(config)
    sess = tf.Session()

    agent = Agent(config, env, sess)

    agent.train()









