#!/usr/bin/env python
# -*- coding: utf-8 -*-

from traceback import print_exc
import itertools as it
import gzip
import cPickle
import tensorflow as tf
import numpy as np
from pdb import set_trace
from random import choice, random

# Implementation of Neural-Q
# Use tensorflow to construct neural network framework
# Need to define reward, loss, training process
# Unlike usual NN, the training and test process do at the same time.
# When we try to predict an action, we update our network when new state/reward arrives.
# Question:
#   1. Can we initialize multiple tf.Session() at the same time?
#      Ans: Seems yes
#   2. How to save/load?
#      Ans: Use cPickle to save, and tf.assign() to load
SEED = 34654


def transform(state, new_shape):
    # Transform C4State class into numpy array
    def f(s):
        if s == 'O':
            return 1
        elif s == 'X':
            return 2
        else:
            return 0
    return np.array(map(f, str(state)), dtype=np.float32).reshape(
        new_shape)


def rndAction(state):
    try:
        s1 = state.ravel()
        return choice([i for i in xrange(7) if s1[i] == 0])
    except:
        print_exc()
        set_trace()


class Simple(object):
    # Simple Player
    def __init__(self, sgn):
        self.sgn = sgn

    def predict(self, state):
        state = str(state)
        for i in xrange(6, -1, -1):
            if state[i] == ' ':
                return i

    def update(self, s1, r):
        """"""

    def reset(self):
        """"""


class Drunk(object):
    # Random Player
    def __init__(self, sgn):
        self.sgn = sgn

    def predict(self, state):
        return rndAction(transform(state, [1, 42]))

    def update(self, s1, r):
        """"""

    def reset(self):
        """"""


def ANN1(self):
    self.new_shape = (1, 42)
    self.state = tf.placeholder(tf.float32, shape=self.new_shape)
    self.fc1_weights = tf.Variable(
        tf.truncated_normal([42, 7], stddev=0.1, seed=SEED)
        )
    self.fc1_biases = tf.Variable(
        tf.zeros([7]))
    self.parms = ('fc1_weights', 'fc1_biases')

    model = tf.nn.softmax(
        tf.matmul(self.state, self.fc1_weights) + self.fc1_biases)
    self.model = model


def ANN2(self):
    self.new_shape = (1, 42)
    self.state = tf.placeholder(tf.float32, shape=self.new_shape)
    self.fc1_weights = tf.Variable(
        tf.truncated_normal([42, 16], stddev=0.1, seed=SEED)
        )
    self.fc1_biases = tf.Variable(
        tf.zeros([16]))
    self.fc2_weights = tf.Variable(
        tf.truncated_normal([16, 7], stddev=0.1, seed=SEED)
        )
    self.fc2_biases = tf.Variable(
        tf.zeros([7]))
    self.parms = ('fc1_weights', 'fc1_biases', 'fc2_weights', 'fc2_biases')

    model = tf.nn.relu(
        tf.matmul(self.state, self.fc1_weights) + self.fc1_biases)
    self.model = tf.nn.softmax(
        tf.matmul(model, self.fc2_weights) + self.fc2_biases)


def CNN(self):
    self.new_shape = (1, 6, 7, 1)
    self.state = tf.placeholder(tf.float32, shape=self.new_shape)

    self.conv1_weights = tf.Variable(
        tf.truncated_normal([4, 4, 1, 16], stddev=0.1, seed=SEED)
        )
    self.conv1_biases = tf.Variable(
        tf.zeros([16]))
    self.fc1_weights = tf.Variable(
        tf.truncated_normal([672, 7], stddev=0.1, seed=SEED)
        # tf.ones([N, self.ncol])
        )
    self.fc1_biases = tf.Variable(
        tf.zeros([7]))
    self.parms = ('conv1_weights', 'conv1_biases', 'fc1_weights', 'fc1_biases')

    conv = tf.nn.conv2d(
        self.state,
        self.conv1_weights,
        strides=[1, 1, 1, 1],
        padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, self.conv1_biases))
    relu_shape = relu.get_shape().as_list()
    reshape = tf.reshape(
        relu,
        [relu_shape[0], relu_shape[1] * relu_shape[2] * relu_shape[3]]
        )

    self.model = tf.nn.softmax(
        tf.matmul(reshape, self.fc1_weights) + self.fc1_biases)


def CNN2(self):
    self.new_shape = (1, 6, 7, 1)
    self.state = tf.placeholder(tf.float32, shape=self.new_shape)

    self.conv1_weights = tf.Variable(
        tf.truncated_normal([4, 4, 1, 16], stddev=0.1, seed=SEED)
        )
    self.conv1_biases = tf.Variable(
        tf.zeros([16]))
    self.fc1_weights = tf.Variable(
        tf.truncated_normal([192, 7], stddev=0.1, seed=SEED)
        )
    self.fc1_biases = tf.Variable(
        tf.zeros([7]))
    self.parms = ('conv1_weights', 'conv1_biases', 'fc1_weights', 'fc1_biases')

    conv = tf.nn.conv2d(
        self.state,
        self.conv1_weights,
        strides=[1, 1, 1, 1],
        padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, self.conv1_biases))
    pool = tf.nn.max_pool(
        relu,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME')
    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(
        pool,
        [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]]
        )

    self.model = tf.nn.softmax(
        tf.matmul(reshape, self.fc1_weights) + self.fc1_biases)


def CNN3(self):
    self.new_shape = (1, 6, 7, 1)
    self.state = tf.placeholder(tf.float32, shape=self.new_shape)

    self.conv1_weights = tf.Variable(
        tf.truncated_normal([4, 4, 1, 32], stddev=0.1, seed=SEED)
        )
    self.conv1_biases = tf.Variable(
        tf.zeros([32]))
    self.conv2_weights = tf.Variable(
        tf.truncated_normal([3, 4, 32, 64], stddev=0.1, seed=SEED)
        )
    self.conv2_biases = tf.Variable(
        tf.zeros([64]))
    self.fc1_weights = tf.Variable(
        tf.truncated_normal([256, 512], stddev=0.1, seed=SEED)
        )
    self.fc1_biases = tf.Variable(
        tf.zeros([512]))
    self.fc2_weights = tf.Variable(
        tf.truncated_normal([512, 7], stddev=0.1, seed=SEED)
        )
    self.fc2_biases = tf.Variable(
        tf.zeros([7]))
    self.parms = ('conv1_weights', 'conv1_biases', 'conv2_weights', 'conv2_biases', 'fc1_weights', 'fc1_biases', 'fc2_weights', 'fc2_biases')

    conv = tf.nn.conv2d(
        self.state,
        self.conv1_weights,
        strides=[1, 1, 1, 1],
        padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, self.conv1_biases))
    pool = tf.nn.max_pool(
        relu,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME')
    conv = tf.nn.conv2d(
        pool,
        self.conv2_weights,
        strides=[1, 1, 1, 1],
        padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, self.conv2_biases))
    pool = tf.nn.max_pool(
        relu,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME')
    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(
        pool,
        [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]]
        )
    hidden = tf.nn.relu(
        tf.matmul(reshape, self.fc1_weights) + self.fc1_biases)
    model = tf.nn.softmax(
        tf.matmul(hidden, self.fc2_weights) + self.fc2_biases)
    self.model = model


class NNQ(object):
    def __init__(self, sgn, algo, alpha=0.5, gamma=0.5, epsilon=0.1):
        self.sgn = sgn
        self.SAR = None  # tuple of (state, action)
        self.alpha = alpha
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon
        self.nrow = 6
        self.ncol = 7
        self.algo = algo

        self.Q = tf.placeholder(tf.float32, shape=[self.ncol])
        eval(algo)(self)

        f = lambda x: tf.nn.l2_loss(self.__getattribute__(x))
        loss = tf.reduce_mean(tf.square(self.model - self.Q))
        regularizer = sum(map(f, self.parms))
        self.loss = loss + 1e-4 * regularizer
        self.optimizer = \
            tf.train.GradientDescentOptimizer(0.5)\
              .minimize(self.loss)

        # self.prediction = tf.argmax(self.model, 1)

        # Before starting, initialize the variables.  We will 'run' this first.
        self.init = tf.initialize_all_variables()

        # Launch the graph.
        self.sess = tf.Session()
        self.sess.run(self.init)

    def reset(self):
        self.SAR = None

    def update(self, state, r0):
        # receive state class
        # needs preprocess
        if self.sgn == 'X':
            r0 = -r0
        state = transform(state, self.new_shape)
        self._update(state, r0)

    def predict(self, state):
        state = transform(state, self.new_shape)
        if random() > self.epsilon:
            act = self._action(state)
        else:
            act = rndAction(state)
        return act

    def _update(self, state, wl):
        if self.SAR is None:
            return
        s0, a0 = self.SAR
        r = self.reward(a0, wl)
        if wl == 0:
            # game not terminated
            r += self.gamma * self.maxR(state)
        r1 = (1 - self.alpha) * self.eval(s0).ravel() + self.alpha * r
        feed_dict = {self.state: s0, self.Q: r1}
        var_list = [self.optimizer, self.loss]
        _, l = self.sess.run(var_list, feed_dict)

    def _action(self, state):
        """
        Return best action given state and win/loss
        Like predict process in NN
        Best action: argmax( R(s, a) + gamma * max(R(s', a')) )
        """
        feed_dict = {self.state: state}
        var_list = [self.model]
        rewards, = self.sess.run(var_list, feed_dict)  # R(s, a)
        s1 = state.ravel()
        for i in np.argsort(rewards.ravel())[::-1]:
            if s1[i] == 0:
                act = i
                break
        self.SAR = state, act
        return act

    def reward(self, a, wl):
        """
        Reward function
        """
        r = np.zeros([self.ncol], dtype=np.float32)
        if wl != 0:
            r[a] = wl
        return r

    def maxR(self, state):
        return self.eval(state).max()

    def eval(self, state):
        r, = self.sess.run(
            [self.model],
            feed_dict={self.state: state})
        return r

    def getparm(self):
        li = []
        for parm in it.imap(self.__getattribute__, self.parms):
            li.append(self.sess.run(parm))
        return li

    def save(self):
        fi = 'NeuralQ.{}.pkl'.format(self.algo)
        cPickle.dump(self.getparm(), gzip.open(fi, 'wb'))

    def load(self):
        fi = 'NeuralQ.{}.pkl'.format(self.algo)
        ret = cPickle.load(gzip.open(fi, 'rb'))
        for vname, var in zip(self.parms, ret):
            self.__setattr__(vname, var)


def test():
    obj = NNQ()
    state = np.ones([1, 42], dtype=np.float32)
    print obj.receive(state, 0)


if __name__ == '__main__':
    test()

