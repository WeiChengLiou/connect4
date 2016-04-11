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
from RL import chkEmpty, StateAct, show1

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
N_BATCH = 10
N_REPSIZE = 2000


def rndAction(state):
    try:
        return choice([i for i in xrange(7) if chkEmpty(state, i)])
    except:
        print_exc()
        set_trace()


class Model(object):
    def evalR(self, wl):
        """ evalaute reward given state """
        if wl == 0:
            return 0.
        elif wl == -1:
            return 1
        elif wl == self.sgn:
            return 2
        else:
            return -2

    def setScore(self, score):
        """"""

    def replay(self):
        """"""


class Simple(Model):
    # Simple Player
    def __init__(self, sgn):
        self.sgn = sgn

    def predict(self, state):
        for i in xrange(6, -1, -1):
            if chkEmpty(state, i):
                return i

    def update(self, s1, r):
        return self

    def reset(self):
        """"""


class Random(Model):
    # Random Player
    def __init__(self, sgn):
        self.sgn = sgn

    def predict(self, state):
        return rndAction(state)

    def update(self, s1, r):
        return self

    def reset(self):
        """"""


class NNQ(Model):
    def __init__(self, sgn, algo, alpha=0.5, gamma=0.5, epsilon=0.1):
        self.sgn = sgn
        self.SARs = []  # List of (state, action)
        self.alpha = alpha
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon
        self.nrow = 6
        self.ncol = 7
        self.algo = algo
        self.RepSize = N_REPSIZE

        self.Q = tf.placeholder(tf.float32, shape=[N_BATCH, self.ncol])
        eval(algo)(self)  # Build up NN structure

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
        if len(self.SARs) > self.RepSize:
            self.SARs = self.SARs[N_BATCH:]

    def update(self, state, r0):
        # receive state class
        if self.SARs:
            s0 = self.SARs[-1]
            if s0.state1 is None:
                s0.state1 = state
                s0.score = r0
                self._update(self.SARs[-1:])
        return self

    def booking(self, SA):
        self.SARs.append(SA)

    def _update(self, SARs):
        try:
            S = np.vstack([sa.state for sa in SARs])
            r1, c1 = S.shape
            S1 = np.vstack([sa.state1 for sa in SARs])
            r0 = np.vstack([self.reward(sa.act, sa.r()) for sa in SARs])
            if r1 < N_BATCH:
                S = np.r_[S, np.zeros((N_BATCH-r1, 84))]
                S1 = np.r_[S1, np.zeros((N_BATCH-r1, 84))]
                r0 = np.r_[r0, np.zeros((N_BATCH-r1, 7))]

            r01 = self.maxR(S1) * self.gamma
            for i, sa in enumerate(SARs):
                r0[i, sa.act] += r01[i]

            R = (1 - self.alpha) * self.eval(S) + self.alpha * r0
            feed_dict = {self.state: S, self.Q: R}
            var_list = [self.optimizer, self.loss]
            _, l = self.sess.run(var_list, feed_dict)
        except:
            print_exc()
            set_trace()

    def predict(self, state):
        """ epsilon-greedy algorithm """
        if random() > self.epsilon:
            act = self._action(state)
        else:
            act = rndAction(state)
        return act

    def _action(self, state):
        """
        Return best action given state and win/loss
        Like predict process in NN
        Best action: argmax( R(s, a) + gamma * max(R(s', a')) )
        """
        rewards = self.eval(state)
        s1 = state.ravel()
        for i in np.argsort(rewards[0, :].ravel())[::-1]:
            if (s1[i] == 0) and (chkEmpty(state, i)):
                act = i
                break
        self.booking(StateAct(state, act, None))
        return act

    def replay(self):
        # R(t+1) = a * R'(St, At) + (1-a) * (R(St, At) + g * max_a(R'(St1, a)))
        N = len(self.SARs)
        if N < N_BATCH:
            return
        idx = np.random.choice(range(N), N_BATCH)
        # idx = np.array(range(N_BATCH))
        SARs = [self.SARs[i] for i in idx]
        self._update(SARs)

    def setScore(self, score):
        last = len(self.SARs) - 1
        for i in range(last, -1, -1):
            s = self.SARs[i]
            if s.score is None:
                s.score = score
                if i != last:
                    s.state1 = self.SARs[i+1].state

    def reward(self, a, r):
        """
        Reward function
        """
        rmat = np.zeros([self.ncol], dtype=np.float32)
        if r != 0:
            rmat[a] = r
        return rmat

    def rewardS(self, sa):
        return self.reward(sa.act, sa.score)

    def maxR(self, state):
        return self.eval(state).max(axis=1)

    def eval(self, state):
        assert type(state) == np.ndarray, type(state)
        r, c = state.shape
        if r < N_BATCH:
            state = np.r_[state, np.zeros((N_BATCH-r, 84))]
        try:
            r, = self.sess.run(
                [self.model],
                feed_dict={self.state: state})
            return r
        except:
            print_exc()
            set_trace()

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


def ANN1(self):
    self.new_shape = (N_BATCH, 84)
    self.state = tf.placeholder(tf.float32, shape=self.new_shape)
    self.fc1_weights = tf.Variable(
        tf.truncated_normal([84, 7], stddev=0.1, seed=SEED)
        )
    self.fc1_biases = tf.Variable(
        tf.zeros([N_BATCH, 7]))
    self.parms = ('fc1_weights', 'fc1_biases')

    model = tf.nn.softmax(
        tf.matmul(self.state, self.fc1_weights) + self.fc1_biases)
    self.model = model


def ANN2(self):
    self.new_shape = (N_BATCH, 84)
    self.state = tf.placeholder(tf.float32, shape=self.new_shape)
    self.fc1_weights = tf.Variable(
        tf.truncated_normal([84, 16], stddev=0.1, seed=SEED)
        )
    self.fc1_biases = tf.Variable(
        tf.zeros([N_BATCH, 16]))
    self.fc2_weights = tf.Variable(
        tf.truncated_normal([16, 7], stddev=0.1, seed=SEED)
        )
    self.fc2_biases = tf.Variable(
        tf.zeros([N_BATCH, 7]))
    self.parms = ('fc1_weights', 'fc1_biases', 'fc2_weights', 'fc2_biases')

    model = tf.nn.relu(
        tf.matmul(self.state, self.fc1_weights) + self.fc1_biases)
    self.model = tf.nn.softmax(
        tf.matmul(model, self.fc2_weights) + self.fc2_biases)


def CNN(self):
    self.new_shape = (N_BATCH, 6, 7, 1)
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


