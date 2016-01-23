#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from pdb import set_trace

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


class NNQ(object):
    def __init__(self, sgn, gamma=0.5):
        self.sgn = sgn
        self.state0, self.a0 = None, None
        self.gamma = gamma  # Discount factor
        self.nrow = 6
        self.ncol = 7
        N = self.nrow * self.ncol

        self.state = tf.placeholder(tf.float32, shape=[1, N])
        self.rewards = tf.placeholder(tf.float32, shape=[self.ncol])

        self.fc1_weights = tf.Variable(
            # tf.truncated_normal([42, 7], stddev=1.)
            tf.ones([N, self.ncol])
            )
        self.fc1_biases = tf.Variable(
            tf.zeros([self.ncol]))

        self.model = tf.matmul(self.state, self.fc1_weights) + self.fc1_biases

        self.loss = tf.reduce_mean(tf.square(self.model - self.rewards))
        self.optimizer = \
            tf.train.GradientDescentOptimizer(0.5)\
              .minimize(self.loss)

        self.prediction = tf.argmax(self.model, 1)

        # Before starting, initialize the variables.  We will 'run' this first.
        self.init = tf.initialize_all_variables()

        # Launch the graph.
        self.sess = tf.Session()
        self.sess.run(self.init)

    def reset(self):
        self.state0, self.a0 = None, None

    def receive(self, state, wl):
        self.update(state, wl)
        act = self.predict(state, wl)
        return act

    def update(self, state, wl):
        if self.a0 is None:
            return
        r = self.reward(self.a0, wl)
        if wl == 0:
            # game not terminated
            r += self.gamma * self.maxR(state)
        feed_dict = {self.state: self.state0, self.rewards: r}
        var_list = [self.optimizer, self.loss]
        _, l = self.sess.run(var_list, feed_dict)  # update first

    def predict(self, state, wl):
        """
        Return best action given state and win/loss
        Like predict process in NN
        Best action: argmax( R(s, a) + gamma * max(R(s', a')) )
        """
        feed_dict = {self.state: state}
        var_list = [self.prediction]
        act = self.sess.run(var_list, feed_dict)  # R(s, a)
        self.state0, self.a0 = state, act
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
        r, = self.sess.run(
            [self.model],
            feed_dict={self.state: state})
        return r.max()


def test():
    obj = QRL()
    state = np.ones([1, 42], dtype=np.float32)
    print obj.receive(state, 0)


if __name__ == '__main__':
    test()
