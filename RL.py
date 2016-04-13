#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from rules import show


class State(object):
    # State Class

    def __init__(self, state, win):
        self.state = state
        self.win = win

    def state(self):
        return self.state


class StateAct(object):
    def __init__(self, state, act, score):
        self.state = state
        self.act = act
        self.score = score
        self.state1 = None

    def r(self):
        if self.score is None:
            return 0.
        else:
            return self.score


def encState0(state):
    """ encode original state into two boards """
    s1 = np.zeros((2, 42), dtype=np.float32)
    for i in xrange(42):
        if state[i] == 1:
            s1[0, i] = 1
        elif state[i] == 2:
            s1[1, i] = 1
    # return s1.ravel()
    return s1.reshape((1, 84))


def chkEmpty0(s1, i):
    return (s1[0, i] == 0) and (s1[0, i+42] == 0)


def encState(state):
    """ encode original state into two boards """
    s1 = np.zeros((1, 6, 7, 2), dtype=np.float32)
    for i in xrange(42):
        c = i % 7
        r = (i - c) / 7
        if state[i] == 1:
            s1[0, r, c, 0] = 1
        elif state[i] == 2:
            s1[0, r, c, 1] = 1
    return s1


def chkEmpty(s1, i):
    assert s1.shape[0] == 1, s1.shape
    c = i % 7
    r = (i - c) / 7
    return (s1[0, r, c, 0] == 0) and (s1[0, r, c, 1] == 0)


def show1(state):
    s = [0] * 42
    for i in range(42):
        if state[0, i] == 1:
            s[i] = 1
        elif state[0, i+42] == 1:
            s[i] = 2
    show(''.join(map(str, s)))
