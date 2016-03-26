#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from rules import show


class State(object):
    # State Class

    def __init__(self, state, win):
        self.state = state
        self.win = win

    def __str__(self):
        return self.state


def encState(state):
    """ encode original state into two boards """
    s1 = np.zeros((2, 42), dtype=np.float32)
    for i in xrange(42):
        if state[i] == 'O':
            s1[0, i] = 1
        elif state[i] == 'X':
            s1[1, i] = 1
    # return s1.ravel()
    return s1.reshape((1, 84))


def chkEmpty(s1, i):
    return (s1[0, i] == 0) and (s1[0, i+42] == 0)


def show1(state):
    s = [' '] * 42
    for i in range(42):
        if state[0, i] == 1:
            s[i] = 'O'
        elif state[0, i+42] == 1:
            s[i] = 'X'
    show(''.join(s))


