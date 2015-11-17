#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools as it
import random


def actions():
    return None


def config(acts):
    global actions
    actions = acts


class Model(object):
    # basic player model

    def __init__(self, **kwargs):
        for k, v in kwargs.iteritems():
            self.__setattr__(k, v)

    def predict(self, state):
        acts = list(actions(state))
        if not acts:
            print state
            raise Exception('no possible actions!')
        return random.choice(acts)

    def update(self, state, pos, sgn, score):
        """"""

