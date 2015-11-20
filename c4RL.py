#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pdb import set_trace
from RL import Action, State, Model, StateAction
from rules import actions
import cPickle
import gzip


def altkey(state):
    """Return key in contrary sign"""
    def newk(s):
        if s == ' ':
            return s
        elif s == 'O':
            return 'X'
        else:
            return 'O'
    return ''.join(map(newk, state))


class C4State(State):
    def __init__(self, state, win, sgn):
        super(C4State, self).__init__(state, win)
        self.sgn = sgn
        if not self.win:
            self.actions = self.allActions(state)

    @staticmethod
    def allActions(state):
        return [Action(a, state) for a in actions(state)]

    def terminate(self):
        return (self.win is not None)


class C4StateAction(StateAction):
    def __missing__(self, k):
        k1 = altkey(k)
        return self.__getitem__(k1)

    def check(self, objs, sgn=None):
        assert sgn is not None
        k = str(objs)
        if k not in self:
            objs.sgn = sgn
            self[k] = objs


class C4Model(Model):
    AllStates = C4StateAction(C4State)

    def __init__(self, **kwargs):
        super(C4Model, self).__init__(**kwargs)
        self.states = self.AllStates
        self.reset()

        if kwargs.get('algo'):
            self.algo = kwargs['algo'].upper()
            self.fupdate = eval('%supdate' % self.algo)
        else:
            """ For Random Player """
            self.algo = None
            self.epsilon = 1

    @staticmethod
    def save(algo):
        fi = algo + '.pkl'
        with gzip.open(fi, 'wb') as f:
            cPickle.dump(C4Model.AllStates, f)

    @staticmethod
    def load(algo):
        fi = algo + '.pkl'
        with gzip.open(fi, 'rb') as f:
            C4Model.AllStates = cPickle.load(f)

    def reset(self):
        self.SAR = None

    def check(self, s):
        self.states.check(s, self.sgn)
        return self.states[str(s)]

    def update(self, s1, action, r):
        if self.algo:
            self.fupdate(self, s1, action, r)


def SARSAupdate(obj, s1, action, r):
    # SARSA learning
    s1 = obj.check(s1)
    Q1 = obj.states.Q(s1, action)
    if s1.sgn != obj.sgn:
        r = -r

    if obj.SAR:
        Q0, r0 = obj.SAR
        if s1.terminate():
            r1 = obj.alpha * (r - Q0.score)
        else:
            r1 = obj.alpha * (r0 + obj.gamma * Q1.score - Q0.score)
        Q0.update(r1)
    obj.SAR = (Q1, r)

