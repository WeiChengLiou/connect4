#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pdb import set_trace
from RL import Action, State, Model, StateAction
from rules import actions, chkwho
import cPickle
import gzip


class C4State(State):
    def __init__(self, state, win):
        super(C4State, self).__init__(state, win)
        # self.sgn = chkwho(self.state)
        if not self.win:
            self.actions = self.allActions(state)

    @staticmethod
    def allActions(state):
        return [Action(a, state) for a in actions(state)]

    def terminate(self):
        return (self.win is not None)


class C4StateAction(StateAction):
    def __missing__(self, k):
        raise Exception('missing key of %s' % k)

    def check(self, objs):
        k = str(objs)
        if k not in self:
            self[k] = objs


class C4Model(Model):
    AllStates = C4StateAction(C4State)

    def __init__(self, **kwargs):
        super(C4Model, self).__init__(**kwargs)
        self.states = self.AllStates
        self.reset()

        if kwargs.get('algo'):
            self.algo = kwargs['algo'].upper()
            if self.algo == 'STUPID':
                # Stupid algo: always pick the first feasible action
                self.predict = self.predict1
                self.algo = None
            else:
                self.fupdate = eval('%supdate' % self.algo)
        else:
            """ For Random Player """
            self.algo = None
            self.epsilon = 1

    @staticmethod
    def clear():
        C4Model.AllStates.clear()

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
        self.states.check(s)
        return self.states[str(s)]

    def update(self, s1, r):
        if self.algo:
            self.fupdate(self, s1, r)

    def predict1(self, objs):
        return objs.actions[0].name

    def predict(self, state):
        a = super(self, C4Model).predict(state)
        self.Q0 = self.states.Q(state, a)
        return a


def Qupdate(obj, s1, r):
    # Q learning
    s1 = obj.check(s1)
    if obj.sgn == 'X':
        r = -r

    if obj.SAR:
        Q0, r0 = obj.SAR
        if s1.terminate():
            r1 = obj.alpha * (r - Q0.score)
        else:
            Q1 = obj.best(s1)
            r1 = obj.alpha * (r0 + obj.gamma * Q1.score - Q0.score)
        Q0.update(r1)
    obj.SAR = (Q1, r)

