#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
from pdb import set_trace
from RL import Action, State, Model, StateAction
from rules import chkwin, actions


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
    def __init__(self, state, sgn):
        super(C4State, self).__init__(state)
        self.sgn = sgn
        self.win = chkwin(state)
        if not self.win:
            self.actions = self.allActions(state)

    @staticmethod
    def allActions(state):
        return [Action(a, state) for a in actions(state)]

    def terminate(self):
        return (self.win is not None)

    def winner(self):
        return self.win


class C4StateAction(StateAction):
    def __missing__(self, k):
        k1 = altkey(k)
        return self.__getitem__(k1)

    def check(self, state, sgn=None):
        assert(sgn is not None)
        if state not in self:
            k1 = altkey(state)
            if k1 not in self:
                self[state] = self.NewState(state, sgn)


class C4Model(Model):
    AllStates = C4StateAction(C4State)

    def __init__(self, **kwargs):
        super(C4Model, self).__init__(**kwargs)
        self.states = self.AllStates
        self.reset()

    def update(self, state, action, r):
        # SARSA learning
        self.states.check(state, self.sgn)
        Q1 = self.states.Q(state, action)
        s1 = self.states[state]
        if s1.sgn != self.sgn:
            r = -r

        if self.SAR:
            Q0, r0 = self.SAR
            if s1.terminate():
                r1 = self.alpha * (r - Q0.score)
            else:
                r1 = self.alpha * (r0 + self.gamma * Q1.score - Q0.score)
            Q0.update(r1)
        self.SAR = (Q1, r)

    def check(self, state):
        self.states.check(state, self.sgn)


