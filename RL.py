#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pdb import set_trace
import random
import abc


class Action(object):
    # Action class
    __metaclass__ = abc.ABCMeta

    def __init__(self, action, state, score=0.):
        self.name = action
        self.state = state
        self.score = score

    def update(self, r):
        """Update score"""
        self.score += r

    def __str__(self):
        return '<Action: %s, %s, %f>' % (self.state, self.name, self.score)

    def __repr__(self):
        return self.__str__()


class State(object):
    # State Class
    __metaclass__ = abc.ABCMeta

    def __init__(self, state, win):
        self.state = state
        self.win = win
        self.actions = [Action(None, state)]

    def __repr__(self):
        return '<%s: %s, %d actions>' % (
            type(self), self.state, len(self.actions))

    def __str__(self):
        return self.state

    @abc.abstractmethod
    def terminate(self):
        """return is state is terminate"""

    @abc.abstractmethod
    def allActions(self):
        """return possible actions"""

    def best(self):
        """Return best action by action's score"""
        assert(len(self.actions) > 0)
        besta = [Action(None, None)]
        besta[0].score = None
        for a in self.actions:
            if a.score > besta[0].score:
                besta = [a]
            elif a.score == besta[0].score:
                besta.append(a)
        return random.choice(besta)

    def random(self):
        """Random select action"""
        return random.choice(self.actions)


class StateAction(abc.types.DictType):
    def __init__(self, factory):
        self.NewState = factory

    def check(self, objs):
        if objs not in self:
            self[str(objs)] = objs

    def Q(self, objs, action):
        for a in objs.actions:
            if a.name == action:
                return a
        if a.name is None:
            return a
        else:
            raise Exception('Unknown action', str(objs), action)


class Model(object):
    # Basic player model
    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):
        self.states = StateAction(State)
        for k, v in kwargs.iteritems():
            self.__setattr__(k, v)

    def check(self, objs):
        self.states.check(objs)
        return self.states[str(objs)]

    def predict(self, objs):
        objs = self.check(objs)
        if objs.terminate():
            return objs.actions[0]
        if random.random() <= self.epsilon:
            return objs.random()
        return objs.best()

    def best(self, objs):
        objs = self.check(objs)
        return objs.best()        

    @abc.abstractmethod
    def update(self, objs, action, r):
        """"""


