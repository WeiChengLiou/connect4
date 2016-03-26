#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
from c4NNQ import encState, NNQ, Actable
from rules import rndstate, show
from traceback import print_exc
from pdb import set_trace


class Testc4NNQ(unittest.TestCase):
    def testtransform(self):
        state = rndstate(10)
        # show(state)
        s1 = encState(state)
        assert s1.shape == (1, 84), s1.shape

        def f(x, sgn):
            if x == 1:
                return sgn
            elif x == 0:
                return ' '
            else:
                raise Exception('Unknown state')

        s2 = u''.join([f(x, 'O') for x in s1[0, :42]])
        for x, y in zip(s2, state):
            if x == 'O':
                assert x == y, (x, y)
        s2 = u''.join([f(x, 'X') for x in s1[0, 42:]])
        for x, y in zip(s2, state):
            if x == 'X':
                assert x == y, (x, y)

    def testevalR(self):
        obj = NNQ('O', 'ANN1')
        assert obj.evalR('O') == 2
        assert obj.evalR('X') == -2
        assert obj.evalR('draw') == 1
        assert obj.evalR(None) == 0
        obj = NNQ('X', 'ANN1')
        assert obj.evalR('O') == -2
        assert obj.evalR('X') == 2
        assert obj.evalR('draw') == 1
        assert obj.evalR(None) == 0

