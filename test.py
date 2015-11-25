#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools as it
from c4RL import C4Model, C4State
from rules import action, initState, show, chkwin, reward
from pdb import set_trace
from traceback import print_exc
import cPickle


def getState(poss):
    f = lambda sgn: 'O' if sgn == 'X' else 'X'
    sgn = 'O'
    s0 = initState
    for pos in map(int, poss):
        s0 = action(s0, pos, sgn)
        sgn = f(sgn)
    win = chkwin(s0)
    return C4State(s0, win, sgn)


def showSA(p):
    for s in p.states.values():
        yn = any([(a.score != 0) for a in s.actions])
        if yn:
            show(str(s))
            for a in s.actions:
                if a.score != 0:
                    print a


def resetAll(players):
    for p in players:
        p.reset()


class TestCase(object):
    def init(self, algos=None):
        if not algos:
            algos = ['SARSA', 'SARSA']
        return [
            C4Model(sgn='O', algo=algos[0], epsilon=0.1,
                    gamma=0.5, alpha=0.5),
            C4Model(sgn='X', algo=algos[1], epsilon=0.1,
                    gamma=0.5, alpha=0.5),
        ]

    def subtest(self, ps, actions):
        resetAll(ps)

        poss = map(int, actions)
        s0 = C4State(initState, None, 'O')
        ss = [s0]
        i = 0
        for j, pos in enumerate(poss):
            s = (getState(poss[:(j+1)]))
            ss.append(s)
            score = reward(s.win)
            ps[i].update(s0, pos, score)

            if s.win:
                ps[i].update(s, None, score)
                sX = ps[i].states[str(ss[-2])]
                assert any([(a.score != 0) for a in sX.actions])

                i = (i + 1) % 2
                ps[i].update(s, None, score)
                sX = ps[i].states[str(ss[-3])]
                assert any([(a.score != 0) for a in sX.actions])
                break

            i = (i + 1) % 2
            s0 = s

        resetAll(ps)

    def testStateAction(self):
        ps = self.init()
        s1 = getState('01')
        s2 = getState('10')
        ps[0].check(s1)
        ps[1].check(s2)
        assert ps[0].states[str(s1)].sgn == 'O'

        self.subtest(ps, '0101010')
        self.subtest(ps, '0101010')
        self.subtest(ps, '0101010')

        p = ps[0]
        s1 = (getState('0101'))
        p.update(s1, 0, reward(s1))
        s2 = (getState('010101'))
        p.update(s2, 0, reward(s2))
        resetAll(ps)

        assert p.states[str(s1)].actions[0].score != 0

        # Test for best case
        poss = map(int, '01010101')
        s0 = C4State(initState, None, 'O')
        ss = [s0]
        i = 0
        for j, pos in enumerate(poss):
            if (len(ss) > 5) and (ps[i].sgn == 'X'):
                act = ps[i].best(s0)
                assert act.name is not None
                assert pos != act.name, 'Best case not work'
                pos = int(act.name)
            s = getState(poss[:(j+1)])
            ss.append(s)
            score = reward(s.win)
            ps[i].update(s0, pos, score)

            if s.win:
                ps[i].update(s, None, score)
                sX = ps[i].states[str(ss[-2])]
                assert(any([(a.score != 0) for a in sX.actions]))

                i = (i + 1) % 2
                ps[i].update(s, None, score)
                sX = ps[i].states[str(ss[-3])]
                assert(any([(a.score != 0) for a in sX.actions]))
                break

            i = (i + 1) % 2
            s0 = s

    def testStateActionX(self):
        algos = ['SARSA', 'Q']

        for algo in algos:
            print algo
            ps = self.init(['stupid', algo])

            cnt = 0
            rets = []
            for k in xrange(1000):
                s0 = C4State(initState, None, 'O')
                i = 0
                poss = []
                for j in xrange(1000):
                    p = ps[i]
                    if p.sgn == 'X':
                        act = p.best(s0)
                    else:
                        act = p.predict(s0)
                    poss.append(int(act.name))
                    pos = poss[j]
                    s = getState(poss[:(j+1)])
                    score = reward(s.win)
                    p.update(s0, pos, score)

                    if s.win:
                        [p1.update(s, None, score) for p1 in ps]
                        break

                    s0 = s
                    i = (i + 1) % 2

                resetAll(ps)
                if s.win == 'X':
                    cnt += 1
                if (k+1) % 1000 == 0:
                    print k+1, cnt
                    rets.append((k+1, cnt))
            assert cnt > 750

            C4Model.clear()


if __name__ == '__main__':
    obj = TestCase()
    obj.testStateActionX()

