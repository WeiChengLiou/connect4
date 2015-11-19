#!/usr/bin/env python
# -*- coding: utf-8 -*-

from c4RL import C4Model
from rules import action, initState, show, chkwin, reward
from pdb import set_trace


def getState(poss):
    sgn = 'O'
    s0 = initState
    for pos in poss:
        s0 = action(s0, pos, sgn)
        sgn = 'O' if sgn == 'X' else 'X'
    return s0


class TestCase(object):
    def init(self):
        return [
            C4Model(sgn='O', algo='sarsa', epsilon=0.1,
                    gamma=0.5, alpha=0.5),
            C4Model(sgn='X', algo='sarsa', epsilon=0.1,
                    gamma=0.5, alpha=0.5),
        ]

    def testStateAction(self):
        ps = self.init()
        s1 = getState([0, 1])
        s2 = getState([1, 0])
        ps[0].check(s1)
        ps[1].check(s2)
        assert len(ps[0].states) == 1
        assert ps[0].states[s1].sgn == 'O'

        poss = '0101010'
        pos1 = []
        s0 = initState
        ss = [s0]
        i = 0
        for pos in map(int, poss):
            pos1.append(pos)
            s = getState(pos1)
            ss.append(s)
            win = chkwin(s)
            score = reward(win)
            ps[i].update(s0, pos, score)

            if win:
                ps[i].update(s, None, score)
                sX = ps[i].states[ss[-2]]
                assert any([(a.score != 0) for a in sX.actions])

                i = (i + 1) % 2
                ps[i].update(s, None, score)
                sX = ps[i].states[ss[-3]]
                assert any([(a.score != 0) for a in sX.actions])
                break

            i = (i + 1) % 2
            s0 = s

        # Test for best case
        poss = '01010101'
        pos1 = []
        s0 = initState
        ss = [s0]
        i = 0
        for pos in map(int, poss):
            if (len(ss) > 5) and (ps[i].sgn == 'X'):
                act = ps[i].predict(s0)
                assert pos != act.name, 'Best case not work'
                pos = int(act.name)
            pos1.append(pos)
            s = getState(pos1)
            ss.append(s)
            win = chkwin(s)
            score = reward(win)
            ps[i].update(s0, pos, score)

            if win:
                ps[i].update(s, None, score)
                sX = ps[i].states[ss[-2]]
                assert(any([(a.score != 0) for a in sX.actions]))

                i = (i + 1) % 2
                ps[i].update(s, None, score)
                sX = ps[i].states[ss[-3]]
                assert(any([(a.score != 0) for a in sX.actions]))
                break

            i = (i + 1) % 2
            s0 = s

if __name__ == '__main__':
    obj = TestCase()
    obj.testStateAction()
