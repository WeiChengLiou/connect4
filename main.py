#!/usr/bin/env python
# -*- coding: utf-8 -*-

from rules import initState, chkwin, reward, action, chkwho, show, rndstate
from c4RL import C4Model, C4State
from pdb import set_trace


def getState(state, sgn):
    win = chkwin(state)
    return C4State(state, win, sgn)


def takeAction(obj, s, pos, sgn):
    state1 = action(str(s), pos, sgn)
    s1 = getState(state1, sgn)
    score = reward(s1.win)
    return s1, score


def game(players, state=None):
    if state is None:
        state = rndstate(-1)

    sgn = chkwho(state)
    s = getState(state, sgn)
    score = reward(s.win)
    idx = 0 if players[0].sgn == s.sgn else 1
    player = players[idx]

    while 1:
        action = player.predict(s)
        pos = action.name

        s1, score = takeAction(player, s, pos, s.sgn)
        player.update(s, pos, score)
        if s1.win:
            s = s1
            for p in players:
                p.update(s1, None, score)
            return s1.win, score

        sgn = 'O' if sgn == 'X' else 'X'
        s = s1
        idx = (idx + 1) % 2
        player = players[idx]


def Train(TrainRun, TestRun):
    """ Run train and evaluation synchronously """
    nRun = 40000

    def run(i):
        # game(TrainRun)
        win, score = game(TestRun, initState)
        run.fitness += float(score)

        if i % 1000 == 0:
            print 'Finish %d runs, mean score: %1.4f' % (i, run.fitness/i)

    run.fitness = 0
    map(run, xrange(1, nRun+1))


def showSA(p):
    for s in p.states.values():
        show(s.state)
        for a in s.actions:
            print a
            if a.score != 0:
                print '--------'


if __name__ == '__main__':
    """"""
    C4Model.load('SARSA')
    TrainRun = [
        C4Model(sgn='O', algo='SARSA', epsilon=0.1,
                gamma=0.5, alpha=0.5),
        C4Model(sgn='X', algo='SARSA', epsilon=0.1,
                gamma=0.5, alpha=0.5),
        ]

    TestRun = [
        C4Model(sgn='O', epsilon=1.,
                gamma=0.5, alpha=0.5),
        C4Model(sgn='X', algo='SARSA', epsilon=0.1,
                gamma=0.5, alpha=0.5),
        ]

    Train(TrainRun, TestRun)
    # C4Model.save('SARSA')

