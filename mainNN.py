#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import argparse
import pandas as pd
from matplotlib import pylab as plt
import seaborn as sb
from rules import initState, chkwin, reward, action, chkwho, show, rndstate, ColException
from c4RL import C4State
from c4NNQ import NNQ, Simple, Drunk
from pdb import set_trace


def getState(state):
    win = chkwin(state)
    return C4State(state, win)


def takeAction(s, pos, sgn):
    state1 = action(str(s), pos, sgn)
    s1 = getState(state1)
    score = reward(s1.win)
    return s1, score


def game(players, state=None):
    def getp(sgn):
        for p in players:
            if p.sgn == sgn:
                return p

    if state is None:
        state = rndstate(-1)

    sgn = chkwho(state)
    s = getState(state)
    score = reward(s.win)
    player = getp(sgn)
    i = players.index(player)

    for j in xrange(1000):
        if j == 999:
            raise Exception('infinite loop')

        try:
            player.update(s, score)
            pos = player.predict(s)
            s1, score = takeAction(s, pos, sgn)

            if s1.win:
                break

            sgn = 'O' if sgn == 'X' else 'X'
            i = (i + 1) % 2
            player = players[i]
            s = s1
        except ColException as e:
            print 'Bump column:', e.idx
            score = -100.
            set_trace()

    for p in players:
        p.update(s1, score)
        p.reset()

    # show(s1.state)
    return s1.win, score


def train(**kwargs):
    """ Run train and evaluation synchronously """
    nRun = kwargs['n']
    wins = [0, 0]

    Players = [
        NNQ(sgn='O'),
        Drunk(sgn='X'),
        ]
    p = Players[0]
    li = []

    def run(i):
        win, score = game(Players, None)
        li.append(map(np.mean, p.getparm()))
        if win == 'O':
            wins[0] += 1
        elif win == 'X':
            wins[1] += 1

        if (i+1) % 1000 == 0:
            print 'Finish %d runs, wins: %d - %d' % (i+1, wins[0], wins[1])
            print (map(np.mean, p.getparm()))
            for i in range(2):
                wins[i] = 0

    map(run, xrange(nRun))
    df = pd.DataFrame(li, columns=p.parms)
    df.plot()
    plt.show()
    # p.save()
    return Players


def showSA(p):
    for s in p.states.values():
        show(s.state)
        for a in s.actions:
            print a
            if a.score != 0:
                print '--------'


if __name__ == '__main__':
    """"""
    parser = argparse.ArgumentParser()
    parser.add_argument('proc', help='Procedures: train')
    parser.add_argument('-n', help='number of runs (default: 50000)',
                        default=100, type=int)
    args = parser.parse_args()

    if args.proc == 'train':
        args = vars(args)
        train(**args)

