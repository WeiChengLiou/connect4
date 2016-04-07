#!/usr/bin/env python
# -*- coding: utf-8 -*-

from traceback import print_exc
import argparse
import pandas as pd
from rules import initState, chkwin, action, chkwho, show, rndstate, ColException
from RL import State, encState
from c4NNQ import NNQ, Simple, Random
from pdb import set_trace


def getState(state):
    win = chkwin(state)
    return State(state, win)


def takeAction(s, pos, sgn):
    state1 = action(str(s), pos, sgn)
    s1 = getState(state1)
    return s1


def game(players, state=None):
    def getp(sgn):
        for p in players:
            if p.sgn == sgn:
                return p

    if state is None:
        state = rndstate(-1)

    sgn = chkwho(state)
    s = getState(state)
    player = getp(sgn)
    score = player.evalR(s.win)
    i = players.index(player)

    for j in xrange(1000):
        if j == 999:
            raise Exception('infinite loop')

        try:
            snew = encState(str(s))
            pos = player.update(snew, score).predict(snew)
            s1 = takeAction(s, pos, sgn)

            if s1.win:
                break

            sgn = 'O' if sgn == 'X' else 'X'
            i = (i + 1) % 2
            player = players[i]
            score = player.evalR(s.win)
            s = s1
        except ColException:
            score = -100.

    for p in players:
        score = p.evalR(s1.win)
        p.setScore(score)
        p.update(encState(str(s1)), score)
        p.replay()
        p.reset()

    return s1.win, score


def train(**kwargs):
    """ Run train and evaluation synchronously """
    nRun = kwargs['n']
    wins = [0, 0, 0]

    Players = [
        NNQ(sgn='O', algo=kwargs['algo']),
        Random(sgn='X'),
        ]
    p = Players[0]
    li = []

    def run(i):
        win, score = game(Players, initState)
        if win == 'O':
            wins[0] += 1
        elif win == 'X':
            wins[1] += 1
        elif win == 'draw':
            wins[2] += 1
        else:
            raise Exception('Unknown game result')

        if (i+1) % 1000 == 0:
            # li.append(map(np.mean, p.getparm()))
            li.append(tuple(wins))
            print 'Finish %d runs, wins: %d - %d - %d' % (
                i+1, wins[0], wins[1], wins[2])
            # print (map(np.mean, p.getparm()))
            for i in range(3):
                wins[i] = 0

    map(run, xrange(nRun))
    if kwargs.get('save'):
        p.save()
        pd.DataFrame(li).to_csv('result.{0}.{1}.csv'.format(p.algo, nRun))
    return Players, li


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
    parser.add_argument('-n', help='number of runs (default: 100)',
                        default=100, type=int)
    parser.add_argument('-algo', default='ANN1', type=str)
    parser.add_argument('-save', default=1, type=int)
    args = parser.parse_args()

    if args.proc == 'train':
        args = vars(args)
        train(**args)

