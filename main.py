#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
from matplotlib import pylab as plt
import seaborn as sb
from rules import initState, chkwin, reward, action, chkwho, show, rndstate
from c4RL import C4Model, C4State
from pdb import set_trace


def getState(state):
    win = chkwin(state)
    return C4State(state, win)


def takeAction(obj, s, pos, sgn):
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

    while 1:
        action = player.predict(s)
        pos = action.name

        s1, score = takeAction(player, s, pos, sgn)
        player.update(s, pos, score)
        if s1.win:
            break

        sgn = 'O' if sgn == 'X' else 'X'
        player = [p for p in players if p is not player][0]
        s = s1

    for p in players:
        p.update(s1, None, score)
        p.reset()
    return s1.win, score


def Train(**kwargs):
    """ Run train and evaluation synchronously """
    nRun = kwargs['n']
    algo = kwargs['algo']
    epsilon = kwargs['epsilon']
    wins = [0, 0]

    TrainRun = [
        C4Model(sgn='O', algo=algo, epsilon=epsilon,
                gamma=0.5, alpha=0.5),
        C4Model(sgn='X', algo=algo, epsilon=epsilon,
                gamma=0.5, alpha=0.5),
        ]

    TestRun = [
        C4Model(sgn='O', algo=None, epsilon=1.,
                gamma=0.5, alpha=0.5),
        C4Model(sgn='X', algo=algo, epsilon=0.1,
                gamma=0.5, alpha=0.5),
        ]

    def run(i):
        win, score = game(TrainRun)
        win, score = game(TestRun, initState)
        if win == 'O':
            wins[0] += 1
        elif win == 'X':
            wins[1] += 1

        if (i+1) % 1000 == 0:
            print 'Finish %d runs, wins: %d - %d' % (i+1, wins[0], wins[1])

    map(run, xrange(nRun))


def ModelCompare(**kwargs):
    algos = ['SARSA', 'Q']
    rets = []
    for algo in algos:
        print algo
        players = [
            C4Model(sgn='O', epsilon=1.,
                    gamma=0.5, alpha=0.5),
            C4Model(sgn='X', algo=algo, epsilon=kwargs['epsilon'],
                    gamma=0.5, alpha=0.5),
            ]

        ret = []
        cnt = 0
        for i in xrange(kwargs['n']):
            win, score = game(players)
            if win == 'X':
                cnt += 1
            if (i+1) % 1000 == 0:
                print i+1, cnt
                ret.append(cnt)
                cnt = 0
        rets.append(ret)
        C4Model.clear()

    df = pd.DataFrame([pd.Series(x) for x in zip(*rets)])
    df.rename(columns={0: algos[0], 1: algos[1]}, inplace=True)
    df.plot()
    plt.savefig('ModelCompare.png')


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
    parser.add_argument('proc', help='Procedures: Train, ModelCompare')
    parser.add_argument('-n', help='number of runs (default: 50000)',
                        default=50000, type=int)
    parser.add_argument('-algo', help='algorithm (default: Q)',
                        default='Q')
    parser.add_argument('-epsilon', help='epsilon (default: 0.1)',
                        default=0.1, type=float)
    args = parser.parse_args()

    if args.proc == 'Train':
        Train(n=args.n, algo=args.algo)
    elif args.proc == 'ModelCompare':
        ModelCompare(n=args.n, epsilon=args.epsilon)

