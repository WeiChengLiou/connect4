#!/usr/bin/env python
# -*- coding: utf-8 -*-

from rules import initState, chkwin, reward, action, chkwho, show
from c4RL import C4Model
from pdb import set_trace


def takeAction(obj, state, pos, sgn):
    state1 = action(state, pos, sgn)
    win = chkwin(state1)
    score = reward(win)
    if sgn == 'X':
        score = -score
    return state1, win, score


def train():
    runs = 2000
    players = [
        C4Model(sgn='O', algo='sarsa', epsilon=0.1,
                gamma=0.5, alpha=0.5),
        C4Model(sgn='X', algo='sarsa', epsilon=0.1,
                gamma=0.5, alpha=0.5),
        ]

    def run(state):
        sgn = chkwho(state)
        player = [p for p in players if p.sgn == sgn][0]
        players.remove(player)

        while 1:
            players.append(player)
            player.check(state)

            action = player.predict(state)
            pos = action.name

            state1, win, score = takeAction(player, state, pos, sgn)
            player.update(state, pos, score)
            if win:
                state = state1
                player.update(state1, None, score)
                players[0].update(state1, None, score)
                break

            sgn = 'O' if sgn == 'X' else 'X'
            state = state1
            player = players.pop(0)

        # show(state)
        return initState

    reduce(lambda state, y: run(state), xrange(runs), initState)
    for p in players:
        cnt = 0
        for s in p.states.values():
            for a in s.actions:
                if a.score != 0:
                    cnt += 1
        print cnt
    return players


def showSA(p):
    for s in p.states.values():
        show(s.state)
        for a in s.actions:
            print a
            if a.score != 0:
                print '--------'


if __name__ == '__main__':
    """"""
    players = train()

