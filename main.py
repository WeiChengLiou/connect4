#!/usr/bin/env python
# -*- coding: utf-8 -*-

from rules import STATE, chkwin, reward, action, chkwho, actions, show
import RL
from pdb import set_trace


def takeAction(state, pos, sgn):
    state1 = action(state, pos, sgn)
    win = chkwin(state1)
    score = reward(win)
    return state1, win, score


def train():
    runs = 5
    state0 = STATE
    players = {
        'O': RL.Model(sgn='O', algo='sarsa'),
        'X': RL.Model(sgn='X', algo='sarsa'),
        }

    def run(state):
        sgn = chkwho(state)
        player = players[sgn]

        while 1:
            pos = player.predict(state)
            state1, win, score = takeAction(state, pos, sgn)
            player.update(state, pos, sgn, score)
            if win is not None:
                break

            if sgn == 'O':
                sgn = 'X'
            else:
                sgn = 'O'
            state = state1
            player = players[sgn]

        return STATE

    reduce(lambda state, y: run(state), xrange(runs), state0)


if __name__ == '__main__':
    """"""
    RL.config(acts=actions)
    train()

