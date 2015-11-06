#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import itertools as it


SIZE = [6, 7]
N = SIZE[0] * SIZE[1]
STATE = ' ' * N


def show(state):
    print 'Board State'
    for r in xrange(SIZE[0]):
        i0 = (r) * SIZE[1]
        i1 = (r+1) * SIZE[1]
        print '||' + '|'.join(state[i0:i1]) + '||'


def pos(r, c):
    assert(c < SIZE[1])
    assert(r < SIZE[0])
    return r * (1 + SIZE[0]) + c


def action(state, c, sgn):
    i0 = c
    if state[c] != ' ':
        raise Exception('No possible position!!')

    for r in xrange(1, SIZE[0]):
        i = pos(r, c)
        if (state[i] != ' '):
            break
        i0 = i
    return state[:i0] + sgn + state[(i0 + 1):]


def newstate(state, r, c, sgn):
    n = pos(r, c)
    print n
    assert(state[n] == ' ')
    return state[:n] + sgn + state[(n + 1):]


def state1(state, n, sgn):
    assert(state[n] == ' ')
    return state[:n] + sgn + state[(n + 1):]


def chkDual(state):
    cnt1 = sum([(x == 'O') for x in state])
    cnt2 = sum([(x == 'X') for x in state])
    assert (cnt1 - cnt2) <= 1, 'Dual step'
    return True


def poswin():
    # Possible winning case
    # line for --
    for r in xrange(SIZE[0]):
        for c in xrange(SIZE[1] - 3):
            yield [pos(r, c + i) for i in xrange(4)]
    # line fo |
    for c in xrange(SIZE[1]):
        for r in xrange(SIZE[0] - 3):
            yield [pos(r + i, c) for i in xrange(4)]
    # line for \
    for r in xrange(SIZE[0] - 3):
        for c in xrange(SIZE[1] - 3):
            yield [pos(r + i, c + i) for i in xrange(4)]
    # line for /
    for r in xrange(SIZE[0] - 3):
        for c in xrange(3, SIZE[1]):
            yield [pos(r + i, c - i) for i in xrange(4)]


def listwin():
    for idxs in poswin():
        s = STATE
        for idx in idxs:
            s = state1(s, idx, 'O')
        show(s)


def chkwin(state):
    for idxs in poswin():
        for sgn in ('O', 'X'):
            if all([(state[i] == sgn) for i in idxs]):
                return sgn
    return None


if __name__ == '__main__':
    # show(STATE)
    # s1 = STATE
    # for i in range(7):
    #     print 'stage', i
    #     sgn = 'O' if i % 2 == 0 else 'X'
    #     s1 = action(s1, 0, sgn)
    #     show(s1)
    #     chkDual(s1)

    listwin()

    idxs = it.islice(poswin(), 50, 51).next()
    for idx in idxs:
        STATE = state1(STATE, idx, 'X')
    show(STATE)
    print chkwin(STATE)

