#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random


SIZE = [6, 7]
N = SIZE[0] * SIZE[1]
initState = ' ' * N


class ColException(Exception):
    def __init__(self, idx):
        self.idx = idx

    def __repr__(self):
        return 'ColException(%d)' % self.idx


def show(state):
    # Show board state
    print 'Board State'
    for r in xrange(SIZE[0]):
        i0 = (r) * SIZE[1]
        i1 = (r+1) * SIZE[1]
        print '||' + '|'.join(state[i0:i1]) + '||'


def posidx(r, c):
    # Position index
    assert c < SIZE[1], 'wrong column %s' % c
    assert r < SIZE[0], 'wrong row %s' % r
    return r * SIZE[1] + c


def action(state, c, sgn):
    # Put sgn at c column then return new state
    i0 = c
    if state[c] != ' ':
        raise ColException(c)

    for r in xrange(1, SIZE[0]):
        i = posidx(r, c)
        if (state[i] != ' '):
            break
        i0 = i
    return state[:i0] + sgn + state[(i0 + 1):]


def actions(state):
    # Return Possible actions
    for c in xrange(SIZE[1]):
        if state[c] == ' ':
            yield c


def rndstate(n):
    while 1:
        if n == -1:
            n = random.randint(0, 10)
        s0 = initState
        sgn = 'O'
        for i in xrange(n):
            pos = random.choice(list(actions(s0)))
            s0 = action(s0, pos, sgn)
            sgn = 'O' if sgn == 'X' else 'X'
        if not chkwin(s0):
            return s0


def newstate(state, r, c, sgn):
    # Put sgn at (r, c) then return new state
    n = posidx(r, c)
    assert(state[n] == ' ')
    return state[:n] + sgn + state[(n + 1):]


def chkDual(state):
    # Check dual step
    cnt1 = sum([(x == 'O') for x in state])
    cnt2 = sum([(x == 'X') for x in state])
    assert (cnt1 - cnt2) <= 1, 'Dual step'
    return True


def poswin():
    # Possible winning case
    # line for --
    for r in xrange(SIZE[0]):
        for c in xrange(SIZE[1] - 3):
            yield [posidx(r, c + i) for i in xrange(4)]
    # line fo |
    for c in xrange(SIZE[1]):
        for r in xrange(SIZE[0] - 3):
            yield [posidx(r + i, c) for i in xrange(4)]
    # line for \
    for r in xrange(SIZE[0] - 3):
        for c in xrange(SIZE[1] - 3):
            yield [posidx(r + i, c + i) for i in xrange(4)]
    # line for /
    for r in xrange(SIZE[0] - 3):
        for c in xrange(3, SIZE[1]):
            yield [posidx(r + i, c - i) for i in xrange(4)]


def listwin():
    # list cases of who's win
    for idxs in poswin():
        s = initState
        print idxs
        for idx in idxs:
            c = idx % SIZE[1]
            r = (idx - c) / SIZE[0]
            s = newstate(s, r, c, 'O')
        show(s)


def chkwin(state):
    # Check who is winner according to possible rules
    if all([(x != ' ') for x in state[:7]]):
        return 'draw'

    for idxs in poswin():
        cnt = sum([_reward(state[i]) for i in idxs])
        if cnt == 4:
            return 'O'
        elif cnt == -4:
            return 'X'
    return None


def _reward(sgn):
    # Return reward by sgn
    if sgn == 'O':
        return 1
    elif sgn == 'X':
        return -1
    else:
        return 0


def chkwho(state):
    # Get who's turn
    cnt = sum(map(_reward, state))
    if cnt == 1:
        return 'X'
    elif cnt == 0:
        return 'O'
    raise Exception('Wrong turn!!')


if __name__ == '__main__':
    # show(initState)
    # s1 = initState
    # for i in range(7):
    #     print 'stage', i
    #     sgn = 'O' if i % 2 == 0 else 'X'
    #     s1 = action(s1, 0, sgn)
    #     show(s1)
    #     chkDual(s1)

    # listwin()
    show(rndstate(5))

