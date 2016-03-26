#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from matplotlib import pylab as plt
import seaborn as sb
fidic = {
    'add1': 'output/result0.ReplNN.20000.csv',
    'addAll': 'output/result1.ReplNN.20000.csv',
    }
fidic1 = {
    '0-layer (10)': 'output/replNN_10.50000.csv',
    '1-layer (10)': 'output/replNN2_10.50000.csv',
    '1-layer (30)': 'output/replNN2_30.50000.csv',
    }
fidic2 = {
    '0-layer (10)': 'output/result.ReplNN_10.20000.csv',
    '0-layer (30)': 'output/result.ReplNN_30.20000.csv',
    }
fidic3 = {
    '1-layer (10)': 'output/result.ANN2_10.20000.csv',
    '1-layer (100)': 'output/result.ANN2_100.20000.csv',
    '1-layer (1000)': 'output/result.ANN2_1000.20000.csv',
    }


def show(fidic, picfi=None):
    dic = {}
    for k, fi in fidic.iteritems():
        df = pd.DataFrame.from_csv(fi)
        s = df['0'].astype(float) / 10
        dic[k] = s
    df = pd.DataFrame(dic)

    df.plot()
    if picfi:
        plt.savefig(picfi)
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    """"""
    show(fidic3, picfi='output/N_Batch.png')
    # show(fidic3)

