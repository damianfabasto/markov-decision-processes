import matplotlib.pyplot as plt
import numpy as np
import seaborn.apionly as sns
sns.set()

import pdb


from mlgtsrc.envs import stock

CMAP_COLOR = "Wistia"

def plotValueFunction(V, env:stock.StockEnv_Base, numStock, ax = None, title = None, removeColorBar = False):

    xy, Vindices = env.getVComponentsWithStock(numStock=numStock)
    Vmask = env.getVMask(V, xy, Vindices)


    CMAP_COLOR = "Wistia"
    M = Vmask.shape[0]
    N = Vmask.shape[1]

    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 20))

    im = ax.imshow(Vmask, cmap=CMAP_COLOR)
    ax.set_xticks(np.arange(N))
    ax.set_yticks(np.arange(M))
    ax.set_xticklabels(np.arange(N))
    ax.set_yticklabels(np.arange(M))
    ax.set_xticks(np.arange(-0.5, N, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, M, 1), minor=True)
    ax.grid(False)
    ax.grid(which='minor', color='w', linewidth=2)

    for i in range(M):
        for j in range(N):
            ax.text(j, i, '%.2f' % (Vmask[i, j]), ha='center', va='center', color='k')
    # fig.tight_layout()
    if not removeColorBar:
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('State-value estimate', rotation=-90, va="bottom")
    if title:
       ax.set_title(title, size = 15)
    return ax

def visualizePolicyForPath(policy, env:stock.StockEnv_Base, ax=None, title = None):


    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 20))

    stockPath, actions = env.render(policy)
    print(actions)
    window_ticks = np.arange(len(stockPath))

    ax.plot(stockPath)

    short_ticks = []
    long_ticks = []
    hold_ticks = []
    for i, tick in enumerate(window_ticks):
        if actions[i] == 'buy':
            long_ticks.append(tick)
        elif actions[i] == 'hold':
            hold_ticks.append(tick)
        elif actions[i] == 'sell':
            short_ticks.append(tick)

    ax.plot(stockPath.index[short_ticks], stockPath.iloc[short_ticks], 'ro')
    ax.plot(stockPath.index[long_ticks], stockPath.iloc[long_ticks], 'go')

    # plt.suptitle(
    #     "Total Reward: %.6f" % self._total_reward + ' ~ ' +
    #     "Total Profit: %.6f" % self._total_profit
    # )
    return stockPath

# From: https://medium.com/swlh/calculating-option-premiums-using-the-black-scholes-model-in-python-e9ed227afbee
from math import log, sqrt, pi, exp
from scipy.stats import norm
from datetime import datetime, date
import numpy as np
import pandas as pd
from pandas import DataFrame

def d1(S,K,T,r,sigma):
    return(log(S/K)+(r+sigma**2/2.)*T)/(sigma*sqrt(T))
def d2(S,K,T,r,sigma):
    return d1(S,K,T,r,sigma)-sigma*sqrt(T)


def bs_call(S, K, T, r, sigma):
    return S * norm.cdf(d1(S, K, T, r, sigma)) - K * exp(-r * T) * norm.cdf(d2(S, K, T, r, sigma))


def bs_put(S, K, T, r, sigma):
    return K * exp(-r * T) - S + bs_call(S, K, T, r, sigma)