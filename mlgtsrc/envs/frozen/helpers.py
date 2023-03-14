# Code from https://github.com/wesley-smith/CS7641-assignment-4/tree/f3d86e37504dda563f65b3267610a30f09d01c77
import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns
import seaborn.apionly as sns
sns.set()

from .constants import TERM_STATE_MAP, GOAL_STATE_MAP

CMAP_COLOR = "Wistia"

def visualize_env(env, env_name, title=None, ax = None):
    shape = env.desc.shape
    M = shape[0]
    N = shape[1]
    arr = np.zeros(shape)
    for i in range(M):
        for j in range(N):
            if (N * i + j) in TERM_STATE_MAP[env_name]:
                arr[i, j] = 0.25
            elif (N * i + j) in GOAL_STATE_MAP[env_name]:
                arr[i, j] = 1.0
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(arr, cmap=CMAP_COLOR)
    ax.set_xticks(np.arange(M))
    ax.set_yticks(np.arange(N))
    ax.set_xticklabels(np.arange(M))
    ax.set_yticklabels(np.arange(N))
    ax.set_xticks(np.arange(-0.5, M, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N, 1), minor=True)
    ax.grid(False)
    ax.grid(which='minor', color='w', linewidth=2)

    for i in range(M):
        for j in range(N):
            if (i, j) == (0, 0):
                ax.text(j, i, 'S', ha='center', va='center', color='k', size=18)
            if (N * i + j) in TERM_STATE_MAP[env_name]:
                ax.text(j, i, 'x', ha='center', va='center', color='k', size=18)
            elif (N * i + j) in GOAL_STATE_MAP[env_name]:
                ax.text(j, i, '$', ha='center', va='center', color='k', size=18)
            if (N * i + j) == env.s:
                ax.text(j, i, 'P', ha='center', va='center', color='r', size=18)
            else:
                pass
    if ax is None:
        fig.tight_layout()
    if title:
        ax.set_title(title)
    if ax is None:
        return fig, ax


def visualize_policy(pi, env_name, shape, title=None, ax = None):
    M = shape[0]
    N = shape[1]
    actions = np.argmax(pi, axis=1).reshape(shape)
    mapping = {
        0: '<',
        1: 'v',
        2: '>',
        3: '^'
    }
    arr = np.zeros(shape)
    for i in range(M):
        for j in range(N):
            if (N * i + j) in TERM_STATE_MAP[env_name]:
                arr[i, j] = 0.25
            elif (N * i + j) in GOAL_STATE_MAP[env_name]:
                arr[i, j] = 1.0
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(arr, cmap=CMAP_COLOR)
    ax.set_xticks(np.arange(M))
    ax.set_yticks(np.arange(N))
    ax.set_xticklabels(np.arange(M))
    ax.set_yticklabels(np.arange(N))
    ax.set_xticks(np.arange(-0.5, M, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N, 1), minor=True)
    ax.grid(False)
    ax.grid(which='minor', color='w', linewidth=2)

    for i in range(M):
        for j in range(N):
            if (N * i + j) in TERM_STATE_MAP[env_name]:
                ax.text(j, i, 'x', ha='center', va='center', color='k', size=18)
            elif (N * i + j) in GOAL_STATE_MAP[env_name]:
                ax.text(j, i, '$', ha='center', va='center', color='k', size=18)
            else:
                ax.text(j, i, mapping[actions[i, j]], ha='center', va='center', color='k', size=18)
    # fig.tight_layout()
    if title:
        ax.set_title(title)
    if ax is None:
        return fig

def render_policy(pi, env_name, shape):
    actions = np.argmax(pi, axis=1)
    for index in TERM_STATE_MAP[env_name]:
        actions[index] = 999
    for index in GOAL_STATE_MAP[env_name]:
        actions[index] = 1000

    pi = np.reshape(actions, shape)

    mapping = {
        0: ' < ',
        1: ' v ',
        2: ' > ',
        3: ' ^ ',
        999: ' . ',
        1000: ' $ '
    }
    mapper = np.vectorize(lambda k: mapping[k])
    np.apply_along_axis(lambda row: print(' '.join(row)), axis=1, arr=mapper(pi))


def visualize_value(V, env_name, shape, title=None, ax = None, removeColorBar = False):
    M = shape[0]
    N = shape[1]
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))
    arr = V.reshape(shape)
    im = ax.imshow(arr, cmap=CMAP_COLOR)
    ax.set_xticks(np.arange(M))
    ax.set_yticks(np.arange(N))
    ax.set_xticklabels(np.arange(M))
    ax.set_yticklabels(np.arange(N))
    ax.set_xticks(np.arange(-0.5, M, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N, 1), minor=True)
    ax.grid(False)
    ax.grid(which='minor', color='w', linewidth=2)
    for i in range(M):
        for j in range(N):
            if (N * i + j) in TERM_STATE_MAP[env_name]:
                ax.text(j, i, 'x', ha='center', va='center', color='k')
            elif (N * i + j) in GOAL_STATE_MAP[env_name]:
                ax.text(j, i, '$', ha='center', va='center', color='k')
            else:
                ax.text(j, i, '%.2f' % (arr[i, j]), ha='center', va='center', color='k')
    # fig.tight_layout()
    if not removeColorBar:
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('State-value estimate', rotation=-90, va="bottom")
    if title:
        ax.set_title(title, size = 14)


def better_desc(desc):
    mapping = {
        b'S': b' S ',
        b'F': b' * ',
        b'H': b' O ',
        b'G': b' $ '
    }
    mapper = np.vectorize(lambda k: mapping[k])
    return mapper(desc)


