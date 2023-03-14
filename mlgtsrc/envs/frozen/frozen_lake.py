# Code from https://github.com/wesley-smith/CS7641-assignment-4/tree/f3d86e37504dda563f65b3267610a30f09d01c77
import numpy as np
import sys
from six import StringIO, b

from gym import utils
from gym.envs.toy_text import discrete

from .helpers import better_desc

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

actionMap = {0: "left",
             1: 'down',
             2: 'right',
             3: 'up'}

MAPS = {
    "4x4": [
        "SFFF",
        "FHFH",
        "HFFF",
        "HFFG"
    ],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFH",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFFHFFG"
    ],
}


class FrozenLakeEnv(discrete.DiscreteEnv):
    """
    Winter is here. You and your friends were tossing around a frisbee at the park
    when you made a wild throw that left the frisbee out in the middle of the lake.
    The water is mostly frozen, but there are a few holes where the ice has melted.
    If you step into one of those holes, you'll fall into the freezing water.
    At this time, there's an international frisbee shortage, so it's absolutely imperative that
    you navigate across the lake and retrieve the disc.
    However, the ice is slippery, so you won't always move in the direction you intend.
    The surface is described using a grid like the following
        SFFF
        FHFH
        FFFH
        HFFG
    S : starting point, safe
    F : frozen surface, safe
    H : hole, fall to your doom
    G : goal, where the frisbee is located
    The episode ends when you reach the goal or fall in a hole.
    You receive a reward of 1 if you reach the goal, and zero otherwise.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc=None, map_name="4x4", slip_rate=0.5, rewards=None):
        if desc is None and map_name is None:
            raise ValueError('Must provide either desc or map_name')
        elif desc is None:
            desc = MAPS[map_name]
        assert not slip_rate or 0.0 < slip_rate <= 1.0, 'Slip rate must be between 0.0 and 1.0'
        if rewards:
            assert isinstance(rewards, (list, tuple,)) and len(rewards) == 3, 'Rewards should be [living, hole, goal]'
            self.reward_range = (min(rewards), max(rewards))
        else:
            self.reward_range = (0, 1)
        self.desc = desc = np.asarray(desc, dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape

        nA = 4
        nS = nrow * ncol

        import pdb
        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()


        P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row * ncol + col

        def inc(row, col, a):
            if a == 0:  # left
                col = max(col - 1, 0)
            elif a == 1:  # down
                row = min(row + 1, nrow - 1)
            elif a == 2:  # right
                col = min(col + 1, ncol - 1)
            elif a == 3:  # up
                row = max(row - 1, 0)
            return (row, col)

        def reward(letter_):
            if rewards:
                if bytes(letter_) in b'SF':
                    return rewards[0]
                elif letter_ == b'H':
                    return rewards[1]
                elif letter_ == b'G':
                    return rewards[2]
                else:
                    raise ValueError('WTF')
            else:
                return float(letter_ == b'G')

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = P[s][a]
                    letter = desc[row, col]
                    if letter in b'GH':
                        li.append((1.0, s, 0, True))
                    else:
                        if slip_rate:
                            actions = [(a - 1) % 4, a, (a + 1) % 4]
                            probs = [slip_rate / 2.0, 1.0 - slip_rate, slip_rate / 2.0]
                            for b, p in zip(actions, probs):
                                newrow, newcol = inc(row, col, b)
                                newstate = to_s(newrow, newcol)
                                newletter = desc[newrow, newcol]
                                done = bytes(newletter) in b'GH'
                                rew = reward(newletter)
                                li.append((p, newstate, rew, done))
                        else:
                            newrow, newcol = inc(row, col, a)
                            newstate = to_s(newrow, newcol)
                            newletter = desc[newrow, newcol]
                            done = bytes(newletter) in b'GH'
                            rew = reward(newletter)
                            li.append((1.0, newstate, rew, done))

        super(FrozenLakeEnv, self).__init__(nS, nA, P, isd)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = better_desc(self.desc).tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Left", "Down", "Right", "Up"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc) + "\n")

        if mode != 'human':
            return outfile

    # def reset(self):
    #     self.s = 55
    #     self.lastaction = None
    #     return self.s