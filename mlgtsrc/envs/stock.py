import numpy as np
import pandas as pd
import sys
from six import StringIO, b
import pdb

from gym import utils
from gym.envs.toy_text import discrete
from collections import defaultdict


actionMap = {0: "buy",
             1: 'hold',
             2: 'sell'}




class StockEnv_Base(discrete.DiscreteEnv):
    """
    Implementation of a stock trading environment, where the stock evolves over time and the agent can decide to
    buy, hold or sell its share. The agent can only hold at most one share, and it cannot sell if it doesn't have a share
    and cannot buy if it already has one share

    The objective is to maximize the total wealth at the end of the evolution of the stock
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def preComputeParams(self, r, sigma, N, T):
        # Compute the up and down factors
        dt = T / N
        # up = np.exp((r-0.5*sigma**2)*dt + sigma*np.sqrt(dt))
        # dn = np.exp((r - 0.5 * sigma ** 2) * dt - sigma * np.sqrt(dt))
        up = np.exp(sigma * np.sqrt(dt))
        dn = np.exp(-sigma * np.sqrt(dt))
        p = (np.exp(r * dt) - dn) / (up - dn)
        return dt, up, dn, p

    def getVComponentsWithStock(self, numStock = 0):
        xyCoords = []
        Vcoords = []
        for xy, Vcoord in self.indexLookUp.items():
            if xy[2] == numStock:
                xyCoords.append([self.N-xy[1], xy[0]])
                Vcoords.append(Vcoord)
        return xyCoords, Vcoords

    def getVMask(self, V, xyCoords, Vcoords):
        Vmask = -1.0*np.inf*np.ones((2*self.N+1, self.N+1))
        for idx, (x, y) in enumerate(xyCoords):
            try:
                Vmask[x, y] = V[Vcoords[idx]]
            except:
                pdb.set_trace()
        return Vmask

    def getNumStates(self, N):
        N_ = N + 1
        return int(0.5 * N_ * (N_ + 1))

    def getNumStatesPriorTo(self, t):
        return sum([i for i in range(1, t + 1)])

    def fromStateToIndexOneDim(self, t, St):
        # Start with the number of states so far
        numStatesSoFar = self.getNumStatesPriorTo(t)
        index = numStatesSoFar + St + t
        return index

    def getStockValue(self, s):
        # return S0 * up**t * dn**(np.abs(s - t))
        SVal = self.S0
        if s < 0:
            SVal = self.S0 * self.dn ** (np.abs(s))
        elif s > 0:
            SVal = self.S0 * self.up ** (np.abs(s))
        else:
            SVal = self.S0
        return SVal

    def getPossibleRangesOfStockIndexForTime(self, t):
        return range(-t, t + 1, 2)

    def getInitialStateDistribution(self):
        # You start at the middle of the lattice
        isd = np.zeros(self.nS)
        t_now, St_now, numSharesBought = 0, 0, 0,
        current_state_index = self.getIndexFullState(t_now, St_now, numSharesBought)
        isd[current_state_index] = 1
        return isd

    def computeNumberOfSatesLoopUpTable(self):
        indexLookupTable = {}
        counter = 0
        # Initial state
        for t_now in range(self.N + 1):
            stockRangeNow = self.getPossibleRangesOfStockIndexForTime(t_now)
            for St_now in stockRangeNow:
                for numSharesBought in range(self.maxNumStockBuy):
                    indexLookupTable[(t_now, St_now, numSharesBought)] = counter
                    counter += 1
        return counter, indexLookupTable

    # Alternative: Construct a loop up table
    def getIndexFullState(self, t_now, St_now, numShares):
        fullIndex = self.indexLookUp[(t_now, St_now, numShares)]

        return fullIndex

    def transitions(self, P):
        # Loop over all possible times and space now
        for t_now in range(self.N):
            stockRangeNow = self.getPossibleRangesOfStockIndexForTime(t_now)
            for St_now in stockRangeNow:
                # Loop over all the possibe shares I could have bought
                for numSharesBought in range(self.maxNumStockBuy):

                    # Get full index for state
                    current_state_index = self.getIndexFullState(t_now, St_now, numSharesBought)

                    for a in range(self.nA):
                        # get the list of next states
                        if current_state_index not in P:
                            P[current_state_index] = defaultdict(list)
                        nextStatesTransitions = P[current_state_index][a]

                        astr = actionMap[a]

                        if astr == 'hold':
                            # Number of shares does not change, I just transition to the next state
                            reward = 0
                            self.addUpDownTransitions(nextStatesTransitions, t_now, St_now,numSharesBought, reward, lastAction='None')

                        elif astr == 'buy':
                            # Check if you don't have a stock already
                            if numSharesBought ==0:
                                # No Shares, you can then buy, and record the date and value at which you bought

                                # Reward is cost of the stock now
                                reward = -1.0 * self.getStockValue(St_now)
                                self.addUpDownTransitions(nextStatesTransitions, t_now, St_now,
                                                     numSharesBought = 1, reward = reward, lastAction=astr)

                            elif numSharesBought ==1:
                                # I already have one share, I cannot buy anymore
                                # Number of shares does not change, I just transition to the next state
                                reward = 0                                
                                self.addUpDownTransitions(nextStatesTransitions, t_now, St_now,
                                                     numSharesBought, reward, lastAction='None')
                            else:
                                raise Exception("Should not have happened!")

                        elif astr == 'sell':
                            # Check if you have a stock already
                            if numSharesBought ==0:
                                # No Shares, you can't sell
                                reward = 0#-10000
                                self.addUpDownTransitions(nextStatesTransitions, t_now, St_now,
                                                     numSharesBought = numSharesBought, reward = reward, lastAction='None')

                            elif numSharesBought ==1:
                                # I got a share already, so I can sell it
                                priceStockNow        = self.getStockValue(St_now)
                                reward = priceStockNow
                                self.addUpDownTransitions(nextStatesTransitions, t_now, St_now,
                                                     numSharesBought = 0, reward = reward, lastAction=astr)
                            else:
                                raise Exception("Should not have happened!")

        # Add the terminal states to P
        # In the terminal states, the reward is zero, and you stay in the terminal state

        t_now = self.N
        stockRangeNow = self.getPossibleRangesOfStockIndexForTime(t_now)
        for St_now in stockRangeNow:
            # Loop over all the possibe shares I could have bought
            for numSharesBought in range(self.maxNumStockBuy):

                # Get full index for state
                current_state_index = self.getIndexFullState(t_now, St_now, numSharesBought)

                for a in range(self.nA):
                    # get the list of next states
                    if current_state_index not in P:
                        P[current_state_index] = defaultdict(list)
                    P[current_state_index][a].append((1, current_state_index, 0, True))


    def __init__(self, r, sigma, N, T, S0, maxNumStockBuy=2):

        self.r, self.sigma, self.N, self.T, self.S0 = r, sigma, N, T, S0

        # Assume S0 is middle of the grid
        # So at most it can go from S0 to S0 * u**N or S0*d**N
        #Grid is t x stock x number shares

        self.maxNumStockBuy = maxNumStockBuy
        # Actions:
        # They can be buy, hold or sell
        self.nA = 3

        self.dt, self.up, self.dn, self.p = self.preComputeParams(r, sigma, N, T)


        # self.nS = getNumStates(N)         *      getNumStates(N)                      *        maxNumStockBuy
        self.nS, self.indexLookUp = self.computeNumberOfSatesLoopUpTable()





        isd = self.getInitialStateDistribution()

        print(f"Total number of states in one time-space dim = {self.getNumStates(N)}")
        print(f"Total number of states = {self.nS}")

        # Loop over all possible times and space now
        P = {}
        self.transitions(P)
        
        super(StockEnv_Base, self).__init__(self.nS, self.nA, P, isd)




    def render(self, policy):

        # Provide a simulated path, plus the actions taken
        dt, up, dn, p = self.preComputeParams(self.r, self.sigma, self.N, self.T)
        historyStockValues = []
        historyOfActions = []
        S = self.S0
        s_coord = 0
        numStockPurchased = 0
        dt, _, _, _ = self.preComputeParams(self.r, self.sigma, self.N, self.T)
        times = dt * np.arange(0, self.N, 1)


        for t_now in range(self.N):
            # See what the policy says we should do here
            indexForState = self.indexLookUp[(t_now, s_coord, numStockPurchased)]
            try:
                action = np.argmax(policy[indexForState])
            except:
                pdb.set_trace()
            actionStr = actionMap[action]
            if actionStr=='buy' and numStockPurchased==0:
                numStockPurchased += 1
                historyOfActions.append(actionStr)
            elif actionStr == 'sell'and numStockPurchased==1:
                numStockPurchased -= 1
                historyOfActions.append(actionStr)
            else:
                numStockPurchased += 0
                historyOfActions.append('hold')
            historyStockValues.append(S)

            # Make the transition to the next state:
            coinToss = np.random.rand()
            if p > coinToss:
                S *= up
                s_coord += 1
            else:
                S *= dn
                s_coord -= 1


        return pd.Series(historyStockValues, index = times), historyOfActions

 

    def renderRandomPolicy(self):

        # Provide a simulated path, plus the actions taken
        dt, up, dn, p = self.preComputeParams(self.r, self.sigma, self.N, self.T)
        historyStockValues = []
        historyOfActions = []
        S = self.S0
        s_coord = 0
        numStockPurchased = 0
        dt, _, _, _ = self.preComputeParams(self.r, self.sigma, self.N, self.T)
        times = dt * np.arange(0, self.N, 1)

        self.reset()

        indexToTimeSpace = {value: key for key, value, in self.indexLookUp.items()}

        rewards = []

        for t_now in range(self.N):
            # See what the policy says we should do here
            indexForState = self.indexLookUp[(t_now, s_coord, numStockPurchased)]

            action = self.action_space.sample()
            actionStr = actionMap[action]

            s, r, d, info = self.step(action)

            t_next, s_coord, numStocks = indexToTimeSpace[int(s)]

            Sval = self.getStockValue(int(s_coord))

            historyStockValues.append(Sval)

            historyOfActions.append(actionStr)
            rewards.append(r)


        return pd.Series(historyStockValues, index=times), historyOfActions, rewards




from collections import deque

class StockEnv_MeanReversion(StockEnv_Base):

    def __init__(self, r, sigma, N, T, S0, meanRevSpeed):
        self.meanRevSpeed = meanRevSpeed

        self.stock_hist = deque(maxlen=3)
        self.stock_hist.append(S0)


        super(__class__, self).__init__(r, sigma, N, T, S0)




    def computeScaledProb(self, St_now):
        # Transition Up
        ################
        # St_next = St_now + 1
        # Snow = getStockValue(St_next)
        Snow = self.getStockValue(St_now)
        self.stock_hist.append(Snow)
        Smean = np.mean(self.stock_hist)
        #print(Smean)
        scaleFactor = np.exp(-self.meanRevSpeed * (Smean/self.S0 - 1))
        probUp = self.p * scaleFactor
        if probUp>1.0:
            probUp = 1.0
        elif probUp<0:
            probUp = 0

        # Transition Down
        ################
        # St_next = St_now - 1
        # Snow = getStockValue(St_next)
        # scaleFactor = np.exp(-0.1 * (Snow - self.S0))
        probDown = 1 - probUp
        totalProb = probUp + probDown
        return probUp / totalProb, probDown / totalProb


    def addUpDownTransitions(self, nextStateTransitions, t_now, St_now, numSharesBought, reward, lastAction):
        # Number of shares does not change, I just transition to the next state
        t_next = t_now + 1

        # Get the prob of transitioning up and down
        prob_up, prob_dn = self.computeScaledProb(St_now)

        # Transition Up
        ################
        if lastAction =='sell':
            # If you sold, you will push the price down
            St_next = St_now - 1
            prob_up = 0.0
            prob_dn = 1.0
        else:
            St_next = St_now + 1

        # Get the full index of the next state
        next_state_index = self.getIndexFullState(t_next, St_next,
                                                  numSharesBought)

        finished = t_now == self.N
        nextStateTransitions.append([prob_up, next_state_index, reward, finished])

        # Transition Down
        ################
        if lastAction =='buy':
            # If you buy, you will push prices up
            St_next = St_now + 1
            prob_up = 1.0
            prob_dn = 0.0
        else:
            St_next = St_now - 1

        # Get the full index of the next state
        next_state_index = self.getIndexFullState(t_next, St_next,
                                                  numSharesBought)

        Snow = self.getStockValue(St_now)
        #print("Snow = {:2.2f}, Last Action = {}, Prob(Up)={:2.5f}, Prob(Down)={:2.5f}".format(Snow, lastAction, prob_up, prob_dn))

        finished = t_now == self.N
        nextStateTransitions.append([prob_dn, next_state_index, reward, finished])


class StockEnv_GBM(StockEnv_Base):

    def __init__(self, r, sigma, N, T, S0):
        super(__class__, self).__init__(r, sigma, N, T, S0)

    def addUpDownTransitions(self, nextStateTransitions, t_now, St_now, numSharesBought, reward, lastAction):
        # Number of shares does not change, I just transition to the next state
        t_next = t_now + 1
        # Transition Up
        ################
        St_next = St_now + 1
        prob = self.p
        # Get the full index of the next state
        next_state_index = self.getIndexFullState(t_next, St_next,
                                             numSharesBought)

        finished = t_now == self.N - 1
        nextStateTransitions.append([prob, next_state_index, reward, finished])

        # Transition Down
        ################
        St_next = St_now - 1
        prob = 1 -self.p
        # Get the full index of the next state
        next_state_index = self.getIndexFullState(t_next, St_next,
                                             numSharesBought)

        finished = t_now == self.N - 1
        nextStateTransitions.append([prob, next_state_index, reward, finished])



class StockEnv_OptionPricing(StockEnv_Base):

    def __init__(self, r, sigma, N, T, S0, strike):
        self.strike = strike
        super(__class__, self).__init__(r, sigma, N, T, S0)

    def addUpDownTransitions(self, nextStateTransitions, t_now, St_now, numSharesBought, reward):
        # Number of shares does not change, I just transition to the next state
        t_next = t_now + 1
        # Transition Up
        ################
        St_next = St_now + 1
        prob = self.p
        # Get the full index of the next state
        next_state_index = self.getIndexFullState(t_next, St_next,
                                             numSharesBought)

        finished = t_now == self.N
        nextStateTransitions.append([prob, next_state_index, reward, finished])

        # Transition Down
        ################
        St_next = St_now - 1
        prob = 1 -self.p
        # Get the full index of the next state
        next_state_index = self.getIndexFullState(t_next, St_next,
                                             numSharesBought)

        finished = t_now == self.N
        nextStateTransitions.append([prob, next_state_index, reward, finished])


    def transitions(self, P):
        # Loop over all possible times and space now
        for t_now in range(self.N):
            stockRangeNow = self.getPossibleRangesOfStockIndexForTime(t_now)
            for St_now in stockRangeNow:
                # Loop over all the possibe shares I could have bought
                for numSharesBought in range(self.maxNumStockBuy):

                    # Get full index for state
                    current_state_index = self.getIndexFullState(t_now, St_now, numSharesBought)

                    for a in range(self.nA):
                        # get the list of next states
                        if current_state_index not in P:
                            P[current_state_index] = defaultdict(list)
                        nextStatesTransitions = P[current_state_index][a]

                        # Actions do not modify now the structure of the payoff, since this is now a Markov Chain
                        if t_now == self.N-1:
                            priceStockNow = self.getStockValue(St_now)
                            reward = max(priceStockNow - self.strike, 0)
                        else:
                            reward = 0
                        self.addUpDownTransitions(nextStatesTransitions, t_now, St_now,
                                                      numSharesBought, reward)



        # Add the terminal states to P
        # In the terminal states, the reward is zero, and you stay in the terminal state

        t_now = self.N
        stockRangeNow = self.getPossibleRangesOfStockIndexForTime(t_now)
        for St_now in stockRangeNow:
            # Loop over all the possibe shares I could have bought
            for numSharesBought in range(self.maxNumStockBuy):

                # Get full index for state
                current_state_index = self.getIndexFullState(t_now, St_now, numSharesBought)

                for a in range(self.nA):
                    # get the list of next states
                    if current_state_index not in P:
                        P[current_state_index] = defaultdict(list)
                    P[current_state_index][a].append((1, current_state_index, 0, True))
