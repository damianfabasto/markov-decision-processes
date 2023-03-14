from tqdm import tqdm
from gym.envs.toy_text import discrete

from time import time
import numpy as np
import pdb
from collections import deque
import pandas as pd
import mlgtsrc.a4plots as a4plots

def valueIteration(env:discrete.DiscreteEnv, theta, gamma, maxNumItersForPlot = None):
    """
    Implements value iteration, per Sutton-Barto 2nd eddition, page 83
    :param env: a gym discrete environemnt
    :param theta: the thresholds to stop iteration
    :param gamma: rate of discount of rewards
    :return:
    """

    # Initialize Values to be zeros. V is a vectorized version of the 8x8 grid
    V = np.zeros(shape = env.nS)
    numIter = 0
    start = time()

    while True:
        valueChangeDelta = 0.0
        numIter+=1
        for state in range(env.nS):
            value_before_update = V[state]
            # Look at all the possible states you can transition from your current state
            # And take the bext action
            bestNextValueAcrossActions = np.zeros(env.nA)
            for action in range(env.nA):
                nextValue = 0
                for prob, next_state, reward, terminal in env.P[state][action]:
                    # Take this action, and see what the corresponding value is
                    nextValue+= prob * (reward + gamma * V[next_state])
                bestNextValueAcrossActions[action] = nextValue
            bestNextValue = max(bestNextValueAcrossActions)
            # Update V(S)
            V[state] = bestNextValue
            valueChangeDelta = max(valueChangeDelta, np.abs(value_before_update - bestNextValue))
        if valueChangeDelta < theta:
            break
        if maxNumItersForPlot is not None:
            if numIter >= maxNumItersForPlot:
                break

    end = time()
    return V, numIter, end - start

def getPolicyAccordingToValue(V, env:discrete.DiscreteEnv, gamma):
    # Policy just follows best value
    # POlicy is matrix of states x actions
    policy = np.zeros([env.nS, env.nA])
    for state in range(env.nS):
        # Look at all the possible states you can transition from your current state
        # And take the bext action
        bestNextValueAcrossActions = np.zeros(env.nA)
        for action in range(env.nA):
            nextValue = 0
            for prob, next_state, reward, terminal in env.P[state][action]:
                # Take this action, and see what the corresponding value is
                nextValue += prob * (reward + gamma * V[next_state])
            bestNextValueAcrossActions[action] = nextValue
        bestAction = np.argmax(bestNextValueAcrossActions)
        policy[state][bestAction] = 1.0 #--> Probability of 1 of choosing the best action
    return policy


####################
# Policy Evaluation
###################

def policyEvaluation(env:discrete.DiscreteEnv, V, policy, theta, gamma):
    """

    :param env: a gym discrete environemnt
    :param policy: a matrix of size numStates x numActions, that specifies the probability of taking each of the possible actions, across all states
    :param theta: the thresholds to stop iteration
    :param gamma: rate of discount of rewards
    :return:
    """

    numIter = 0
    start = time()

    while True:
        valueChangeDelta = 0.0
        numIter+=1
        for state in range(env.nS):
            value_before_update = V[state]
            # Look at all the possible states you can transition from your current state
            #I need to take action with respect to policy
            policyAction = np.argmax(policy[state])
            nextValue = 0
            for prob, next_state, reward, terminal in env.P[state][policyAction]:
                # Take this action, and see what the corresponding value is
                nextValue+= prob * (reward + gamma * V[next_state])
            # Update V(S)
            V[state] = nextValue
            valueChangeDelta = max(valueChangeDelta, np.abs(value_before_update - V[state]))
        if valueChangeDelta < theta:
            break

    end = time()
    return V, numIter, end - start

def policyImprovement(env:discrete.DiscreteEnv, V, policy, gamma):

    policyStable = True
    numIter = 0
    start = time()
    for state in range(env.nS):
        oldAction = np.argmax(policy[state])
        # Look at all the possible states you can transition from your current state
        # And take the bext action
        bestNextValueAcrossActions = np.zeros(env.nA)
        for action in range(env.nA):
            nextValue = 0
            for prob, next_state, reward, terminal in env.P[state][action]:
                # Take this action, and see what the corresponding value is
                nextValue += prob * (reward + gamma * V[next_state])
            bestNextValueAcrossActions[action] = nextValue
        bestAction = np.argmax(bestNextValueAcrossActions)
        # Update the policy, so that it takes this best action only, for this state
        policy[state] = 0
        policy[state][bestAction] = 1
        if oldAction != bestAction:
            policyStable = False

    return policyStable, policy

def policyIteration(env:discrete.DiscreteEnv, theta, gamma, maxNumItersForPlot = None):
    """
    Performs the calls between Policy Evaluation and Policy Improvement
    """
    # Initialize the policy and V

    # Get a random policy
    np.random.seed(1)
    policy = np.random.rand(env.nS, env.nA)
    # Normalize, so that the sum across rows = 1, since you must take one action across all the possible actions
    policy /= policy.sum(axis = 1, keepdims=True)



    # Initialize V
    V = np.zeros(env.nS)
    numPolicyEvals = 0

    policyStable = False
    start = time()
    idx=0
    while not policyStable:
        # Do a round of policy Evaluation
        V, numIter, runTimeEval = policyEvaluation(env, V, policy, theta, gamma)
        # Do a round of policy improvement
        policyStable, policy = policyImprovement(env, V, policy, gamma)
        numPolicyEvals += numIter
        idx+=1
        if maxNumItersForPlot:
            if idx >= maxNumItersForPlot:
                break
    end = time()
    totalRunTime = end - start
    # The loop above will continue until policyStable = True
    return V, policy, numPolicyEvals, totalRunTime


#######################################
# Q-Learning and Variations
#######################################
from abc import ABC, abstractmethod
from copy import deepcopy

class EpsilonScheduler():

    """
    Implements various falvors of epsilon scheduling.
    """

    def __init__(self, params: dict):



        self.running_epsilon = params["epsilon_start"]

        self.epsilon_scheduler = params["epsilon_scheduler"]

        self.epsilon_executor_menu = {}
        self.epsilon_executor_menu["constantScaleRampDownWithMin"] = self.constantScaleRampDownWithMin
        self.epsilon_executor_menu["ramp_down"] = self.linearEpsilon
        self.epsilon_executor_menu["ramp_down_jump_zero"] = self.linearEpsilonJumpToZero
        self.epsilon_executor_menu["constant"] = self.constantEpsilon
        self.epsilon_executor_menu["constant_scale_ramp_down"] = self.constantScaleRampDown
        self.epsilon_executor_menu["constant_scale_ramp_down_jump_zero"] = self.constantScaleRampDownJumpZero
        self.epsilon_executor_menu["hyperbolic"] = self.hyperbolic
        self.epsilon_executor_menu["constant_jump_zero"] = self.constantJumpZeroEpsilon

        self.epsilon_executor = self.epsilon_executor_menu[params["epsilon_scheduler"]]

        self.params = params

    def updateRewards(self, reward):
        self.rewards.append(reward)

    # Various different implementations of epsilon schedule
    def linearEpsilon(self, episode_idx):
        episode_factor = self.params["nEpisodes"] // self.params["numEpisodesToEndEpsilonRampDown"]
        return max(min(1.0, self.params["minEpisodesFullExploration"] / (episode_factor * episode_idx + 1)),
                   self.params["min_epsilon"])

    def getMinNumEpisodesBeforeJumpToZero(self):
        if self.params["minNumEpisodesBeforeJumpToZero"] <= 1:
            episodesThresholdToJumpToZero = int(
                self.params["minNumEpisodesBeforeJumpToZero"] * self.params["nEpisodes"])
        else:
            episodesThresholdToJumpToZero = self.params["minNumEpisodesBeforeJumpToZero"]
        return episodesThresholdToJumpToZero


    def linearEpsilonJumpToZero(self, episode_idx):
        episodesThresholdToJumpToZero = self.getMinNumEpisodesBeforeJumpToZero()

        if episode_idx > episodesThresholdToJumpToZero:
            return 0.0
        return self.linearEpsilon(episode_idx)

    def constantEpsilon(self, episode_idx):
        return self.params["min_epsilon"]

    def constantJumpZeroEpsilon(self, episode_idx):
        episodesThresholdToJumpToZero = self.getMinNumEpisodesBeforeJumpToZero()
        if episode_idx > episodesThresholdToJumpToZero:
            return 0.0
        return self.params["min_epsilon"]

    def constantScaleRampDownJumpZero(self, episode_idx):
        episodesThresholdToJumpToZero = self.getMinNumEpisodesBeforeJumpToZero()
        if episode_idx > episodesThresholdToJumpToZero:
            return 0.0
        return max(self.params["epsilon_start"] * self.params["epsilon_decay"] ** episode_idx,
                   self.params["min_epsilon"])

    def constantScaleRampDown(self, episode_idx):
        new_epsilon = self.running_epsilon
        self.running_epsilon *= self.params["epsilon_decay"]
        return new_epsilon

    def constantScaleRampDownWithMin(self, episode_idx):
        new_epsilon = max(self.params["min_epsilon"], self.running_epsilon)
        self.running_epsilon *= self.params["epsilon_decay"]
        return new_epsilon

    def hyperbolic(self, episode_idx):
        episodesThresholdToJumpToZero = self.getMinNumEpisodesBeforeJumpToZero()
        if episode_idx > episodesThresholdToJumpToZero:
            return 0.0
        return 1.0 / (1.0 + episode_idx)

    def getEpsilon(self, episode_idx):
        return self.epsilon_executor(episode_idx)

class EnvironmentStats():

    """
    Returns rolling mean and std of the rewards, and checks if the environment has been solved
    """

    def __init__(self, numEpisodesWindow = 100):
        self.rewards = deque(maxlen = numEpisodesWindow)
        self.numEpisodesWindow = numEpisodesWindow
        self.numEpisodesWindow = numEpisodesWindow

    def insertReward(self, rewards:list):
        self.rewards.append(rewards)

    def reset(self):
        self.rewards = deque(maxlen = self.numEpisodesWindow)

    def meanRollingRewards(self):
        return np.mean(self.rewards)

    def episodeSolved(self, threshold = 200):
        windowed_rewards = np.mean(self.rewards)
        return windowed_rewards >= threshold

    def statistics(self, window):
        rolling = pd.Series(self.rewards).rolling(window=window)
        return {"mean": np.mean(self.rewards), "std": np.std(self.rewards), "rolling mean": rolling.mean(), "rolling std": rolling.std()}


class Q_Based_Learning(ABC):

    def __init__(self, env:discrete.DiscreteEnv, params:dict, seed = 1234):

        env.seed(seed)
        self.env = env
        self.params = params
        self.experimentName = params["ExperimentName"]
        # Initiallize Q
        # Initialize the Q value function
        # state x actions
        self.Q = np.zeros((env.nS, env.nA))

        # Initialize the seed
        np.random.seed(seed)



    def providePolicy(self):
        optimalPolicy = np.zeros([self.env.nS, self.env.nA])
        policy = np.argmax(self.Q, axis = 1)
        for idx, pol in enumerate(policy):
            optimalPolicy[idx, pol] = 1
        return optimalPolicy

    def provideValueFunction(self):
        V = np.max(self.Q, axis=1)
        return V

    def updateAlpha(self, episode):
        return max(self.alpha ,  np.power((episode+1), -0.8))
        #return self.alpha *  np.power((episode + 1), -0.45)


    @abstractmethod
    def learningUpdate(self, reward, current_state, current_action, next_sate, next_action):
        pass


    @abstractmethod
    def reset(self):
        """
        Implements any cleanin up or initialization that the algorithm may need
        :return:
        """
        pass


    def runOneEpisode(self, epsilon):

        # Restart the environment and get current state
        current_state = self.env.reset()
        # Restart the steps
        self.time = 0
        # Perform any reset, clearnup or initilization, that the learning algorithm may need
        self.reset()

        # Save actions taken
        actions_followed = []

        reached_terminalSate = False


        # Choose action based on epsilon greedy
        current_action = self.performEpsilonGreedy(epsilon)

        actions_followed.append(current_action)

        # Store the episode rewards
        episode_rewards = 0

        while not reached_terminalSate:
            # Perform action in the given state
            next_state, reward, reached_terminalSate, info = self.env.step(current_action)
            # Get the next action
            next_action = self.performEpsilonGreedy(epsilon)

            #print("Current state = {}, current action = {}, reward = {}, next action = {}".format(current_state, current_action, reward, next_action))
            #pdb.set_trace()
            episode_rewards += reward


            # Perform update
            self.learningUpdate(reward, current_state, current_action, next_state, next_action)
            #print(self.Q)
            # Update current_state and current_action
            current_state = next_state
            current_action = next_action

            actions_followed.append(current_action)

            self.time +=1

            if reached_terminalSate:
                break


        #return self.time, actions_followed
        return episode_rewards

    def runEpisodes(self, epsilonScheduler: EpsilonScheduler, stopEarlyIfNoProgress=False):

        time_start = time()

        training_rewards = []
        session_stats = EnvironmentStats()
        best_average_reward = -1e10

        #self.max_env_steps = self.env._max_episode_steps

        episode_intervals_to_save_results = [int(interval * self.params["nEpisodes"]) for interval in
                                             self.params["saveResultIntervals"]]

        pbar = tqdm(range(self.params["nEpisodes"] + 1))

        env_solved = False
        made_progress = True

        for episode_idx in pbar:
            running_time_before = time()
            current_state = self.env.reset()
            episode_rewards = 0
            step_count = 0
            # Get the new Epsilon
            new_epsilon = epsilonScheduler.getEpsilon(episode_idx)
            # Get the new alpha
            self.alpha  = self.updateAlpha(episode_idx)
            episode_rewards = self.runOneEpisode(new_epsilon)



            session_stats.insertReward(episode_rewards)  # save most recent score
            training_rewards.append(episode_rewards)  # save most recent score
            running_time_now = time()

            rolling_mean_reward = session_stats.meanRollingRewards()

            pbar.set_postfix({"Total Episode Rewards ": rolling_mean_reward, "Episode ": episode_idx,
                              "Running time": running_time_now - time_start,
                              "Interval time": running_time_now - running_time_before
                              })

            # Check to see if you need to save intermediate results
            if (episode_idx + 1) in episode_intervals_to_save_results:
                a4plots.saveQLearningPlots(self.params, episode_idx, self.experimentName, training_rewards, env_solved)

            #print('\rEpisode {}\tAverage Reward: {:.2f}'.format(episode_idx, rolling_mean_reward), end="")
            if episode_idx % self.params['print_every'] == 0:
                print(
                    '\rEpisode {}\tAverage Reward: {:.2f}\tRunning time: {:.2f}\tInternal time: {:.2f}'.format(
                        episode_idx, rolling_mean_reward, running_time_now - time_start,
                                                          running_time_now - running_time_before
                        ))

            env_solved = session_stats.episodeSolved()
            if env_solved:
                print('\nEnvironment solved in {:d} episodes!\tAverage Reward: {:.2f}\tTotal time: {:.2f}'.format(
                    episode_idx, rolling_mean_reward, running_time_now - time_start))
                break

            if stopEarlyIfNoProgress:
                # Update best average and see if we made any progress
                if best_average_reward < rolling_mean_reward and episode_idx % 20 == 0:
                    best_average_reward = rolling_mean_reward
                    made_progress = True
                else:
                    made_progress = False

                if not made_progress and episode_idx > 300 and episode_idx % 20 == 0:
                    print(
                        '\nEnvironment reward decreased - not making progress after {:d} episodes - Breaking\tAverage Reward: {:.2f}\t Best average Reward: {:.2f}\tTotal time: {:.2f}'.format(
                            episode_idx, rolling_mean_reward, best_average_reward, running_time_now - time_start))
                    test_run_rewards = None
                    break

        # Save the model and optimizer
        a4plots.saveQLearningPlots(self.params, episode_idx, self.experimentName, training_rewards, env_solved)
        # Save the parameters in a pickle file
        total_time = time() - time_start
        a4plots.saveResults(self, self.env, self.params, episode_idx, self.experimentName, training_rewards, env_solved, total_time)

        _, _ = self.runTestEpisodes()

        #return env_solved, made_progress, training_rewards, test_run_rewards

        return env_solved, made_progress

    def runOneTestEpisode(self):

        # Obtain the policy with respect to Q
        bestPolicy = self.providePolicy()

        # Restart the environment and get current state
        current_state = self.env.reset()
        # Restart the steps
        self.time = 0


        # Save actions taken
        actions_followed = []

        reached_terminalSate = False


        # Choose action based on epsilon greedy
        current_action = np.argmax(bestPolicy[current_state, :])

        actions_followed.append(current_action)

        # Store the episode rewards
        episode_rewards = 0
        while not reached_terminalSate:
            # Perform action in the given state
            next_state, reward, reached_terminalSate, info = self.env.step(current_action)
            # Get the next action
            next_action = np.argmax(bestPolicy[next_state, :])

            episode_rewards += reward

            # Update current_state and current_action
            current_state = next_state
            current_action = next_action

            actions_followed.append(current_action)

            self.time +=1



            if reached_terminalSate:
                break


        #return self.time, actions_followed
        return episode_rewards

    def runTestEpisodes(self):

        """
        Set epsilon to zero and see performance of learner
        :return:
        """

        time_start = time()

        training_rewards = []
        session_stats = EnvironmentStats()
        best_average_reward = -1e10

        #self.max_env_steps = self.env._max_episode_steps

        episode_intervals_to_save_results = [int(interval * self.params["nEpisodes"]) for interval in
                                             self.params["saveResultIntervals"]]

        pbar = tqdm(range(self.params["nEpisodes"] + 1))

        env_solved = False
        made_progress = True

        for episode_idx in pbar:
            running_time_before = time()
            current_state = self.env.reset()
            episode_rewards = 0
            step_count = 0
            episode_rewards = self.runOneTestEpisode()



            session_stats.insertReward(episode_rewards)  # save most recent score
            training_rewards.append(episode_rewards)  # save most recent score
            running_time_now = time()

            rolling_mean_reward = session_stats.meanRollingRewards()

            pbar.set_postfix({"Total Episode Rewards ": rolling_mean_reward, "Episode ": episode_idx,
                              "Running time": running_time_now - time_start,
                              "Interval time": running_time_now - running_time_before
                              })



            #print('\rEpisode {}\tAverage Reward: {:.2f}'.format(episode_idx, rolling_mean_reward), end="")
            if episode_idx % self.params['print_every'] == 0:
                print(
                    '\rTest Episode {}\tAverage Reward: {:.2f}\tRunning time: {:.2f}\tInternal time: {:.2f}'.format(
                        episode_idx, rolling_mean_reward, running_time_now - time_start,
                                                          running_time_now - running_time_before
                        ))

            env_solved = session_stats.episodeSolved()
            if env_solved:
                print('\nEnvironment solved in {:d} episodes!\tAverage Reward: {:.2f}\tTotal time: {:.2f}'.format(
                    episode_idx, rolling_mean_reward, running_time_now - time_start))
                break

        # Save the model and optimizer
        a4plots.saveQLearningPlots(self.params, episode_idx, "Test Run " + self.experimentName, training_rewards, env_solved)
        # Save the parameters in a pickle file
        total_time = time() - time_start
        a4plots.saveResults(self, self.env, self.params, episode_idx, "Test Run " + self.experimentName, training_rewards, env_solved, total_time)


        return env_solved, made_progress


class QLearning(Q_Based_Learning):

    def __init__(self, env:discrete.DiscreteEnv, params):

        super(__class__, self).__init__(env, params)


        # Count the number of steps in episode
        self.time = 0

        self.alpha = params["alpha"]

        self.gamma = params["gamma"]

    def greedy(self):
        # Get the values of Q at the current state
        values_ = self.Q[self.env.s, :]
        # Choose the best action
        best_action = np.argmax(values_)
        return best_action

    def performEpsilonGreedy(self, epsilon):
        # choose an action based on epsilon-greedy algorithm
        if np.random.rand() < epsilon:
            action = np.random.choice(range(self.env.nA))
        else:
            action = self.greedy()

        return action

    def learningUpdate(self, reward, current_state, current_action, next_sate, next_action):
        # if reward == 0:
        #     Q_Sprime_Max_A = 0
        # else:
        #     Q_Sprime_Max_A = np.max(self.Q[next_sate, :])

        Q_Sprime_Max_A = np.max(self.Q[next_sate, :])
        #print(self.alpha)
        # Q-learning update
        self.Q[current_state, current_action] += \
            self.alpha * (reward +  self.gamma * (Q_Sprime_Max_A - self.Q[current_state, current_action]))


    def updateEpsilon(self, newEpsilon):
        self.epsilon = newEpsilon


    def reset(self):
        """
        This algorithm does not need to do any cleanup or initialization for each episode
        :return:
        """
        return


# class SarsaOnPolicy(Q_Based_Learning):
#
#     def __init__(self, epsilon, alpha, gridWorld:GridWorld):
#
#         super(__class__, self).__init__(gridWorld)
#
#         self.epsilon = epsilon
#
#         # Count the number of steps in episode
#         self.time = 0
#
#         self.alpha = alpha
#
#     def greedy(self):
#         values_ = self.Q[self.env.current_loc[0], self.env.current_loc[1], :]
#         # Choose the best action
#         best_action = np.argmax(values_)
#         action = self.env.actions[best_action]
#         return action
#
#     def performEpsilonGreedy(self):
#         # choose an action based on epsilon-greedy algorithm
#         if np.random.rand() < self.epsilon:
#             action = np.random.choice(self.env.actions)
#         else:
#             action = self.greedy()
#
#         return action
#
#     def learningUpdate(self, reward, current_state, current_action, next_sate, next_action):
#         if reward == 0:
#             Q_Sprime_Aprime = 0
#         else:
#             Q_Sprime_Aprime = self.Q[next_sate[0], next_sate[1], self.env.actionToInteger(next_action)]
#
#         if current_state[0] == self.env.terminalState[0] and current_state[1] == self.env.terminalState[1]:
#             pdb.set_trace()
#
#         # Sarsa update
#         self.Q[current_state[0], current_state[1], self.env.actionToInteger(current_action)] += \
#             self.alpha * (reward +  Q_Sprime_Aprime -
#                      self.Q[current_state[0], current_state[1], self.env.actionToInteger(current_action)])
#
#
#     def updateEpsilon(self, newEpsilon):
#         self.epsilon = newEpsilon
#
#     def reset(self):
#         """
#         This algorithm does not need to do any cleanup or initialization for each episode
#         :return:
#         """
#         return
#
#
#
#
#
#
#
#
#
#
# class ExpectedQLearning(Q_Based_Learning):
#
#     def __init__(self, epsilon, alpha, gamma, policy, gridWorld:GridWorld):
#
#         super(__class__, self).__init__(gridWorld)
#
#         self.epsilon = epsilon
#
#         # Count the number of steps in episode
#         self.time = 0
#
#         self.alpha = alpha
#
#         self.gamma = gamma
#
#         self.policy = policy
#
#     def greedy(self):
#         values_ = self.Q[self.env.current_loc[0], self.env.current_loc[1], :]
#         # Choose the best action
#         best_action = np.argmax(values_)
#         action = self.env.actions[best_action]
#         return action
#
#     def performEpsilonGreedy(self):
#         # choose an action based on epsilon-greedy algorithm
#         if np.random.rand() < self.epsilon:
#             action = np.random.choice(self.env.actions)
#         else:
#             action = self.greedy()
#
#         return action
#
#     def learningUpdate(self, reward, current_state, current_action, next_sate, next_action):
#         if reward == 0:
#             Q_Sprime_Max_A = 0
#         else:
#             Q_Sprime_Max_A = np.sum([self.policy[a] * self.Q[next_sate[0], next_sate[1], self.env.actionToInteger(a)] for a in self.env.actions])
#
#
#         # Sarsa update
#         self.Q[current_state[0], current_state[1], self.env.actionToInteger(current_action)] += \
#             self.alpha * (reward +  Q_Sprime_Max_A -
#                      self.Q[current_state[0], current_state[1], self.env.actionToInteger(current_action)])
#
#
#     def updateEpsilon(self, newEpsilon):
#         self.epsilon = newEpsilon
#
#
#     def reset(self):
#         """
#         This algorithm does not need to do any cleanup or initialization for each episode
#         :return:
#         """
#         return
#
#
#
# class SarsaLambda(Q_Based_Learning):
#
#     def __init__(self, epsilon, lbda, alpha, gamma, gridWorld:GridWorld):
#
#         super(__class__, self).__init__(gridWorld)
#
#         self.epsilon = epsilon
#
#         # Count the number of steps in episode
#         self.time = 0
#
#         self.alpha = alpha
#
#         self.gamma = gamma
#
#         self.lbda = lbda
#
#         # Initialize the eligibility traces
#         self.reset()
#
#
#
#     def greedy(self):
#         values_ = self.Q[self.env.current_loc[0], self.env.current_loc[1], :]
#         # Choose the best action
#         best_action = np.argmax(values_)
#         action = self.env.actions[best_action]
#         return action
#
#     def performEpsilonGreedy(self):
#         # choose an action based on epsilon-greedy algorithm
#         if np.random.rand() < self.epsilon:
#             action = np.random.choice(self.env.actions)
#         else:
#             action = self.greedy()
#
#         return action
#
#     def reset(self):
#         """
#         This algorithm requires to initialize the eligibility traces to zero at the begining of each episode!
#         """
#         # state x actions
#         self.eligTraces = np.zeros((self.env.Nx, self.env.Ny, len(self.env.actions)))
#         return
#
#     def learningUpdate(self, reward, current_state, current_action, next_sate, next_action):
#         if reward == 0:
#             Q_Sprime_Aprime = 0
#         else:
#             Q_Sprime_Aprime = self.Q[next_sate[0], next_sate[1], self.env.actionToInteger(next_action)]
#
#         delta = reward + self.gamma * Q_Sprime_Aprime - self.Q[current_state[0], current_state[1], self.env.actionToInteger(current_action)]
#
#         # Update eligibility
#         self.eligTraces[current_state[0], current_state[1], self.env.actionToInteger(current_action)] += 1
#
#         # Now loop over all the actions and states
#         for x_idx in range(self.env.Nx):
#             for y_idx in range(self.env.Ny):
#                 for action in self.env.actions:
#                     action_idx = self.env.actionToInteger(action)
#                     self.Q[x_idx, y_idx, action_idx] += (self.alpha * delta *
#                                                                               self.eligTraces[x_idx, y_idx,action_idx])
#
#                     self.eligTraces[x_idx, y_idx, action_idx] *= self.gamma * self.lbda * self.eligTraces[x_idx, y_idx, action_idx]
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#


import itertools
import numpy as np
import sys

from collections import defaultdict





def make_epsilon_greedy_policy(Q, epsilon, decay, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    """

    def policy_fn(observation, episode):
        e_prime = epsilon * decay ** episode
        A = np.ones(nA, dtype=float) * e_prime / nA
        if np.all(np.isclose(Q[observation], np.zeros(nA))):
            best_action = np.random.randint(nA)
        else:
            best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - e_prime)
        return A

    return policy_fn

def make_exploration_function(Rplus, Ne):
    """
    Creates an "exploratory" policy (Exploration Function from AIMA chapter 21) based on a given Q-function and epsilon.
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        Nsa: A dictionary that maps state -> number of times an action has been taken
        Rplus: Large reward value to assign before iteration limit
        Ne: Minimum number of times that each action will be taken at each state
        nA: Number of actions in the environment.
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    """

    def exploration_fn(u, n):
        if n < Ne:
            return Rplus
        else:
            return u
    return np.vectorize(exploration_fn)


import matplotlib
import numpy as np
import pandas as pd
from collections import namedtuple
from matplotlib import pyplot as plt

EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])

def q_learning(env, method, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1, decay=1.0, Rplus=None, Ne=None):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy
    Args:
        env: OpenAI environment.
        method: ['greedy', 'explore'] whether to use a greedy or an explorative policy
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
        decay: exponential decay rate for epsilon
        Rplus: Optimistic reward given to unexplored states
        Ne: Minimum number of times that each action will be taken at each state
    Returns:
        A tuple (Q, episode_lengths).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Keeps track of useful statistics
    stats = EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))
    # Keeps track of how many times we've taken action a in state s
    Nsa = defaultdict(lambda: np.zeros(env.action_space.n))

    # The policy we're following
    if method == 'greedy':
        policy = make_epsilon_greedy_policy(Q, epsilon, decay, env.action_space.n)
        def get_next_action(state_, episode):
            action_probs = policy(state_, episode)
            return np.random.choice(np.arange(len(action_probs)), p=action_probs)
    elif method == 'explore':
        if not Rplus:
            Rplus = max(env.reward_range)
        if not Ne:
            Ne = 100
        exploration_fn = make_exploration_function(Rplus, Ne)
        done_exploring = False
        def get_next_action(state_, episode):
            exploration_values = exploration_fn(Q[state_], Nsa[state_])
            if np.allclose(exploration_values, exploration_values[0]):
                return np.random.randint(env.nA)
            else:
                return np.argmax(exploration_values)
    else:
        raise ValueError('Unsupported method type')

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()

        # Reset the environment and pick the first action
        state = env.reset()

        # One step in the environment
        # total_reward = 0.0
        for t in itertools.count():
            t += 1
            # Get an action based on the exploration function
            action = get_next_action(state, i_episode)
            next_state, reward, done, _ = env.step(action)

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            Nsa[state][action] += 1

            # TD Update
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta

            if method == 'explore' and not done_exploring:
                for arr in Nsa.values():
                    if not np.all(arr >= Ne):
                        break
                else:
                    done_exploring = True
                    print('All done with exploration at episode %i, step %i' % (i_episode, t))
            if done:
                break

            state = next_state

    final_policy = np.zeros((env.nS, env.nA))
    for state in range(env.nS):
        final_policy[state] = Q[state]
    return Q, stats, Nsa, final_policy


import os
import pickle
from mlgtsrc.envs.frozen import helpers as frozen_helpers

class ResultsAnalyzer():

    def __init__(self, src):
        self.src = src

    def summarizeResults(self, stockStateDim = None):
        bestResults = defaultdict(list)
        for file in os.listdir(self.src):
            if file.endswith("pkl") and "test" not in file.lower():
                res = pickle.load(open(os.path.join(self.src, file), "rb"))
                if stockStateDim:
                    # Check if the result is a stock environment with the given number of states
                    if res['env'].N!=stockStateDim:
                        continue

                bestResults['Ave Rewards'].append(np.mean(res['rewards_per_episode'][-100:]))
                bestResults['Rewards Improvement'].append(
                    np.mean(res['rewards_per_episode'][-100:]) - np.mean(res['rewards_per_episode'][:100]))
                bestResults['Exp'].append(res['configParams']['ExperimentName'])
                bestResults['Epsilon Scheduler'].append(res['configParams']["epsilon_scheduler"])
                bestResults['Run Time'].append(res['run time'])
                bestResults['Rewards'].append(res['rewards_per_episode'])

        bestResults = pd.DataFrame(bestResults)
        bestResults = bestResults.sort_values(by='Ave Rewards', ascending=False)
        return bestResults

    def loadExperiment(self, expName):
        for file in os.listdir(self.src):
            if file.endswith("pkl") and "test" not in file.lower() and "experiment {}".format(expName) in file.lower():
                res = pickle.load(open(os.path.join(self.src, file), "rb"))
                return res
        return None



    def plotAllParamsCombinations(self, bestResults, layout= 'horizontal', figsize = None):
        if layout =='horizontal':
            fig = plt.figure(figsize=(15, 5) if figsize is None else figsize)
        else:
            fig = plt.figure(figsize=(12, 8) if figsize is None else figsize)
        # Plot the average of the last 100 rewards
        if layout =='horizontal':
            ax = fig.add_subplot(1, 2, 1)
        else:
            ax = fig.add_subplot(2, 1, 1)

        bestResults.reset_index().loc[:, 'Ave Rewards'].plot(ax=ax, kind='bar')
        ax.set_title('Reward average of the last 100 episodes\n across all hyperparameters and epsilon decay schedules', size = 15)

        # Plot the best reward, per epsilon scheduler
        if layout =='horizontal':
            ax = fig.add_subplot(1, 2, 2)
        else:
            ax = fig.add_subplot(2, 1, 2)


        schedule = 'constantScaleRampDownWithMin'
        filtered = bestResults.loc[bestResults.loc[:, 'Epsilon Scheduler'] == schedule, :]
        bestResultForScheduler = filtered.iloc[0]

        ax.plot(bestResultForScheduler.loc["Rewards"], alpha=0.1, color='black')
        ax.plot(pd.Series(bestResultForScheduler.loc["Rewards"]).rolling(window=100).mean(), alpha=0.7, color='red',
                label=r'Constant scale ramp-down $\epsilon$ decay')

        schedule = 'hyperbolic'
        filtered = bestResults.loc[bestResults.loc[:, 'Epsilon Scheduler'] == schedule, :]
        bestResultForScheduler = filtered.iloc[0]

        ax.plot(bestResultForScheduler.loc["Rewards"], alpha=0.1, color='black')
        ax.plot(pd.Series(bestResultForScheduler.loc["Rewards"]).rolling(window=100).mean(), alpha=0.7, color='blue',
                label=r'Hyperbolic $\epsilon$ decay')

        ax.set_title('Rewards and Rolling Average rewards\n for the best two Epsilon decay schedules', size = 15)
        # ax.legend([None, r'Constant scale ramp-down $\epsilon$ decay', None, r'Hyperbolic $\epsilon$ decay'])
        ax.legend(loc = 'lower right', fontsize = 15)
        return fig, ax

    def plotValueForQMatrixFrozen(self, fileName, frozenEnvName, frozenShape):
        res = pickle.load(open(os.path.join(self.src, fileName), "rb"))
        Qclass = res['Qlearner']
        VQ = Qclass.provideValueFunction()
        fig = plt.figure(figsize=(15, 5))
        ax = fig.add_subplot(1, 2, 1)
        title = "Q-learning value function"
        frozen_helpers.visualize_value(VQ, frozenEnvName, frozenShape, ax=ax, removeColorBar=True, title = title)

        ax = fig.add_subplot(1, 2, 2)
        title = "Q-learning Policy"
        policyQ = Qclass.providePolicy
        frozen_helpers.visualize_policy(policyQ, frozenEnvName, frozenShape, title=title, ax = ax)

        return fig, ax
