### References ###
# https://jeremykun.com/2013/11/08/adversarial-bandits-and-the-exp3-algorithm/
# https://arxiv.org/pdf/1506.03271.pdf

import numpy as np

def exp3_Bianchi(MAB, T, eta):
    """
    Exp3 algorithm as in "Regret Analysis of Stochastic and
    Nonstochastic Multi-armed Bandit Problems"
    by Bubeck and Bianchi

    MAB: list of arms from Arm class
    T: time horizon
    eta: list of length T of exploration parameters of Exp3
    """

    K = len(MAB) # number of arms

    # Initialize estimated cumulative rewards
    R = np.zeros(K)

    # History of rewards and weights
    reward_hist, weights_hist = [], []

    for t in range(T):

        # Compute probability distribution for chosing arm
        p = np.exp(eta[t] * (R - np.max(R))) # np.max(R) term for normalization
        p /= np.sum(p)

        # Draw arm index
        drawn_index = np.random.choice(a=K, p=p)

        # Draw corresponding reward
        drawn_reward = MAB[drawn_index].sample(t)

        # Compute estimated reward
        estimated_reward = drawn_reward / p[drawn_index]

        # Update estimated cumulative rewards
        R[drawn_index] += drawn_reward / p[drawn_index]

        # Save obtained reward and weights
        reward_hist.append(drawn_reward)
        weights_hist.append(np.exp(eta[t] * R))

    return reward_hist, weights_hist

def exp3P_Bianchi(MAB, T, eta, gamma, beta):
    """
    Exp3.P algorithm as in "Regret Analysis of Stochastic and
    Nonstochastic Multi-armed Bandit Problems"
    by Bubeck and Bianchi

    MAB: list of arms from Arm class
    T: time horizon
    eta: parameter used in the exponential
    gamma: random exploration probability
    beta: added bias
    """

    K = len(MAB) # number of arms

    # Initialize estimated cumulative rewards
    R = np.zeros(K)

    # History of rewards and weights
    reward_hist, weights_hist = [], []

    for t in range(T):

        # Compute probability distribution for chosing arm
        p = np.exp(eta * (R - np.max(R)))
        p /= np.sum(p) # does not work for me
        p = (1 - gamma) * p + gamma / K

        # Draw arm index
        drawn_index = np.random.choice(a=K, p=p)

        # Draw corresponding reward
        drawn_reward = MAB[drawn_index].sample(t)

        # Compute estimated reward
        estimated_reward = drawn_reward / p[drawn_index]

        # Update estimated cumulative rewards
        R[drawn_index] += estimated_reward # add reward for drawn arm
        R += beta/p # add beta for exploration

        # Save obtained reward and weights
        reward_hist.append(drawn_reward)
        weights_hist.append(np.exp(eta * R))

    return reward_hist, weights_hist


def exp3_IX(MAB, T, eta, gamma):
    """
    Exp3-IX algorithm as in Explore no more (G. Neu)
    MAB: list of arms from Arm class
    T: the time horizon
    eta: learning rate (>0)
    gamma: implicit exploration parameter (>0)
    """

    K = len(MAB) # number of arms
    W = np.ones(K) # initialize weights

    # History of rewards and weights
    reward_hist, weights_hist = [], []

    R = np.zeros(K) # Estimated cumulative rewards

    for t in range(T):

        # Set probabilities of drawing each arm
        p = np.exp(eta[t] * (R - np.max(R)))
        p = p / np.sum(p)

        # Draw arm index
        drawn_index = np.random.choice(a=K, p=p)

        # Draw corresponding reward
        drawn_reward = MAB[drawn_index].sample(t)

        # Compute estimated reward with implicit exploration
        estimated_reward = drawn_reward/(p[drawn_index] + gamma)

        R[drawn_index] += estimated_reward # Update estimated cumulative rewards

        # Save obtained reward and weights
        reward_hist.append(drawn_reward)
        weights_hist.append(np.exp(eta * R))

    return reward_hist, weights_hist
