### References ###
# https://jeremykun.com/2013/11/08/adversarial-bandits-and-the-exp3-algorithm/
# https://arxiv.org/pdf/1506.03271.pdf

import numpy as np

def exp3_Bianchi(MAB, T, eta, gamma=0):
    """
    Exp3 algorithm as in "Regret Analysis of Stochastic and
    Nonstochastic Multi-armed Bandit Problems"
    by Bubeck and Bianchi

    MAB: list of arms from Arm class
    T: time horizon
    eta: list of length T of exploration parameters of Exp3
    """

    if not isinstance(eta, list) or len(eta)<T:
        eta = [eta for _ in range(T)]

    K = len(MAB) # number of arms

    # Initialize estimated cumulative rewards
    R = np.zeros(K)

    # History of rewards, weights and probability distribution
    reward_hist, weights_hist, prob_hist = np.zeros(T), [], []

    for t in range(T):

        # Compute probability distribution for chosing arm
        p = np.exp(eta[t] * (R - np.max(R))) # np.max(R) term for normalization
        p /= np.sum(p)
        p = (1 - gamma) * p + gamma / K

        # Draw arm index
        drawn_index = np.random.choice(a=K, p=p)

        # Draw corresponding reward
        drawn_reward = MAB[drawn_index].sample(t)

        # Compute estimated reward
        estimated_reward = drawn_reward / p[drawn_index]

        # Update estimated cumulative rewards
        R[drawn_index] += drawn_reward / p[drawn_index]

        # Save obtained reward and weights
        reward_hist[t] = drawn_reward
        weights_hist.append(np.exp(eta[t] * R))
        prob_hist.append(p)

    return reward_hist, weights_hist, prob_hist

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

    # History of rewards, weights and probability distribution
    reward_hist, weights_hist, prob_hist = np.zeros(T), [], []

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
        reward_hist[t] = drawn_reward
        weights_hist.append(np.exp(eta * R))
        prob_hist.append(p)

    return reward_hist, weights_hist, prob_hist


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

    # History of rewards, weights and probability distribution
    reward_hist, weights_hist, prob_hist = np.zeros(T), [], []

    L = np.zeros(K) # Estimated cumulative loss
    for t in range(T):

        # Set probabilities of drawing each arm
        p = np.exp(-eta * (L - np.min(L)))
        p = p / np.sum(p)

        # Draw arm index
        drawn_index = np.random.choice(a=K, p=p)

        # Draw corresponding reward
        drawn_reward = MAB[drawn_index].sample(t)
        
        drawn_loss = 1 - drawn_reward #as in the article we use losses
        
        # Compute estimated reward with implicit exploration
        estimated_loss = drawn_loss / (p[drawn_index] + gamma)
        
        L[drawn_index] += estimated_loss #update cumulative loss
        
        # Save obtained reward and weights
        reward_hist[t] = drawn_reward
        weights_hist.append(np.exp(-eta * L))
        prob_hist.append(p)

    return reward_hist, weights_hist, prob_hist