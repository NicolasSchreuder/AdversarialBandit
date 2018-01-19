import numpy as np

def UCB1(MAB, T, rho=0.5):
    """
    UCB1 algorithm on a bandit game MAB of length T

    rho determines the exploration/exploitation policy :
    rho < 0.5 -> polynomial regret w.r.t. n
    rho > 0.5 -> logarithmic regret w.r.t. n
    """

    K = len(MAB) # number of arms
    N = np.zeros(K) # number of draws
    S = np.zeros(K) # sum of rewards
    rew = [] # sequence of the T rewards obtained
    draws = [] # sequence of the T arms drawn

    # Draw each arm once
    for k in range(K):
        N[k] += 1 # Updates number of draws for arm k
        reward = float(MAB[k].sample())
        S[k] += reward # Updates sum of rewards

        rew.append(reward) # Saves obtained reward
        draws.append(k) # Saves selected arm

    # Continues playing until time T
    for t in range(K, T):

        mu = S/N # Empirical mean of the rewards

        arm_estimated_values = [mu[k] + rho*np.sqrt(log(t)/(2*N[k])) for k in range(K)]

        selected_arm = np.argmax(arm_estimated_values) # Selects best arm

        N[selected_arm] += 1 # Updates number of times arm pulled
        draws.append(selected_arm) # Saves selected arm

        # Draws a sample from the selected arm and saves received reward
        instant_reward = float(MAB[selected_arm].sample())

        rew.append(instant_reward) # Saves obtained reward

        S[selected_arm] += instant_reward # Updates sum of rewards

    return rew

def Random(MAB, T):
    """
    Random algorithm for MAB
    """
    reward_hist = []
    for i in range(T):
        index = np.random.randint(len(MAB))
        reward = MAB[index].sample(i)
        reward_hist.append(reward)
    return reward_hist
