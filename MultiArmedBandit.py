impor numpy as np

class Arm():
    """
    Defines an arm with arbitrary finite sequence of reward
    """
    def __init__(self, rewards_sequence):
        self.rewards = rewards_sequence

    def sample(self, t):
        return self.rewards[t]

class bernoulliArm():
    """
    Defines a Bernoulli arm
    """
    def __init__(self, mean):
        self.mean = mean

    def sample(self, t):
        return np.random.binomial(p=self.mean, n=1)

class evolvingBernoulliArm():
    def __init__(self, means, switching_time):
        self.means = means
        self.switching_time = switching_time

    def sample(self, t):
        if t <= self.switching_time:
            return np.random.binomial(p=self.means[0], n=1)
        else:
            return np.random.binomial(p=self.means[1], n=1)
