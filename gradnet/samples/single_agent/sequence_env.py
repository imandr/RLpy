import math, random
from gym import spaces
import numpy as np

class SequenceEnv(object):
    """
    Environment to test using RL to train RNN-based sequence generator
    """

    def __init__(self, nvalues, distance):
        self.NWords = nvalues
        self.Distance = distance
        self.Eye = np.eye(nvalues)
        self.Sequence = []
        self.Done = False
        
        self.action_space = spaces.Discrete(nvalues)
        high = np.ones((nvalues,))
        low = np.zeros((nvalues,))
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        
        
    def reset(self):
        w = random.randint(0, self.NWords-1)
        self.Sequence = [w]
        self.Done = False
        return self.Eye[w]
        
    def step(self, x):
        t = len(self.Sequence)
        reward = 0.0
        done = False
        
        d = max(x, self.Distance)
        
        if x in self.Sequence[-d:]:
            done = True
            reward = -1.0
        
        self.Sequence.append(x)
        self.Done = done
        return self.Eye[x], reward, done, {}

    def render(self, mode='human'):
        if self.Done or len(self.Sequence)>=50:
            print(self.Sequence, self.Done)
