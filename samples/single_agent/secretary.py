import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import random

# Actions

class SecretaryHiringEnv(gym.Env):
    
    NObservation = 3

    PASS=0
    HIRE=1

    NActions = 2

    action_space = spaces.Discrete(NActions)
    low = np.zeros((NObservation,))
    high = np.ones((NObservation,))*100
    observation_space = spaces.Box(low, high, dtype=np.float32)
    
    def __init__(self, ncandidates=20):
        self.NCandidates = ncandidates
        
    def reset(self):
        self.VMin = self.VMax = self.Value = None
        self.Candidates = [random.random() for _ in range(self.NCandidates)]
        self.CRange = (min(self.Candidates), max(self.Candidates))
        self.ToRender = None
        self.next_candidate()
        return self.observation()
        
    def next_candidate(self):
        self.Value = value = self.Candidates.pop()
        if self.VMin is None:
            self.VMin = self.VMax = value
        elif value > self.VMax:
            self.VMax = value
        elif value < self.VMin:
            self.VMin = value
        return value

    def observation(self):
        value = self.Value
        rel_value = 0.0 if value == self.VMin else (value - self.VMin)/(self.VMax - self.VMin)
        return np.array([1.0 if len(self.Candidates) <= 1 else 0.0, len(self.Candidates)/self.NCandidates, rel_value])

    def step(self, action):
        # for rendering

        reward = 0.0
        done = False
        if action == self.HIRE:
            #reward = (self.Value - self.CRange[0])/(self.CRange[1] - self.CRange[0])
            reward = 27.18 if self.Value == self.CRange[1] else -10.0
            done = True
        elif not self.Candidates:
            reward = -10.0
            done = True

        self.ToRender = [self.VMin, self.VMax, self.Value, action, reward, done]

        if not done:
            self.next_candidate()

        return self.observation(), reward, done, {}
        
    def render(self):
        if self.ToRender is None:
            # right after the reset()
            print("---- begin episode: crange:", self.CRange)
        else:
            vmin, vmax, value, action, reward, done = self.ToRender
            rel_value = 0.0 if vmax == vmin else (value - vmin)/(vmax - vmin)
            print("step: vmin/vmax:", vmin, vmax,
                "   value/rel:", value, rel_value, "   action:", action
            )
            if done:
                print("------ end episode: reward:", reward)
