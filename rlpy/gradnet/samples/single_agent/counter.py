import math, time
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import random, math
from draw2d import Viewer, Frame, Line, Polygon, Circle, Text


class CounterEnv(gym.Env):
    
    NActions = 5
    StateSize = NActions
    
    def __init__(self):
        self.action_space = spaces.Discrete(self.NActions)
        high = np.ones((self.StateSize,))
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.Viewer = None
        
    def seed(self, seed):
        random.seed(seed)
        
    def reset(self):
        self.Seen = np.zeros((self.NActions,))
        return self.observation(0)
        
    def observation(self, a):
        obs = np.zeros((self.StateSize))
        #obs[:self.NActions] = self.Seen
        obs[a] = 1.0
        return obs

    def step(self, action):
        self.Action = action
        done = False
        reward = 0.0
        if action == 0:
            self.Seen[0] = 1
            reward = 1.0 if all(self.Seen == 1) else -1.0
            done = True
        elif self.Seen[action]:
            reward = -1.0
            done = True
        else:
            self.Seen[action] = 1.0
        self.Reward = reward
        self.Done = done
        return self.observation(action), reward, done, {}
        
    def render(self):
        print(self.Action, self.Seen, self.Reward, self.Done)
        time.sleep(0.1)
        
        
        
    
    
