import math, time
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import random, math
from draw2d import Viewer, Frame, Line, Polygon, Circle, Text


class AscentionEnv(gym.Env):
    
    dirs = np.array([
        (1,0),
        (1,1),
        (0,1),
        (-1,1),
        (-1,0),
        (-1,-1),
        (0,-1),
        (1,-1)
    ]*2, dtype=np.float32)
    
    R = 0.1
    
    steps = dirs*R
    
    def __init__(self):
        self.NActions = len(self.steps)
        self.StateSize = 3+2
        self.action_space = spaces.Discrete(self.NActions)
        high = np.ones((self.StateSize,))
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.Viewer = None
        
    def seed(self, seed):
        random.seed(seed)
        
    def reset(self):
        
        self.xc = random.random()
        self.yc = random.random()
        self.a = 2+random.random()
        self.b = 2+random.random()
        
        self.x = random.random()
        self.y = random.random()

        return self.observation(0.0, 0.0)
        
    def f(self):
        return 1.0 - self.a*(self.x-self.xc)**2 - self.b*(self.y-self.yc)**2
        
    def reward_f(self):
        d2 = (self.x - self.xc)**2 + (self.y - self.yc)**2
        r = math.sqrt(d2)/self.R
        return max(-1.0, 1.0-r/2)
        
    def observation(self, dx, dy):
        obs = np.zeros((self.StateSize,))
        obs[:3] = np.array([self.x, self.y, self.f()])
        obs[3] = dx
        obs[4] = dy
        return obs
        
    def step(self, action):
        self.Action = action
        done = 0
        dx = dy = 0.0
        if action == 0:
            reward = self.reward_f()
            done = True
        else:
            dx, dy = self.steps[action-1]
            self.x += dx
            self.y += dy
            if self.x > 1.0 or self.x < 0.0 or self.y > 1.0 or self.y < 0.0:
                reward = -2.0
                done = True
            else:
                reward = -0.002
        return self.observation(dx, dy), reward, done, {}
        
    def render(self):
        if self.Viewer is None:
            self.Viewer = Viewer(500,500)
            self.Frame = self.Viewer.frame(0,1,0,1)
            self.Target = Circle(0.01, filled=False).color(1,0,0)
            self.Probe = Circle(0.01, filled=True).color(1,1,1)
            self.Frame.add(self.Target)
            self.Frame.add(self.Probe)
            self.Circle1 = Circle(self.R*2, filled=False).color(0.5, 0.5, 0.5)
            self.Circle2 = Circle(self.R, filled=False).color(0.5, 0.5, 0.5)
            self.Frame.add(self.Circle1)
            self.Frame.add(self.Circle2)
            

        self.Circle1.move_to(self.xc, self.yc)
        self.Circle2.move_to(self.xc, self.yc)
        self.Target.move_to(self.xc, self.yc)
        self.Probe.move_to(self.x, self.y)
        
        self.Viewer.render()
        time.sleep(0.1)
        
        
        
    
    
