import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

class AirDefenseEnv(gym.Env):
    
    NActions = 2
    NControls = 1
    
    DEG2RAD = math.PI / 180.0
    
    H0 = 1.0
    VMissle = H0/20.0
    VAntiMissle = H0/30.0
    D = 1.0
    AMax = 20 * DEG2RAD
    X0 = 0.2
    R = 0.01            # kill radius
    
    def __init__(self):
        self.seed()
        self.viewer = None
        self.state = None

        self.MissleX = None
        self.MissleY = None
        self.MissleA = None
        self.MissleVX = None
        self.MissleVY = None

        self.AMissleX = None
        self.AMissleY = None
        self.AMissleVX = None
        self.AMissleVY = None
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def state(self):
        fired = self.self.AMissleX is not None
        return np.array([1.0 if fired else 0.0, self.MissleX, self.MissleY, self.MissleA])
    
    def reset(self):
        x, a = self.np_random.uniform(low=-1.0, high=1.0, size=(2,))
        self.MissleX = x * self.X0 
        self.MissleY = self.H0
        self.MissleA = a * self.AMax
        self.MissleVX = self.VMissle * math.sin(self.MissleA)
        self.MissleVY = -self.VMissle * math.cos(self.MissleA)
        
        self.AMissleX = self.AMissleY = None
        
        return self.state()
        
    def step(self, action):
        fire, angle = action
        angle = angle[0]
        
        self.MissleX += self.MissleVX
        self.MissleY += self.MissleVY
        
        hit = False
        
        if fire and self.AMissleX is None:
            self.AMissleX = self.AMissleY = 0.0
            self.AMissleA = angle
            self.AMissleVX = self.VAntiMissle * math.sin(self.AMissleA)
            self.AMissleVY = self.VAntiMissle * math.cos(self.AMissleA)
        
        if self.AMissleX is not None:
            self.AMissleX += self.AMissleVX
            self.AMissleY += self.AMissleVY
            dx = self.AMissleX - self.MissleX
            dy = self.AMissleY - self.MissleY
            hit = math.sqrt(dx*dx + dy*dy) <= self.R
        
        ground = self.MissleY <= 0.0
        
        reward = 1.0 if hit else (-1.0 if ground else 0.0)
        done = ground or hit
        return self.state(), reward, done, {}
            
                
        
