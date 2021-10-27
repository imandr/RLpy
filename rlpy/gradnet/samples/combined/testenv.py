import numpy as np

class TestEnv(object):
    
    W = 3
    NA = 2
    NC = 2
    
    ObservationSpec = [(W,), (W,)]
    ActionSpec = ([2,2], 2)
    
    def __init__(self):
        self.State = None
        
    def next(self):
        self.State = [np.random.random((self.W,)), np.random.random((self.W,))]
                
    def observation(self):
        return self.State
        
    def reset(self):
        self.next()
        return self.observation()
        
    def step(self, action):
        actions, controls = action
        x, y = self.State
        x2 = np.sum(x**2)
        y2 = np.sum(y**2)
        sign = x2 > y2
        reward = 0.0
        
        a1, a2 = actions
        c1, c2 = controls
        
        reward += (1.0 if (sign and a1 == 1) else 0.0) + (1.0 if (not sign and a2 == 1) else 0.0)
        reward -= (c1 - np.mean(x))**2 + (c2 - np.mean(y))**2
        
        self.next()
        return self.observation(), reward, False, {}
        
        
        
        