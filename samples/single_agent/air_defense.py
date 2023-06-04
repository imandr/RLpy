import math, time
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

class AirDefenseEnv(gym.Env):
    
    NActions = 2
    NControls = 1
    
    DEG2RAD = math.pi / 180.0
    
    H0 = 1.0
    VMissle = H0/20.0
    VAntiMissle = H0/30.0
    D = 1.0
    AMax = 20 * DEG2RAD
    X0 = 0.2
    R = 0.1            # kill radius
    
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

        low = np.array([0.0, -self.D, 0.0, -math.pi/2])
        high = np.array([1.0, self.D, self.H0, math.pi/2])

        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def state_vector(self):
        fired = self.AMissleX is not None
        return np.array([1.0 if fired else 0.0, self.MissleX, self.MissleY, self.MissleA])
    
    def reset(self):
        x, a = self.np_random.uniform(low=-1.0, high=1.0, size=(2,))
        self.MissleX = x * self.X0 
        self.MissleY = self.H0
        self.MissleA = a * self.AMax
        self.MissleVX = self.VMissle * math.sin(self.MissleA)
        self.MissleVY = -self.VMissle * math.cos(self.MissleA)
        
        self.AMissleX = self.AMissleY = self.LastDist = None
        self.Hit = False
        return self.state_vector()
        
    def step(self, action):
        fire, angle = action
        angle = angle[0]
        
        self.MissleX += self.MissleVX
        self.MissleY += self.MissleVY
        
        hit = closer = ground = False
        
        if fire and self.AMissleX is None:
            print("fire, at angle:", angle)
            if angle < 0.0 or angle > math.pi:
                ground = True
            else:
                self.AMissleX = self.AMissleY = 0.0
                self.AMissleA = angle
                self.AMissleVX = -self.VAntiMissle * math.sin(self.AMissleA)
                self.AMissleVY = self.VAntiMissle * math.cos(self.AMissleA)
        
        if self.AMissleX is not None:
            self.AMissleX += self.AMissleVX
            self.AMissleY += self.AMissleVY
            dx = self.AMissleX - self.MissleX
            dy = self.AMissleY - self.MissleY
            dist = math.sqrt(dx*dx + dy*dy)
            hit = dist <= self.R
            closer = self.LastDist is not None and dist < self.LastDist
            self.LastDist = dist
        
        ground = ground or self.MissleY <= 0.0
        
        reward = 0.0
        done = False
        if ground:
            reward = -5.0
            done = True
        elif hit:
            reward = 5.0
            done = True
        else:
            reward = 0.1 if closer else 0.0
        self.Hit = self.Hit or hit
        return self.state_vector(), reward, done, {}
            
                
    def render(self):
        screen_width = 800
        screen_height = 400
        
        if self.viewer is None:
            from draw2d import Viewer, Frame, FilledPolygon, Circle
            self.viewer = Viewer(screen_width, screen_height)
            self.outer_frame = self.viewer.frame(-self.D, self.D, 0, self.H0)
            
            self.MissleFrame = Frame(scale = 0.1)
            missle = FilledPolygon([(-0.1, -0.1), (0.1, -0.1), (0.0, 0.2)]).color(0.9, 0.3, 0.2).rotate_by(math.pi)
            self.MissleFrame.add(missle)

            self.AMissleFrame = Frame(scale = 0.1)
            amissle = FilledPolygon([(-0.1, -0.1), (0.1, -0.1), (0.0, 0.2)]).color(0.3, 0.9, 0.2)
            self.AMissleFrame.add(amissle)
            self.AMissleFrame.hide()
            
            self.outer_frame.add(self.MissleFrame)
            self.outer_frame.add(self.AMissleFrame)
            self.Bang = Circle(0.15).color(1.0, 1.0, 0.6)
            
        self.MissleFrame.show()
        if self.AMissleX is not None:
            self.AMissleFrame.move_to(self.AMissleX, self.AMissleY)
            self.AMissleFrame.rotate_to(self.AMissleA)
            self.AMissleFrame.show()
        
        if self.Hit:
            self.AMissleFrame.hide()
            self.MissleFrame.hide()
            self.outer_frame.add(self.Bang, at=(self.AMissleX, self.AMissleY))
            self.Bang.show()

        self.MissleFrame.move_to(self.MissleX, self.MissleY)
        self.MissleFrame.rotate_to(self.MissleA)

        time.sleep(0.1)
        if self.Hit:
            time.sleep(0.2)

        self.viewer.render()
        self.Bang.hide()