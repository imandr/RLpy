import random
import numpy as np
import math, time
from gym import spaces
from draw2d import Viewer, Frame, Line, Polygon, Circle, Text


class TankTargetEnv(object):
    
    FireRange = 0.1
    Speed = 0.02
    RotSpeed = math.pi*2/50
    Width = 0.01
    TimeHorizon = 100
    GasReward = 0.0
    IdleReward = 0.0
    MissReward = -0.02
    HitReward = 10.0
    
    X0 = 0.0
    X1 = 1.0
    Y0 = 0.0
    Y1 = 1.0
    Margin = 0.1
    
    FIRE = 0
    FWD = 1
    FFWD = 2
    LEFT = 3
    RIGHT = 4
    NActions = 5
    NState = 6
    
    

    def __init__(self):
        self.Viewer=None
        self.Hit = False
        self.Fire = False
        self.EpisodeReward = 0.0
        self.T = self.TimeHorizon
        
        high = np.array([1.0]*self.NState, dtype=np.float32)
        self.action_space = spaces.Discrete(self.NActions)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        
        
    def bind_angle(self, a):
        while a < -math.pi:
            a += math.pi*2
        while a >= math.pi:
            a -= math.pi*2
        return a
        
    def observation(self):
        obs = np.empty((self.NState,))
        obs[0] = self.X
        obs[1] = self.Y
        obs[2] = self.Angle
        dx = self.TargetX - self.X
        dy = self.TargetY - self.Y
        obs[3] = math.sqrt(dx*dx + dy*dy)
        c = math.atan2(dy, dx)
        obs[4] = self.bind_angle(c-self.Angle)
        obs[5] = self.T/self.TimeHorizon
        return obs
        
    def seed(self, x):
        pass
        
    def reset(self):
        self.TargetX = self.Margin + random.random()*(self.X1-self.X0-self.Margin*2)
        self.TargetY = self.Margin + random.random()*(self.Y1-self.Y0-self.Margin*2)
        self.X = self.Margin + random.random()*(self.X1-self.X0-self.Margin*2)
        self.Y = self.Margin + random.random()*(self.X1-self.X0-self.Margin*2)
        self.Angle = self.bind_angle(random.random()*2*math.pi - math.pi)
        self.EpisodeReward = 0.0
        self.T = self.TimeHorizon
        
        return self.observation()
        
    def step(self, action):
        self.Hit = self.Fire = False
        self.Reward = 0.0
        done = False
        reward = self.IdleReward
        if action in (self.FWD, self.FFWD):
            d = self.Speed/2 if action == self.FWD else self.Speed*2
            reward = self.GasReward/4 if action == self.FWD else self.GasReward*2
            x = self.X + math.cos(self.Angle)*d
            y = self.Y + math.sin(self.Angle)*d
            x1 = max(self.X0, min(self.X1, x))
            y1 = max(self.Y0, min(self.Y1, y))
            if x1 != x or y1 != y:  # bump ?
                reward = -1.0
                done = True
            self.X, self.Y = x1, y1
            #self.Reward += 0.001
        elif action == self.FIRE:
            self.Fire = True
            dx = self.TargetX - self.X
            dy = self.TargetY - self.Y
            a = math.atan2(dy, dx)
            distance = math.sqrt(dx*dx + dy*dy)
            delta = distance * math.sin(abs(a-self.Angle))
            self.Hit = abs(self.Angle - a) < math.pi/4 and delta < self.Width and distance < self.FireRange + self.Width
            if self.Hit:
                print("hit")
                done = True
                reward = self.HitReward
            else:
                reward = self.MissReward
        elif action == self.LEFT:
            self.Angle += self.RotSpeed
            self.Angle = self.bind_angle(self.Angle)
        elif action == self.RIGHT:
            self.Angle -= self.RotSpeed
            self.Angle = self.bind_angle(self.Angle)
            
        self.T -= 1
        if self.T <= 0:
            done = True
        self.Reward = reward
        self.EpisodeReward += self.Reward
        
        return self.observation(), reward, done, {}
        
    def render(self):
        if self.Viewer is None:
            self.Viewer = Viewer(600, 600)
            self.Frame = self.Viewer.frame(0.0, 1.0, 0.0, 1.0)
            
            self.Tank = Frame()
            self.Tank.add(
                Polygon([(-0.02, -0.01), (0.02, 0.0), (-0.02, 0.01)]).color(0.0, 0.5, 0.1)
            )
            self.Beam = Line(end=(self.FireRange, 0)).color(1.0, 0.5, 0.0)
            self.Tank.add(self.Beam)
            self.Frame.add(self.Tank)

            self.Target = Circle(self.Width, filled=False)
            self.Frame.add(self.Target)
            
            self.ScoreText = Text("", anchor_x="left", size=8).color(0.5, 0.5, 0.5)
            self.Frame.add(self.ScoreText, at=(0.01, 0.01))
            
        self.Tank.move_to(self.X, self.Y)
        self.Tank.rotate_to(self.Angle)
        self.Beam.hidden = not self.Fire
        self.Target.move_to(self.TargetX, self.TargetY)
        if self.Hit:
            self.Target.color(1.0, 1.0, 0.5)
        else:
            self.Target.color(0.5, 0.5, 0.5)
            
        self.ScoreText.Text = "r:%.3f R:%.3f %s" % (self.Reward, self.EpisodeReward, self.observation())
            
        self.Viewer.render()
        
        if self.Hit:
            time.sleep(0.2)
        

            
        
        
        
        

