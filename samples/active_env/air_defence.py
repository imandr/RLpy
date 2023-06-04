import random
import numpy as np
import math, time
from gym import spaces
from draw2d import Viewer, Frame, Line, Polygon, Circle, Text
from rlpy import ActiveEnvironment

X0 = -1.0
X1 = 1.0
Y0 = 0.0
Y1 = 2.0

LaunchX = 0.0
LaunchY = 0.0

R = 0.1
V0 = 0.02
V1 = 0.05

DT = 1.0

class Missile(object):
    def __init__(self, x, y, v, direction):
        self.X = x
        self.Y = y
        self.V = v
        self.F = direction
        self.CosF = math.cos(self.F)
        self.SinF = math.sin(self.F)
        self.Hit = False
        
    def move(self, dt):
        self.X += self.V * self.CosF * dt
        self.Y += self.V * self.SinF * dt
        return self
        
    def distance(self, other):
        return math.sqrt((self.X-other.X)**2 + (self.Y-other.Y)**2)
        
    def hit(self, other):
        return self.distance(other) < R

class AirDefenceEnv(ActiveEnvironment):
    
    def __init__(self):
        obs_high = np.array([X1, Y1])
        obs_low = np.array([X0, Y0])
        ActiveEnvironment.__init__(self, 
                name="AirDefenceEnv", 
                action_space=spaces.Tuple(
                    [ 
                        spaces.Discreet(2),
                        spaces.Box(low=math.pi/4, high=math.pi*3/4)  
                    ]
                ),
                observation_space=spaces.Box(low, high, dtype=np.float32)
        )
        self.Viewer = None
        
    def incomingMissile():
        v = V0 + math.random()*(V1-V0)
        x0 = X0 + math.random()*(X1-X0)
        x1 = X0 + math.random()*(X1-X0)
        direction = math.atan2(Y0-Y1, x1-x0)
        return Missile(x0, Y1, v, direction)
    
    def seed(self, x):
        pass
        
    def reset(self, agents, training=True):
        assert len(agents) == 1, "Currently, only one agent is allowed"
        self.Target = self.incomingMissile()
        self.AntiMissile = None
        self.Agent = agents[0]
        
    def observation(self):
        return np.array([self.Target.X, self.Target.Y, float(self.AntiMissile is not None)])
        
    def turn(self):
        done = False
        reward = 0.0
        self.Target.move(DT)
        if self.AntiMissile is not None:
            self.AntiMissile.move(DT)
            if self.AntiMissile.hit(self.Target):
                am.Hit = self.Target.Hit = True
                done = True
                reward = 1.0

        if not done:
            if self.Target.Y < Y0:
                done = True
                reward = -1.0
        
        do_update = True
        if not done:
            if self.AntiMissile is None:
                fire, direction = self.Agent.action(self.observation())
                do_update = False
                if fire:
                    self.AntiMissile = Missile(LaunchX, LaunchY, V0, direction)

        if done:
            self.Agent.done(reward, self.observation())
        elif do_update:
            self.Agent.update(self.observation())
            
        return done
            
    def missile_sprite(self, *color):
        sprite = Frame()
        body = Polygon([(-0.02, -0.01), (0.02, 0.0), (-0.02, 0.01)]).color()
        return sprite.add(body)
            
    def render(self):
        if self.Viewer is None:
            W = 600
            H = W * (Y1-Y0)/(X1-X0)
            self.Viewer = Viewer(W, H)
            self.Frame = self.Viewer.frame(X0, X1, Y0, Y1)
        
            self.TargetSprite = self.missile_sprite(1.0, 0.5, 0.1)
            self.AntiSprite = None
            
            self.Frame.add(self.MissileSprite)
            
        self.TargetSprite.move_to(self.Target.X, self.Target.Y).turn_to(self.Target.F)
        if self.AntiMissile:
            if self.AntiSprite is None:
                self.AntiSprite = self.missile_sprite(0.1, 1.0, 0.1)
                self.Frame.add(self.AntiSprite)
            self.AntiStprite.move_to(self.AntiMissile.X, self.AntiMissile.Y).turn_to(self.AntiMissile.F)
        self.Viewer.render()
        time.sleep(0.1)
        
    def create_model(self, hidden):
        from gradnet import Input, Model
        from gradnet.layers import LSTM, Flatten, Concatenate, Dense

        inp = Input(None, 3)        # target x, y, antimissile fired
        lstm = LSTM(hidden)(inp)
        
        fire = Dense(2, activation="softmax")(lstm)
        direction_mean = Dense(1, activation="linear")(lstm)
        direction_sigma = Dense(1, activation="linear")(lstm)
        value = Dense(1, activation="linear")(lstm)

        model = Model([inp], [fire, direction])
    
        model["value"] = value
        model["probs"] = fire
        model["means"] = direction_mean
        model["sigmas"] = direction_sigma
        
        return model
    
    
            
            
            
            
            
            
            
        
        
        
                
        
    