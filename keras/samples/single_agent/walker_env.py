import math, random, time
from gym import spaces
import numpy as np
from draw2d import Viewer, Circle, PolyLine, Line, Frame, Polygon, Text

import tensorflow.keras.layers as layers
from tensorflow import keras

class WalkerEnv(object):

    Size = 50
    Window = 5
    VMax = Window+2
    Offset = Window * 4
    HoleDensity = 0.3
    ObservationSize = Window*Window + 6
    
    def __init__(self):
        
        self.Space = np.zeros((self.Size+2*self.Offset, self.Size+2*self.Offset))
        self.Space[...] = -1.0
        self.Field = self.Space[self.Offset:self.Offset+self.Size, self.Offset:self.Offset+self.Size]
        self.Field[...] = 0.0
        self.Viewer = None
        
        self.action_space = spaces.Discrete(9)
        high = np.array([1.0]*(self.Window*self.Window+6))
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.generate_field()
        self.render(field_only=True)
    
    SmoothMask = np.array([
        [0.0,0.2,0.3,0.4,0.2],
        [0.1,0.4,0.5,0.7,0.4],
        [0.3,0.5,1.0,1.0,0.6],
        [0.4,0.8,1.0,0.9,0.5],
        [0.2,0.4,0.6,0.5,0.3]
    ]) 

    SmoothMask = np.array([
        [0.3,0.8,0.3],
        [0.8,1.0,0.8],
        [0.3,0.8,0.3]
    ]) 
    
    MaskAverage = np.mean(SmoothMask)
    SmoothMask/=MaskAverage
    
    def create_model(self, hidden):
        num_actions = self.action_space.n
        sensor_shape = (self.Window, self.Window, 1)
        sensor_input = layers.Input(shape=sensor_shape, name="sensor")
        position_input = layers.Input(shape=(6,), name="position")
    
        conv1 = layers.Conv2D(4, 3, activation="relu")(sensor_input)
        conv2 = layers.Conv2D(self.Window*self.Window, 3, activation="relu")(conv1)
        flat = layers.Flatten()(conv2)
        concat = layers.Concatenate()([flat, position_input])
    
        common = layers.Dense(hidden, activation="relu", name="common")(concat)

        action1 = layers.Dense(max(hidden/5, num_actions)/2, activation="relu", name="action1")(common)
        action = layers.Dense(num_actions, activation="softmax", name="action")(action1)
    
        critic1 = layers.Dense(hidden/5, name="critic1", activation="softplus")(common)
        critic = layers.Dense(1, name="critic")(critic1)

        return keras.Model(inputs=[sensor_input, position_input], outputs=[action, critic])
    
    def smooth_field(self, field, alpha = 0.5):
        field_1 = field.copy()
        for x in range(1, self.Size-1):
            for y in range(1, self.Size-1):
                f1 = np.mean(field[x-1:x+2,y-1:y+2] * self.SmoothMask)
                field_1[x,y] = -1 if f1 <= -alpha else 0
        field[...] = field_1
        
        
    def generate_field(self):
        self.Field[...] = 0.0
        self.Field[0,:] = -1.0
        self.Field[-1,:] = -1.0
        self.Field[:,0] = -1.0
        self.Field[:,-1] = -1.0
        
        for x in range(self.Size):
            for y in range(self.Size):
                if random.random() < self.HoleDensity:
                    self.Field[x,y] = -1.0
        self.smooth_field(self.Field)
        self.smooth_field(self.Field)
        self.smooth_field(self.Field)
        
    def reset(self):
        self.generate_field()
        
        x = random.randint(0, self.Size-1)
        y = random.randint(0, self.Size-1)
        while self.Field[x,y]:
            x = random.randint(0, self.Size-1)
            y = random.randint(0, self.Size-1)
        self.X, self.Y = x, y
        
        x = random.randint(0, self.Size-1)
        y = random.randint(0, self.Size-1)
        while self.Field[x,y]:
            x = random.randint(0, self.Size-1)
            y = random.randint(0, self.Size-1)
        self.TargetX, self.TargetY = x, y
        self.Field[x,y] = 1.0
        
        self.VX = self.VY = 0
        
        self.Trace = [(self.X, self.Y)]
        self.EpisodeReward = self.Reward = 0.0
        
        self.Done = False
        
        return self.observation()
        
    def observation(self):
        obs = np.empty((self.ObservationSize,))
        obs[0] = self.X/self.Size
        obs[1] = self.Y/self.Size
        obs[2] = self.VX
        obs[3] = self.VY
        obs[4] = (self.TargetX-self.X)/self.Size
        obs[5] = (self.TargetY-self.Y)/self.Size
        wx = self.Offset + self.X + self.VX - self.Window//2
        wy = self.Offset + self.Y + self.VY - self.Window//2
        w = self.Space[wx:wx+self.Window,wy:wy+self.Window]
        #print(self.X, self.Y, wx, wy, w.shape)
        #obs[6:6+self.Window*self.Window] = w.reshape((-1,))[:]
        #print("observation:", obs)
        return [w[..., None], obs[:6]]
            
    def step(self, action):
        #print("setep: action:", action)
        done = False
        reward = 0.0
        
        dx = [-1, 0, 1,-1, 0, 1,-1,0,1]
        dy = [-1,-1,-1, 0, 0, 0,1,1,1]
        
        vx = max(-self.VMax, min(self.VMax, self.VX + dx[action]))
        vy = max(-self.VMax, min(self.VMax, self.VY + dy[action]))
        
        self.LastX = self.X
        self.LastY = self.Y
        self.LastVX = self.VX
        self.LastVY = self.VY
        self.Action = action
        
        x = self.X + vx
        y = self.Y + vy
        if self.Space[x+self.Offset, y+self.Offset] < 0.0:
            done = True
            #print("--- crash")
            reward = -2.0
        elif x == self.TargetX and y == self.TargetY:
            done = True
            reward = 10.0
            #print("--- hit")
        elif vx == 0 and vy == 0 and self.VX == 0 and self.VY == 0:
            reward = -0.05
        else:
            reward = 0.0
        self.VX, self.VY = vx, vy
        self.X, self.Y = x, y
        self.Trace.append((self.X, self.Y))
        self.EpisodeReward += reward
        self.Reward = reward
        self.Done = done
        return self.observation(), reward, done, {}

    def render(self, field_only=False):
        if self.Viewer is None:
            self.Viewer = Viewer(600, 600, clear_color=(1,1,1, 0.9))
            self.Frame = self.Viewer.frame(0.0, self.Size, 0.0, self.Size)
            
            self.Target = Circle(0.6, filled=True).color(0.9,0.7,0.1)
            #self.Walker = Circle(0.4, filled=True).color(0.1,0.5,0.9)
            self.Walker = Polygon([(-0.6, -0.4), (-0.6, 0.4), (0.6, 0.0)], filled=True).color(0.1,0.5,0.9)
            
            self.Text = Text("", anchor_x="left", size=8, color=(0.0, 0.0, 0.5))

        #print("old:", self.LastX, self.LastY, "  action:", self.Action, "  now:", self.X, self.Y, "  next:", next_x, next_y)
        
        self.Frame.remove_all()

        for x in range(self.Size):
            for y in range(self.Size):
                if self.Field[x,y] < 0:
                    self.Frame.add(Polygon([(0,0),(1,0),(1,1),(0,1)], filled=True).color(0.5,0.5,0.5), at=(x, y))

        if not field_only:
            self.Frame.add(self.Target, at=(self.TargetX+0.5, self.TargetY+0.5))
            self.Walker.rotate_to(math.atan2(self.VY, self.VX))
            self.Frame.add(self.Walker, at=(self.X+0.5, self.Y+0.5))
        
            self.Frame.add(PolyLine([(x+0.5, y+0.5) for x, y in self.Trace], False).color(0,0.1,0.5))
            for x, y in self.Trace:
                self.Frame.add(Circle(0.1, filled=True).color(0.1,0.5,0.9), at=(x+0.5, y+0.5))

            wx = self.X - self.Window//2
            wy = self.Y - self.Window//2
            if not self.Done:
                wx += self.VX
                wy += self.VY
            self.Frame.add(Polygon([
                (wx, wy), (wx+self.Window, wy), (wx+self.Window, wy+self.Window), (wx, wy+self.Window)
            ], filled=False).color(0.1, 0.1, 0.1).line_width(0.5))
            

            self.Text.Text = f"action:{self.Action}  reward:{self.Reward:.2f}({self.EpisodeReward:.2f})"
            self.Frame.add(self.Text, at=(1, 1))
                            
        self.Viewer.render()
        time.sleep(0.05)

