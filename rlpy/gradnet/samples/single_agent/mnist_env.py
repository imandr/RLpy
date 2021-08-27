from tensorflow.keras.datasets import mnist
import numpy as np, random, time
from gym import spaces
from draw2d import Viewer, Rectangle, Text

(x_train, y_train), (x_test, y_test) = mnist.load_data()

def one_hot(labels, n):
    eye = np.eye(n)
    return eye[labels]
    
x_train = (x_train/256.0).reshape((-1, 28,28,1))
x_test = (x_test/256.0).reshape((-1, 28,28,1))
n_train = len(x_train)

class MNISTEnv(object):

    def __init__(self):
        self.Viewer = None
        self.action_space = spaces.Discrete(10)
        high = np.array([1.0]*(28*28))
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        
    def create_model(self, hidden):
        from gradnet import Input, Model
        from gradnet.layers import Conv2D, Flatten, Concatenate, Dense, Pool
        num_actions = self.action_space.n
        inp = Input((28,28,1), name="sensor")

        conv1 = Conv2D(3,3,32, activation="relu")(inp)
        pool1 = Pool(2,2, "max")(conv1)
        conv2 = Conv2D(3,3,64, activation="relu")(pool1)
        pool2 = Pool(2,2, "max")(conv2)
        common = Flatten()(pool2)

        #probs1 = Dense(hidden, activation="relu", name="probs1")(common)
        probs = Dense(num_actions, activation="softmax", name="probs")(common)
    
        #value1 = Dense(hidden, activation="relu", name="value1")(common)
        value = Dense(1, name="value")(common)

        model = Model(inp, [probs, value])
    
        model["value"] = value
        model["probs"] = probs
        
        return model
    
    def observation(self):
        self.I = random.randint(0, n_train-1)
        return x_train[self.I]
        
    def reset(self):
        return self.observation()

    def step(self, action):
        self.Image = x_train[self.I].reshape((28,28))
        self.Label = action
        reward = 0.0 if action != int(y_train[self.I]) else 1.0
        self.Reward = reward
        return self.observation(), reward, False, {}
        
    def render(self):
        if self.Viewer is None:
            self.Viewer = Viewer(100,100)
            self.Frame = self.Viewer.frame(0,28, 0,28)
            self.Grid = []
            for irow in range(28):
                row = []
                for icol in range(28):
                    r = Rectangle(icol, icol+1, 27-irow-1, 27-irow+1).color(0,0,0)
                    row.append(r)
                    self.Frame.add(r)
                self.Grid.append(row)
            self.Display = Text(anchor_x="left", color=(0,0,0))
            self.Frame.add(self.Display, at=(0, 0))

        col = (0.5,0.1,0.1) if self.Reward == 0 else (0.1,0.5,0.1)
        col = np.array(col)
        white = np.ones((3,))
        for irow, row in enumerate(self.Image):
            for icol, pixel in enumerate(row):
                
                pcolor = pixel*col + (1-pixel)*white
                
                self.Grid[irow][icol].color(*pcolor)
                
        self.Display.Text = f"{self.Label}"
        self.Viewer.render()
        time.sleep(0.2)
        
        
        
        