import math, time
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import random
from draw2d import Viewer, Frame, Line, Polygon, Circle, Text, PolyLine



class Generator(object):
    
    ClusterSize = 0.1
    
    def __init__(self, nsources, ndimensions, xmin, xmax):
        self.NSources = nsources
        self.NDimensions = ndimensions
        self.XMax = xmax
        self.XMin = xmin
        self.Sources = self.generate_sources(nsources)
        self.Sizes = (np.random.random((self.NSources,))+0.5) * self.ClusterSize

    def generate_sources(self, n, nattempts=2):
        best_config = None
        max_dist = 0.0
        for _ in range(nattempts):
            clusters = self.XMin + self.ClusterSize + np.random.random((n, self.NDimensions)) * (self.XMax-self.XMin-2*self.ClusterSize)
            d = 0
            for i, c1 in enumerate(clusters[:-1]):
                for c2 in clusters[i+1:]:
                    d += math.sqrt(np.sum((c1-c2)**2))
            if d > max_dist:
                best_config = clusters
                max_dist = d
        return best_config

    def generate(self):
        ic = random.randint(0, self.NSources-1)
        return np.random.normal(self.Sources[ic], self.Sizes[ic], self.NDimensions)
        
class ClusterEnv(gym.Env):
    
    Alpha = 0.01
    XMin = -1.0
    XMax = 1.0
    
    def __init__(self, nsources=6, nclusters=None, ndimensions=2, cluster_size=0.1, points=200):
        self.NSources = nsources
        self.NClusters = nclusters or nsources
        self.NDimensions = ndimensions
        self.ClusterSize = 0.1
        self.NPoints = points
        
        self.Generator = Generator(nsources, ndimensions, self.XMin, self.XMax)
        
        low = np.ones((self.NDimensions,))*self.XMin
        high = np.ones((self.NDimensions,))*self.XMax
        self.observation_space = spaces.Box(low, high)
        self.action_space = spaces.Discrete(self.NClusters)
        self.Centers = np.random.random((self.NClusters, self.NDimensions))*(self.XMax-self.XMin) + self.XMin
        self.Centers2 = self.Centers**2
        self.Sigmas2 = np.ones((self.NClusters,))
                
        # for rendering
        self.Points = []
        self.CenterPaths = [[] for _ in range(self.NClusters)]
        self.Viewer = None
        
    def next_point(self):
        point = self.Generator.generate()
        self.LastPoint = point
        return point
        
    def reward(self, point, ic):
        d2 = np.sum((point - self.Centers)**2, axis=-1)
        d = np.sqrt(d2/self.Sigmas2)
        f = 1-np.exp(-d)
        w = np.ones(self.NClusters)
        w[ic] = -1
        r = np.sum(f*w)
        return r/(self.NClusters+1)

    def reward(self, point, ic):
        d2 = np.sum((point - self.Centers)**2, axis=-1)
        d = np.sqrt(d2/self.Sigmas2)
        f = np.log(d+1)
        w = np.ones(self.NClusters)
        w[ic] = -1
        r = np.sum(f*w)
        return r/(self.NClusters+1)

    def reward(self, point, ic):
        d2 = np.sum((point - self.Centers)**2, axis=-1)
        f = (np.exp(-d2/self.Sigmas2) - 0.5)*2
        w = np.ones(self.NClusters)
        w[ic] = 0 # -1
        r = -np.sum(f*w)
        return r/(self.NClusters+1)
        
    def update_center(self, point, ic):
        self.Centers[ic] += self.Alpha * (point - self.Centers[ic])
        self.Centers2[ic] += self.Alpha * (point**2 - self.Centers2[ic])
        self.Sigmas2[ic] = np.mean(self.Centers2[ic] - self.Centers[ic]**2, axis=-1)
        #print("update_center: sigmas2[ic]->", self.Sigmas2[ic])
        self.Sigmas = np.sqrt(self.Sigmas2)
        
    def update_center_(self, point, ic):
        d2 = np.sum((point - self.Centers)**2, axis=-1)
        nearest = np.argmin(d2)
        self.Centers[nearest] += self.Alpha * (point - self.Centers[nearest])
        self.Centers2[nearest] += self.Alpha * (point**2 - self.Centers2[nearest])
        self.Sigmas = np.sqrt(np.mean(self.Centers2 - self.Centers**2, axis=-1))
        #self.Sigmas[nearest] = np.sqrt(np.mean(self.Centers2[nearest] - self.Centers[nearest]**2, axis=-1))
        
    def reset(self):
        self.Points = []
        for path, center in zip(self.CenterPaths, self.Centers):
            path.append(center.copy())
        return self.next_point()
        
    def step(self, action):
        self.add_point_history(action, self.LastPoint)
        self.update_center(self.LastPoint, action)
        r = self.reward(self.LastPoint, action)
        self.Done = done = len(self.Points) >= self.NPoints
        return self.next_point(), r, done, {}

    def add_point_history(self, ic, point):
        self.Points.append((ic, point))

    def render(self):
        
        from colorsys import hsv_to_rgb
        
        if not self.Done:
            return
        
        if self.Viewer is None:
            self.Viewer = Viewer(600,600)
            self.Frame = self.Viewer.frame(self.XMin, self.XMax, self.XMin, self.XMax)
            hues = np.arange(self.NClusters+2)[1:-1]/self.NClusters
            s = 0.7
            v = 0.8
            colors = [np.array(hsv_to_rgb(h, s, v)) for h in hues]
            self.ClusterColors = colors
            
        self.Frame.clear()

        for i, c in enumerate(self.Generator.Sources):
            self.Frame.add(
                Circle(0.02, res=6, filled=False).color(0.8,0.8,0.8),
                at = c[:2]
            )
            self.Frame.add(Circle(self.Generator.Sizes[i], filled=False).color(0.6,0.6,0.6), at=c[:2])
        
        for i, c in enumerate(self.Centers):
            self.Frame.add(
                Circle(0.01, res=6).color(*self.ClusterColors[i]),
                at = c[:2]
            )
            #self.Frame.add(Text(f"{i}", anchor_x="left", anchor_y="bottom", color=tuple(self.ClusterColors[i]*0.8)), at = c[:2])
            self.Frame.add(Circle(self.Sigmas[i], filled=False).color(*self.ClusterColors[i]*0.5), at=c[:2])
            
        
        for i, point in self.Points:
            self.Frame.add(
                Circle(0.005, res=4).color(*self.ClusterColors[i]*0.8),
                at = point[:2]
            )
            
        # center paths
        for i, path in enumerate(self.CenterPaths):
            #print(i, path[:10])
            n = len(path)
            if n > 100:
                # prune
                d = int((n+99)/100)
                pruned = [path[j] for j in range(0, n-1, d)]
                need = 100-len(pruned)
                if need > 0:
                    pruned += path[-need:]
                path = pruned
            self.Frame.add(PolyLine(path, False).color(*self.ClusterColors[i]))
            
        self.Viewer.render()
        #time.sleep(0.0001)

            
        
        
        
        
        
        