import math, time
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import random
from draw2d import Viewer, Frame, Line, Polygon, Circle, Text


class ClusterKEnv(gym.Env):
    
    XMin = -1.0
    XMax = 1.0
    Alpha = 0.01
    
    def __init__(self, nclusters=5, ndimensions=2, cluster_size=0.1, points=200):
        self.NClusters = nclusters
        self.NDimensions = ndimensions
        self.ClusterSize = 0.1
        self.NPoints = points
        
        low = np.ones((self.NDimensions,))*self.XMin
        high = np.ones((self.NDimensions,))*self.XMax
        self.observation_space = spaces.Box(low, high)
        self.action_space = spaces.Discrete(self.NClusters)
        self.TrueCenters = self.generate_clusters()
        self.Centers = np.random.random((self.NClusters, self.NDimensions))*(self.XMax-self.XMin) + self.XMin
        self.Centers2 = self.Centers**2
        self.Sigmas2 = np.ones((self.NClusters,))
                
        # for rendering
        self.Points = []
        self.Viewer = None
        
    def generate_clusters(self, nattempts=10):
        best_config = None
        max_dist = 0.0
        for _ in range(nattempts):
            clusters = self.XMin + self.ClusterSize + np.random.random((self.NClusters, self.NDimensions)) * (self.XMax-self.XMin-2*self.ClusterSize)
            d = 0
            for i, c1 in enumerate(clusters[:-1]):
                for c2 in clusters[i+1:]:
                    d += math.sqrt(np.sum((c1-c2)**2))
            if d > max_dist:
                best_config = clusters
                max_dist = d
        return best_config


    def kseparation(self, k=10):
        # calculate distances between points
        n = len(self.Points)
        dist = np.empty((n,n))
        points = np.array([p for _, p in self.Points])
        clusters = np.array([c for c, _ in self.Points])
        dist = np.sqrt(np.sum((points[None,:]-points[:,None])**2, axis=-1))
        
        n_matches = 0.0
        n_total = 0.0
        
        for i, (ci, pi) in enumerate(self.Points):
            sorted_points = sorted(zip(dist[i], clusters))[1:]   # remove first element because it will be <i,i>
            k_nearest = sorted_points[:k]
            n_matches += sum(c == ci for _, c in k_nearest)
            n_total += len(k_nearest)
            
        return n_matches/n_total

    def next_point(self):
        ic = random.randint(0, self.NClusters-1)
        point = np.random.normal(self.TrueCenters[ic], self.ClusterSize, self.NDimensions)
        self.LastPoint = point
        return point
        
    def reward(self, point, ic):
        r = 0.0
        for i, c in enumerate(self.Centers):
            d = math.sqrt(np.sum((c-point)**2))
            d = 1-math.exp(-d/(0.01+self.Sigmas[i]))
            if i == ic:
                r -= d
            else:
                r += d
        return r/(self.NClusters-1)
    
    def update_center(self, point, ic):
        self.Centers[ic] += self.Alpha * (point - self.Centers[ic])
        self.Centers2[ic] += self.Alpha * (point**2 - self.Centers2[ic])
        self.Sigmas = np.sqrt(np.mean(self.Centers2 - self.Centers**2, axis=-1))
        
    def reset(self):
        self.Points = []
        return self.next_point()
        
    def step(self, action):
        self.add_point_history(action, self.LastPoint)
        self.update_center(self.LastPoint, action)
        reward = 0.0
        done = False
        if len(self.Points) >= self.NPoints:
            reward = self.kseparation()
            done = True
        return self.next_point(), reward, done, {}

    def add_point_history(self, ic, point):
        self.Points.append((ic, point))

    def render(self):
        if self.Viewer is None:
            self.Viewer = Viewer(600,600)
            self.Frame = self.Viewer.frame(self.XMin, self.XMax, self.XMin, self.XMax)
            self.ClusterColors = np.random.random((self.NClusters, 3))
            for c in self.ClusterColors:
                icmin = np.argmin(c)
                icmax = np.argmax(c)
                c[icmin] = 0.1
                c[icmax] = 0.9
            
        self.Frame.clear()

        for i, c in enumerate(self.TrueCenters):
            self.Frame.add(
                Circle(0.01, res=6).color(0.5,0.5,0.5),
                at = c[:2]
            )
        
        for i, c in enumerate(self.Centers):
            self.Frame.add(
                Circle(0.01, res=6).color(*self.ClusterColors[i]),
                at = c[:2]
            )
        
        for i, point in self.Points:
            self.Frame.add(
                Circle(0.005, res=4).color(*self.ClusterColors[i]),
                at = point[:2]
            )
        self.Viewer.render()
        time.sleep(0.001)

            
        
        
        
        
        
        