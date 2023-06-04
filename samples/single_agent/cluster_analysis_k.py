import math, time
from colorsys import hsv_to_rgb
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
        
class ClusterKEnv(gym.Env):
    
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
        self.Done = False
        
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
        
    def update_center(self, point, ic):
        self.Centers[ic] += self.Alpha * (point - self.Centers[ic])
        self.Centers2[ic] += self.Alpha * (point**2 - self.Centers2[ic])
        self.Sigmas2[ic] = np.mean(self.Centers2[ic] - self.Centers[ic]**2, axis=-1)
        #print("update_center: sigmas2[ic]->", self.Sigmas2[ic])
        self.Sigmas = np.sqrt(self.Sigmas2)
        
    def kseparation(self, points_clusters, k=10):
        # calculate distances between points
        n = len(self.Points)
        if k == 0 or n < k*2:
            return 0.0
        dist = np.empty((n,n))
        points = np.array([p for _, p in points_clusters])
        clusters = np.array([c for c, _ in points_clusters])
        dist = np.sqrt(np.sum((points[None,:]-points[:,None])**2, axis=-1))
        #print("dist:", dist[:5, :5])
        
        n_matches = n_anti_matches = n_total = 0.0
        
        for i, (ci, pi) in enumerate(points_clusters):
            sorted_points = sorted(list(zip(dist[i], clusters)))   # remove first element because it will be <i,i>
            #print("sorted_points:", sorted_points[:10], sorted_points[-10:])
            sorted_points = sorted_points[1:]   # remove first element because it will be <i,i>
            k_nearest = sorted_points[:k]
            k_distant = sorted_points[-k:]
            n_matches += sum(int(c == ci) for _, c in k_nearest)
            n_total += len(k_nearest)
            n_anti_matches += sum(int(c != ci) for _, c in k_distant)
            n_total += len(k_distant)
        
        #print("nmatches, nanti, ntotal:", n_matches, n_anti_matches, n_total)
        return (n_matches+n_anti_matches)/n_total
        
    def variety(self, points_clusters):
        hist, _ = np.histogram([c for c, _ in points_clusters], bins=self.NClusters, range=(0,self.NClusters-1))
        #print("hist:", hist)
        n1 = np.max(hist)
        n2 = np.min(hist)
        return n2/n1
        
    def reward(self, points):
        if len(points) < 2*self.NClusters:
            return 0.0
        
        last_c, last_p = points[-1]
        r = 0.0
        for c, p in points[:-1]:
            d = math.sqrt(float(np.sum((p-last_p)**2)))
            if c == last_c:
                r += math.exp(-d)
            else:
                r += 1-math.exp(-d)
        r /= len(points)-1
        
        nclusters = np.zeros((self.NClusters,))
        for c, _ in points:
            #print(c)
            nclusters[c] = 1
        xpresent = sum(nclusters)/self.NClusters
        return r * xpresent

    def reset(self):
        self.Points = []
        self.PrevCluster = None
        self.PrevPoint = None
        self.Done = False
        return self.next_point()
        
    def step(self, action):
        self.add_point_history(action, self.LastPoint)
        self.update_center(self.LastPoint, action)
        reward = 0.0
        done = False
        if len(self.Points) >= self.NPoints:
            reward = self.kseparation(self.Points) + self.variety(self.Points)
            done = True
        elif len(self.Points) >= 20:
            reward = self.reward(self.Points[-self.NClusters*2:])/100
        elif False and self.PrevPoint is not None:
            prev_cluster = self.PrevCluster
            prev_point = self.PrevPoint
            this_cluster = action
            this_point = self.LastPoint
            dist = math.sqrt(np.sum((this_point-prev_point)**2))
            if this_cluster == prev_cluster:
                f = -dist
            else:
                f = (1-math.exp(dist))/self.NClusters
            reward = f
            #print("cluster:", prev_cluster, cluster, " distance:", dist, "   reward:", reward)
            self.PrevCluster = action
            self.PrevPoint = self.LastPoint
        next_point = self.next_point()
        self.Done = done
        return next_point, reward, done, {}

    def add_point_history(self, ic, point):
        self.Points.append((ic, point))

    def render(self):
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

        for path, center in zip(self.CenterPaths, self.Centers):
            path.append(center.copy())

        for i, c in enumerate(self.Generator.Sources):
            self.Frame.add(
                Circle(0.02, res=6, filled=False).color(0.8,0.8,0.8),
                at = c[:2]
            )
        
        for i, c in enumerate(self.Centers):
            self.Frame.add(
                Circle(0.01, res=6).color(*self.ClusterColors[i]),
                at = c[:2]
            )
            #self.Frame.add(Text(f"{i}", anchor_x="left", anchor_y="bottom", color=tuple(self.ClusterColors[i]*0.8)), at = c[:2])
            
        
        for i, point in self.Points:
            self.Frame.add(
                Circle(0.005, res=4).color(*self.ClusterColors[i]*0.8),
                at = point[:2]
            )
            
        # center paths
        for i, cluster_path in enumerate(self.CenterPaths):
            #print(i, path[:10])
            if False:
                n = len(cluster_path)
                if n > 100:
                    # prune
                    d = int((n+99)/100)
                    pruned = [cluster_path[j] for j in range(0, n-1, d)]
                    need = 100-len(pruned)
                    if need > 0:
                        pruned += cluster_path[-need:]
                    path = pruned
            path = cluster_path[-10:]
            self.Frame.add(PolyLine(path, False).color(*self.ClusterColors[i]))
            
        self.Viewer.render()
        time.sleep(0.0001)

            
        
        
        
        
        
        