import random
import numpy as np
import math, time
from gym import spaces
from draw2d import Viewer, Frame, Line, Polygon, Circle, Text, Rectangle
from rlpy import ActiveEnvironment

FireRange = 0.2
TargetSize = 0.01

X0 = 0.0
X1 = 1.0
Y0 = 0.0
Y1 = 1.0
Margin = 0.1

FIRE = 0
FWD = 1
LEFT = 2
RIGHT = 3
FFWD = 4
BCK = 5


class Projectile(object):
    
    Velocity = 0.1

    def __init__(self, x, y, a):
        self.X = x
        self.Y = y
        self.Angle = a

    def move(self, my_tank, tanks):
        dx = self.Velocity * math.cos(self.Angle)
        dy = self.Velocity * math.sin(self.Angle)
        hit_t = None
        hit_tank = None
        hit_point = None
        for tank in tanks:
            if tank is not my_tank:
                dt_x, dt_y = tank.X - self.X, tank.Y - self.Y
                t = (dt_x*dx + dt_y*dy)/(dx*dx + dy*dy)
                t = min(max(t, 0.0), 1.0)
                hit_x, hit_y = self.X + dx*t, self.Y + dy*t
                distance = math.sqrt((hit_x - tank.X)**2 + (hit_y - tank.Y)**2)
                if distance < TargetSize:
                    if hit_t is None or hit_t > t:
                        hit_t = t
                        hit_tank = tank
                        hit_point = (t*dx, t*dy) 
        if hit_tank:
            self.X, self.Y = hit_point
        else:
            self.X += dx
            self.Y += dy
        return hit_tank
    
    def out_of_bounds(self):
        return self.X > X1 or self.X < X0 or self.Y > Y1 or self.Y < Y0

class Tank(object):

    ActionCapacity = 2
    ProjectileVelocity = 1.0
    Speed = 0.01
    RotSpeed = 5/180.0*math.pi
    
    def __init__(self):
        self.X = self.Y = None
        self.Angle = None
        self.Reward = 0.0       # accumulated since last action
        self.Hit = False
        self.Projectile = None
        self.FellOff = False

    def random_init(self, x0, x1, y0, y1, margin):
        self.X = margin + random.random()*(x1-x0-margin*2)
        self.Y = margin + random.random()*(y1-y0-margin*2)
        self.Angle = random.random()*2*math.pi - math.pi
        self.Reward = 0.0
        self.Fire = self.Hit = False
        
    def fire(self):
        if self.Projectile is None:
            self.Projectile = Projectile(self.X, self.Y, self.Angle)
            self.Fired = True

    def move(self, action):
        fell = False
        if action in (FWD, FFWD, BCK):
            d = self.Speed if action == FWD else (
                self.Speed*2 if action == FFWD else
                -self.Speed/2.1415
            )
            x = self.X + math.cos(self.Angle)*d
            y = self.Y + math.sin(self.Angle)*d
            x1 = max(X0, min(X1, x))
            y1 = max(Y0, min(Y1, y))
            fell = x1 != x or y1 != y
            self.X, self.Y = x1, y1
            #self.Reward += 0.001
        elif action == LEFT:
            self.Angle += self.RotSpeed
            self.Angle = self.bind_angle(self.Angle)
        elif action == RIGHT:
            self.Angle -= self.RotSpeed
            self.Angle = self.bind_angle(self.Angle)
        self.FellOff = fell
        return fell

    def bind_angle(self, a):
        while a < -math.pi:
            a += math.pi*2
        while a >= math.pi:
            a -= math.pi*2
        return a
        

class TankDuelProjectileEnv(ActiveEnvironment):
    
    TimeHorizon = 200
    BaseReward = 0.0
    FallReward = -20.0
    MissReward = -0.1
    WinReward = 20.0
    LooseReward = -WinReward
    
    NActions = 4
    NState = 10
    ObservationShape = (NState,)

    AvailableMask = np.ones((NActions,))

    def __init__(self, duel=True, target=True, compete=True):

        self.Compete = compete

        high = np.array([1.0]*self.NState, dtype=np.float32)
        ActiveEnvironment.__init__(self, name="MultiTankEnv", 
                action_space=spaces.Discrete(self.NActions), observation_space=spaces.Box(-high, high, dtype=np.float32))
        self.Duel = duel
        self.HitTarget = target
        self.Viewer=None
        self.Hit = False
        self.Fire = False
        self.EpisodeReward = 0.0
        self.T = self.TimeHorizon
        
        self.Tanks = [Tank() for _ in (0,1)]
        self.Target = Tank()
        
    def observation(self, i):
        tank = self.Tanks[i]
        other = self.Tanks[1-i]
        obs = np.empty((self.NState,))
        obs[0] = tank.X
        obs[1] = tank.Y
        obs[2] = tank.Angle
        dx = other.X - tank.X
        dy = other.Y - tank.Y
        bearing = math.atan2(dy, dx)
        obs[3] = math.sqrt(dx*dx + dy*dy)
        obs[4] = bearing
        obs[5] = other.Angle - tank.Angle
        
        dx = self.Target.X - tank.X
        dy = self.Target.Y - tank.Y
        obs[6] = math.sqrt(dx*dx + dy*dy)
        obs[7] = math.atan2(dy, dx)
        
        obs[8] = self.T/self.TimeHorizon
        obs[9] = (tank.Projectile is not None and 1.0) or 0.0
        
        return obs
        
    def seed(self, x):
        pass
        
    def reset(self, agents, training=True):
        self.Agents = agents
        [t.random_init(X0, X1, Y0, Y1, Margin) for t in self.Tanks]
        self.Target.random_init(X0, X1, Y0, Y1, Margin)
        for t in self.Tanks:    t.Hit = False
        [a.reset(training) for a in agents]
        self.T = self.TimeHorizon
        self.Side = 0
        
    def turn(self):

        # reset things
        for tank in self.Tanks:
            tank.Hit = False
            tank.Reward = 0.0
            tank.FellOff = False

        done = False
        
        actions = [self.Agents[side].action(self.observation(side)) for side in (0,1)]
        
        # Launch projectiles
        for action, tank in zip(actions, self.Tanks):
            if action == FIRE:
                tank.fire()
                tank.Reward += self.MissReward

        # move projectiles
        for side in (0,1):
            tank = self.Tanks[side]
            other = self.Tanks[1-side]
            p = tank.Projectile
            if p is not None:
                hit_tank = p.move(tank, [other, self.Target])
                if p.out_of_bounds() or hit_tank:
                    tank.Projectile = None
                if hit_tank:
                    hit_tank.Hit = True
                    tank.Reward += self.WinReward
                    hit_tank.Reward += self.LooseReward
                    done = True
        
        # move tanks
        if not done:
            for side in (0,1):
                tank = self.Tanks[side]
                other = self.Tanks[1-side]
                if not tank.Hit:
                    done = fell = tank.move(action)
                    tank.FellOff = fell
                    if fell:
                        tank.Reward += self.LooseReward
                        # other.Reward += self.WinReward    do not reward the other tank
        if not done:
            self.T -= 1
            if self.T <= 0:
                done = True

        for tank, agent in zip(self.Tanks, self.Agents):
            agent.update(reward=tank.Reward)

        if done:
            for side in (0,1):
                obs = self.observation(side)
                self.Agents[side].done(obs)

        return done
            
    def render(self):
        if self.Viewer is None:
            self.Viewer = Viewer(600, 600)
            self.Frame = self.Viewer.frame(0.0, 1.0, 0.0, 1.0)
            
            self.TankSprites = []
            self.Projectiles = []
            self.TankBodies = []
            self.TankColors = [
                (0.0, 0.5, 0.1),
                (0, 0.1, 0.7)
            ]
            self.StatusLabels = {}
            for i,tank in enumerate(self.Tanks):
                agent = self.Agents[i]
                sprite = Frame()
                color = self.TankColors[i]
                body = Polygon([(-0.02, -0.01), (0.02, 0.0), (-0.02, 0.01)]).color(*color)
                sprite.add(body)
                self.Frame.add(sprite)
                
                projectile = Rectangle(-0.005, 0.005, -0.001, 0.001, transient=True).color(255,255,255)
                self.Projectiles.append(projectile)

                status = Text("", anchor_x="center", anchor_y="top", size=12, color=(255,255,255))
                self.StatusLabels[agent.ID] = status
                self.Frame.add(status)
                
                self.TankSprites.append(sprite)
                self.Projectiles.append(projectile)
                self.TankBodies.append(body)

            self.TargetSprite = Circle(TargetSize, filled=True).color(0.4, 0.4, 0.3)
            self.Frame.add(self.TargetSprite)

        self.TargetSprite.move_to(self.Target.X, self.Target.Y)
        hit = False
        for i, (t, s, p, d) in enumerate(zip(self.Tanks, self.TankSprites, self.Projectiles, self.TankBodies)):
            agent = self.Agents[i]
            s.move_to(t.X, t.Y)
            s.rotate_to(t.Angle)
            proj = t.Projectile
            if proj is not None:
                self.Frame.add(p)
                p.move_to(proj.X, proj.Y).rotate_to(proj.Angle)
            if t.Hit:   
                d.color(1.0,0.5,0.1)
            else:
                d.color(*self.TankColors[i])
            hit = hit or t.Hit
            status = self.StatusLabels[agent.ID]
            status.Text = "%.3f" % (agent.EpisodeReward,)
            status.move_to(t.X, t.Y - 0.03)
        if self.Target.Hit:
            hit = True
            self.TargetSprite.color(1.0, 0.5, 0.3)
        else:
            self.TargetSprite.color(0.4, 0.4, 0.3)
            
        #self.ScoreText.Text = "r:%.3f R:%.3f %s" % ([], self.EpisodeReward, self.observation())
            
        self.Viewer.render()
        time.sleep(0.03)
        if hit:
            time.sleep(1.5)
        


        
        
        
        

