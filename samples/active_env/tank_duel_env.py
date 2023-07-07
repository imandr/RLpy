import random
import numpy as np
import math, time
from gym import spaces
from draw2d import Viewer, Frame, Line, Polygon, Circle, Text
from rlpy import ActiveEnvironment

FireRange = 0.2
TargetSize = 0.01

X0 = 0.0
X1 = 1.0
Y0 = 0.0
Y1 = 1.0
Margin = 0.1


class Tank(object):

    ActionCapacity = 2
    
    def __init__(self):
        self.X = self.Y = None
        self.Angle = None
        self.Reward = 0.0       # accumulated since last action
        self.Hit = False

    def random_init(self, x0, x1, y0, y1, margin):
        self.X = margin + random.random()*(x1-x0-margin*2)
        self.Y = margin + random.random()*(y1-y0-margin*2)
        self.Angle = random.random()*2*math.pi - math.pi
        self.Reward = 0.0
        self.Fire = self.Hit = False
        
    def hit(self, other):
        dx = other.X - self.X
        dy = other.Y - self.Y
        a = math.atan2(dy, dx)
        distance = math.sqrt(dx*dx + dy*dy)
        delta = distance * math.sin(abs(a-self.Angle))
        return abs(self.Angle - a) < math.pi/4 and delta < TargetSize and distance < FireRange + TargetSize
        
class TankDuelEnv(ActiveEnvironment):
    
    Speed = 0.01
    RotSpeed = 5/180.0*math.pi
    BaseReward = 0.0
    MissReward = -0.1
    WinReward = 20.0
    FallReward = -WinReward
    DrawReward = -WinReward
    
    FIRE = 0
    FWD = 1
    LEFT = 2
    RIGHT = 3
    FFWD = 4
    NActions = 5
    BCK = 5

    NState = 9
    ObservationShape = (NState,)

    AvailableMask = np.ones((NActions,))

    def __init__(self, duel=True, target=True, compete=True, time_limit=200):

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
        self.T = self.TimeHorizon = time_limit
        
        self.Tanks = [Tank() for _ in (0,1)]
        self.Target = Tank()
        
    def bind_angle(self, a):
        while a < -math.pi:
            a += math.pi*2
        while a >= math.pi:
            a -= math.pi*2
        return a
        
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
        obs[4] = bearing #- tank.Angle
        obs[5] = other.Angle
        
        dx = self.Target.X - tank.X
        dy = self.Target.Y - tank.Y
        obs[6] = math.sqrt(dx*dx + dy*dy)
        obs[7] = math.atan2(dy, dx) #- tank.Angle
        
        obs[8] = self.T/self.TimeHorizon
        
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
        done = False
        for side in (0,1):
            done = self.move_tank(side)
            if done:
                break
        if not done:
            self.T -= 1
            if self.T <= 0:
                done = True
                for tank, agent in zip(self.Tanks, self.Agents):
                    agent.update(reward=self.DrawReward)
                    tank.Reward += self.DrawReward
        if done:
            for side in (0,1):
                obs = self.observation(side)
                self.Agents[side].done(obs)
        return done
            
        
    def move_tank(self, side):
        #print("turn: side:", side, "   accumulated tank rewards:", [t.Reward for t in self.Tanks])
        other_side = 1-side
        tank = self.Tanks[side]
        other = self.Tanks[other_side]
        agent = self.Agents[side]
        other_agent = self.Agents[other_side]
        obs = self.observation(side)
        #print("side:", side, " tank.Reward since last action:", tank.Reward)
        action = agent.action(obs)

        tank.Reward = reward = self.BaseReward
        other_reward = 0.0
        
        tank.Fire = False        # for viewing
        done = False
        hit = ""

        if action in (self.FWD, self.FFWD, self.BCK):
            if action == self.FFWD: d = self.Speed * 2
            elif action == self.BCK: d = -self.Speed * 0.67
            else: d = self.Speed            # FWD
            x = tank.X + math.cos(tank.Angle) * d
            y = tank.Y + math.sin(tank.Angle) * d
            x1 = max(X0, min(X1, x))
            y1 = max(Y0, min(Y1, y))
            if x1 != x or y1 != y:  # bump ?
                reward = self.FallReward
                done = True
            tank.X, tank.Y = x1, y1
            #self.Reward += 0.001
        elif action == self.FIRE:
            tank.Fire = True
            if self.Duel and tank.hit(other):
                #print(f"hit {side} -> {other_side}")
                hit = f"{side}->{other_side}"
                other.Hit = True
                reward = self.WinReward
                other_reward = -self.WinReward
                done = True
            elif self.HitTarget and tank.hit(self.Target):
                #print(f"hit {side} -> target")
                hit = f"{side}->target"
                reward = self.WinReward
                self.Target.Hit = True
                if self.Compete:
                    other_reward = -self.WinReward
                done = True
            else:
                #print(f"miss {self.Side}")
                reward = self.MissReward
        elif action == self.LEFT:
            tank.Angle += self.RotSpeed
            tank.Angle = self.bind_angle(tank.Angle)
        elif action == self.RIGHT:
            tank.Angle -= self.RotSpeed
            tank.Angle = self.bind_angle(tank.Angle)

        agent.update(reward=reward)
        other_agent.update(reward=other_reward)

        if done:
            print("done: rewards:", reward, other_reward, "hit:", hit)

        return done
           
    def render(self):
        if self.Viewer is None:
            self.Viewer = Viewer(600, 600)
            self.Frame = self.Viewer.frame(0.0, 1.0, 0.0, 1.0)
            
            self.TankSprites = []
            self.TankBeams = []
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
                beam = Line(end=(FireRange, 0)).color(1.0, 0.5, 0.0)
                sprite.add(beam)
                self.Frame.add(sprite)

                status = Text("", anchor_x="center", anchor_y="top", size=12, color=(255,255,255))
                self.StatusLabels[agent.ID] = status
                self.Frame.add(status)
                
                self.TankSprites.append(sprite)
                self.TankBeams.append(beam)
                self.TankBodies.append(body)

            self.TargetSprite = Circle(TargetSize, filled=False, width=2, transient=True).color(1.0, 0.5, 0.3)
            self.HitTargetSprite = Circle(TargetSize, filled=True, transient=True).color(1.0, 0.5, 0.3)

        hit = False
        for i, (t, s, b, d) in enumerate(zip(self.Tanks, self.TankSprites, self.TankBeams, self.TankBodies)):
            agent = self.Agents[i]
            s.move_to(t.X, t.Y)
            s.rotate_to(t.Angle)
            if t.Fire:
                b.show()
            else:
                b.hide()
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
            self.Frame.add(self.HitTargetSprite, at=(self.Target.X, self.Target.Y))
        else:
            self.Frame.add(self.TargetSprite, at=(self.Target.X, self.Target.Y))
            
        #self.ScoreText.Text = "r:%.3f R:%.3f %s" % ([], self.EpisodeReward, self.observation())
            
        self.Viewer.render()
        time.sleep(0.03)
        if hit:
            time.sleep(1.5)
        


        
        
        
        

