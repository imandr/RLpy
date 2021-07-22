import random
import numpy as np
import math, time
from gym import spaces
from draw2d import Viewer, Frame, Line, Polygon, Circle, Text
from AC import ActiveEnvironment

FireRange = 0.1
TargetSize = 0.01

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

BCK = 5

NState = 9 + NActions*2


class Tank(object):

    ActionCapacity = 2
    
    def __init__(self):
        self.X = self.Y = None
        self.Angle = None
        self.Reward = 0.0       # accumulated since last action
        self.ActionCache = np.ones([NActions,])

    def use_action(self, a):
        self.ActionCache[a] = max(0, self.ActionCache[a]-1)
        
    def credit_action(self, a):
        self.ActionCache[a] = min(self.ActionCapacity, self.ActionCache[a]+1)
        
    def random_init(self, x0, x1, y0, y1, margin):
        self.X = margin + random.random()*(x1-x0-margin*2)
        self.Y = margin + random.random()*(y1-y0-margin*2)
        self.Angle = random.random()*2*math.pi - math.pi
        self.Reward = 0.0
        self.Fire = self.Hit = False
        self.ActionCache = np.ones([NActions,])
        
    def hit(self, other):
        dx = other.X - self.X
        dy = other.Y - self.Y
        a = math.atan2(dy, dx)
        distance = math.sqrt(dx*dx + dy*dy)
        delta = distance * math.sin(abs(a-self.Angle))
        return abs(self.Angle - a) < math.pi/4 and delta < TargetSize and distance < FireRange + TargetSize
        
class TankDuelEnv(ActiveEnvironment):
    
    Speed = 0.01
    RotSpeed = math.pi*2/6/math.pi      # ~ 2pi/10
    TimeHorizon = 100
    BaseReward = 0.0
    FallReward = -1.0
    MissReward = -0.02
    HitReward = 10.0
    
    AvailableMask = np.ones((NActions,))

    def __init__(self, duel=True, target=True):

        self.NActions = NActions
        self.NState = NState


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
        obs[4] = bearing
        obs[5] = other.Angle
        
        dx = self.Target.X - tank.X
        dy = self.Target.Y - tank.Y
        obs[6] = math.sqrt(dx*dx + dy*dy)
        obs[7] = math.atan2(dy, dx)
        
        obs[8] = self.T/self.TimeHorizon
        
        obs[9:9+NActions] = tank.ActionCache
        obs[9+NActions:9+NActions*2] = other.ActionCache
        
        return obs
        
    def seed(self, x):
        pass
        
    def reset(self, agents, training=True):
        self.Agents = agents
        [t.random_init(X0, X1, Y0, Y1, Margin) for t in self.Tanks]
        self.Target.random_init(X0, X1, Y0, Y1, Margin)
        for t in self.Tanks:    t.Hit = False
        [a.reset(training=training) for a in agents]
        self.T = self.TimeHorizon
        self.Side = 0
        
    def turn(self, training):
        done = False
        for side in (0,1):
            done = self.move_tank(side, training)
            if done:
                break
        else:
            self.T -= 1
            if self.T <= 0:
                done = True
        if done:
            for side in (0,1):
                obs = self.observation(side)
                self.Agents[side].done(obs)
        return done
            
        
    def move_tank(self, side, training):
        #print("turn: side:", side, "   accumulated tank rewards:", [t.Reward for t in self.Tanks])
        other_side = 1-side
        tank = self.Tanks[side]
        other = self.Tanks[other_side]
        agent = self.Agents[side]
        other_agent = self.Agents[other_side]
        obs = self.observation(side)
        #print("side:", side, " tank.Reward since last action:", tank.Reward)
        mask = tank.ActionCache > 0
        action = agent.action(obs, mask, training=training)

        tank.use_action(action)
        other.credit_action(action)

        assert mask[action] > 0, "Invalid action {action}, mask:{mask}"
        reward = tank.Reward = other_reward_delta = 0.0
        
        tank.Fire = False        # for viewing
        done = False
        hit = False

        if False and side == 1:
            # debug - make second tank a passive fixed target
            pass
        else:
            if action in (FWD, FFWD, BCK):
                d = self.Speed/2 if action == FWD else (
                    self.Speed*2 if action == FFWD else
                    -self.Speed/2
                )
                x = tank.X + math.cos(tank.Angle)*d
                y = tank.Y + math.sin(tank.Angle)*d
                x1 = max(X0, min(X1, x))
                y1 = max(Y0, min(Y1, y))
                if x1 != x or y1 != y:  # bump ?
                    agent.reward(self.FallReward)
                    done = True
                tank.X, tank.Y = x1, y1
                #self.Reward += 0.001
            elif action == FIRE:
                tank.Fire = True
                if self.Duel and tank.hit(other):
                    print(f"hit {side} -> {other_side}")
                    other.Hit = True
                    agent.reward(self.HitReward)
                    other_agent.reward(-self.HitReward)
                    done = True
                elif self.HitTarget and tank.hit(self.Target):
                    print(f"hit {side} -> target")
                    done = True
                    agent.reward(self.HitReward)
                    other_agent.reward(-self.HitReward)
                    self.Target.Hit = True
                else:
                    #print(f"miss {self.Side}")
                    agent.reward(self.MissReward)
            elif action == LEFT:
                tank.Angle += self.RotSpeed
                tank.Angle = self.bind_angle(tank.Angle)
            elif action == RIGHT:
                tank.Angle -= self.RotSpeed
                tank.Angle = self.bind_angle(tank.Angle)
                
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
            for i,tank in enumerate(self.Tanks):
                sprite = Frame()
                body = Polygon([(-0.02, -0.01), (0.02, 0.0), (-0.02, 0.01)])
                color = self.TankColors[i]
                sprite.add(body.color(*color))
                beam = Line(end=(FireRange, 0)).color(1.0, 0.5, 0.0)
                sprite.add(beam)
                self.Frame.add(sprite)
                self.TankSprites.append(sprite)
                self.TankBeams.append(beam)
                self.TankBodies.append(body)

            self.ScoresText = Text("", anchor_x="left", size=8).color(0.5, 0.5, 0.5)
            self.TargetSprite = Circle(TargetSize, filled=True).color(0.4, 0.4, 0.3)
            self.Frame.add(self.TargetSprite)

        self.TargetSprite.move_to(self.Target.X, self.Target.Y)
        hit = False
        for i, (t, s, b, d) in enumerate(zip(self.Tanks, self.TankSprites, self.TankBeams, self.TankBodies)):
            s.move_to(t.X, t.Y)
            s.rotate_to(t.Angle)
            b.hidden = not t.Fire
            if t.Hit:   
                d.color(1,0.5,0.1)
            else:
                d.color(*self.TankColors[i])
            hit = hit or t.Hit
        if self.Target.Hit:
            hit = True
            self.TargetSprite.color(1.0, 0.5, 0.3)
        else:
            self.TargetSprite.color(0.4, 0.4, 0.3)
        self.ScoresText.Text = "--- hit ---" if hit else ""
            
        #self.ScoreText.Text = "r:%.3f R:%.3f %s" % ([], self.EpisodeReward, self.observation())
            
        self.Viewer.render()
        
        if hit:
            time.sleep(0.5)
        


        
        
        
        

