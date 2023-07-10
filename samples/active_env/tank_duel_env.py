import random
import numpy as np
import math, time
from gym import spaces
from draw2d import Viewer, Frame, Line, Polygon, Circle, Text
from rlpy import ActiveEnvironment



X0 = 0.0
X1 = 1.0
Y0 = 0.0
Y1 = 1.0
Margin = 0.1


def bind_angle(a):
    # returns the same angle but in the range -pi <= a < pi
    while a < -math.pi:
        a += math.pi*2
    while a >= math.pi:
        a -= math.pi*2
    return a


class Target(object):

    TargetSize = 0.01
    
    def __init__(self, x=None, y=None, radius=None):
        self.Radius = radius or self.TargetSize
        self.X = x
        self.Y = y
        self.Dead = False
        self.Hit = False

    def random_init(self, x0, x1, y0, y1, margin):
        self.X = margin + random.random()*(x1-x0-margin*2)
        self.Y = margin + random.random()*(y1-y0-margin*2)
        self.Dead = False
        self.Hit = False

    def hit_by(self, shooter):
        dx = self.X - shooter.X
        dy = self.Y - shooter.Y
        a = math.atan2(dy, dx)
        distance = math.sqrt(dx*dx + dy*dy)
        delta = distance * math.sin(abs(a-shooter.Angle))
        return abs(shooter.Angle - a) < math.pi/4 and delta < self.Radius and distance < shooter.FireRange + self.Radius

class Shooter(Target):

    FireRange = 0.2

    def __init__(self):
        Target.__init__(self)
        self.Angle = None
        self.Fired = False

    def random_init(self, x0, x1, y0, y1, margin):
        Target.random_init(self, x0, x1, y0, y1, margin)
        self.Angle = random.random()*2*math.pi - math.pi
        self.Fired = False
        
    def hit(self, target):
        return target.hit_by(self)
        
class Tank(Shooter):

    # actions
    FIRE = 0
    FWD = 1
    LEFT = 2
    RIGHT = 3
    FFWD = 4
    NActions = 5
    BCK = 5
    Speed = 0.01
    RotSpeed = 5/180.0*math.pi

    def __init__(self, agent):
        Shooter.__init__(self)
        self.Agent = agent

    def move(self, env, action, other, target):
        
        if self.Dead:
            return False, 0.0, 0.0
        
        #print("turn: side:", side, "   accumulated tank rewards:", [t.Reward for t in self.Tanks])
        agent = self.Agent
        other_agent = other.Agent
        #print("side:", side, " tank.Reward since last action:", tank.Reward)

        reward = env.BaseReward
        other_reward = 0.0
        
        self.Fired = False        # for viewing
        done = False
        hit = ""

        if action in (self.FWD, self.FFWD, self.BCK):
            if action == self.FFWD: d = self.Speed * 2
            elif action == self.BCK: d = -self.Speed * 0.67
            else: d = self.Speed            # FWD
            x = self.X + math.cos(self.Angle) * d
            y = self.Y + math.sin(self.Angle) * d
            x1 = max(X0, min(X1, x))
            y1 = max(Y0, min(Y1, y))
            if x1 != x or y1 != y:  # bump ?
                reward += env.FallReward
                done = True
            self.X, self.Y = x1, y1
            #self.Reward += 0.001
        elif action == self.FIRE:
            self.Fired = True
            if not other.Dead and self.hit(other):
                #print(f"hit {side} -> {other_side}")
                hit = f"tank hit"
                print("tank hit")
                other.Hit = True
                other.Dead = True
                if env.Duel:
                    done = True
                reward += env.WinReward
                other_reward -= env.WinReward
            elif not target.Dead and env.HitTarget and self.hit(target):
                #print(f"hit {side} -> target")
                hit = f"target hit"
                print("target hit")
                reward += env.WinReward
                target.Hit = True
                target.Dead = True
                if env.Compete:
                    other_reward -= env.WinReward
                done = True
            else:
                reward += env.MissReward
        elif action == self.LEFT:
            self.Angle = bind_angle(self.Angle + self.RotSpeed)
        elif action == self.RIGHT:
            self.Angle = bind_angle(self.Angle - self.RotSpeed)

        if done:
            print("done: rewards:", reward, other_reward, "hit:", hit)

        return done, reward, other_reward


class TankDuelEnv(ActiveEnvironment):
    
    BaseReward = 0.0
    MissReward = -0.1
    WinReward = 20.0
    FallReward = -WinReward
    DrawReward = -WinReward
    
    NActions = Tank.NActions

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
        
        self.Tanks = []
        self.Target = Target()
        
    NState = 10
    ObservationShape = (NState,)

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
        obs[6] = other.Hit and 1.0 or 0.0
        
        dx = self.Target.X - tank.X
        dy = self.Target.Y - tank.Y
        obs[7] = math.sqrt(dx*dx + dy*dy)
        obs[8] = math.atan2(dy, dx) #- tank.Angle
        
        obs[9] = self.T/self.TimeHorizon
        
        return obs
        
    def seed(self, x):
        pass
        
    def reset(self, agents, training=True):
        self.Tanks = [Tank(agent) for agent in agents]
        [t.random_init(X0, X1, Y0, Y1, Margin) for t in self.Tanks]
        self.Target.random_init(X0, X1, Y0, Y1, Margin)
        [a.reset(training) for a in agents]
        self.T = self.TimeHorizon
        
    def turn(self):
        done = False
        
        actions = {}
        observations = {}

        for side, tank in enumerate(self.Tanks):
            if not tank.Dead:
                obs = self.observation(side)
                actions[side] = tank.Agent.action(obs)
                observations[side] = obs
            tank.Fired = tank.Hit = False    # for rendering

        for side, tank in enumerate(self.Tanks):
            if not tank.Dead:
                other = self.Tanks[1-side]
                tank_done, reward, other_reward = tank.move(self, actions[side], other, self.Target)
                done = done or tank_done
                tank.Agent.update(reward=reward)
                other.Agent.update(reward=other_reward)

        if not done:
            self.T -= 1
            if self.T <= 0:
                done = True
                for tank in self.Tanks:
                    tank.Agent.update(reward=self.DrawReward)
        if done:
            for side, tank in enumerate(self.Tanks):
                obs = self.observation(side)
                tank.Agent.done(obs)

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
            for i, tank in enumerate(self.Tanks):
                agent = tank.Agent
                sprite = Frame()
                color = self.TankColors[i]
                body = Polygon([(-0.02, -0.01), (0.02, 0.0), (-0.02, 0.01)]).color(*color)
                sprite.add(body)
                beam = Line(end=(tank.FireRange, 0)).color(1.0, 0.5, 0.0)
                sprite.add(beam)
                self.Frame.add(sprite)

                status = Text("", anchor_x="center", anchor_y="top", size=12, color=(255,255,255))
                self.StatusLabels[agent.ID] = status
                self.Frame.add(status)
                
                self.TankSprites.append(sprite)
                self.TankBeams.append(beam)
                self.TankBodies.append(body)

            self.TargetSprite = Circle(self.Target.Radius, filled=False, width=2, transient=True).color(1.0, 0.5, 0.3)
            self.HitTargetSprite = Circle(self.Target.Radius, filled=True, transient=True).color(1.0, 0.5, 0.3)

        hit = False
        for i, (t, s, b, d) in enumerate(zip(self.Tanks, self.TankSprites, self.TankBeams, self.TankBodies)):
            agent = t.Agent
            s.move_to(t.X, t.Y)
            s.rotate_to(t.Angle)
            if t.Fired:
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
        if self.Target.Dead:
            self.Frame.add(self.HitTargetSprite, at=(self.Target.X, self.Target.Y))
        else:
            self.Frame.add(self.TargetSprite, at=(self.Target.X, self.Target.Y))
            
        #self.ScoreText.Text = "r:%.3f R:%.3f %s" % ([], self.EpisodeReward, self.observation())
            
        self.Viewer.render()
        time.sleep(0.03)
        if hit:
            time.sleep(1.5)
        


        
        
        
        

