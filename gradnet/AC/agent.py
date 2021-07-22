import random, math
import numpy as np

class Agent(object):

    def __init__(self, brain, nactions, alpha=0.01):
        self.Brain = brain
        self.NActions = nactions
        self.RunningReward = None
        self.Alpha = alpha
        self.EpisodeReward = 0.0
        self.resetRunningReward()

    def resetRunningReward(self):
        self.RunningReward = None
                
    def play_episode(self, env, max_steps=None, render=False, training=False):
        action_probs_history = []
        rewards_history = []
        action_history = []
        observation_history = []
        valids_history = []
        
        if False:
            print("---")
        self.Brain.reset_episode()

        tup = env.reset()
        # the env may return either just the state, or a tuple, (state, metadata)
        if isinstance(tup, tuple) and len(tup) == 2:
            state, meta = tup
        else:
            state, meta = tup, {}
        #print("p:", " ", state)
        self.EpisodeReward = 0.0
        done = False
        steps = 0
        
        while (not done) and (max_steps is None or steps < max_steps):
            #print("calculating probs...")
            if not isinstance(state, list):
                state = [state]                 # always a list of np arrays
                
            probs, value = self.Brain.evaluate_single(state)
            valid_actions = meta.get("valid_actions")       # None if not there
            action = self.Brain.policy(probs, training, valid_actions=valid_actions)
            
            observation_history.append(state)
            action_history.append(action)
            action_probs_history.append(probs)
            
            new_state, reward, done, meta = env.step(action)
            if False:
                print("state:", state, "  probs:", probs, "  action:", action, "  reward:", reward, " new state:", new_state, "  done:", done)
            #print("p:", action, state, reward, done)
            
            state = new_state
            
            if valid_actions is not None:
                # assume it's consistent through the episode
                valids_history.append(valid_actions)

            if render:
                env.render()
            
            rewards_history.append(reward)
            self.EpisodeReward += reward
            steps += 1
            #print("Agent.play_episode: done:", done)
        #print("returning")
        
        if self.RunningReward is None:
            self.RunningReward = self.EpisodeReward
        else:
            self.RunningReward += self.Alpha*(self.EpisodeReward - self.RunningReward)
        #print("Agent.play_episode: episode reward:", self.EpisodeReward, "  running reward ->", self.RunningReward)
        self.EpisodeHistory = dict(
            rewards = np.array(rewards_history),
            observations = observation_history,
            actions = np.array(action_history),
            probs = np.array(action_probs_history),
            episode_reward = self.EpisodeReward,
            valids = np.array(valids_history) if valids_history else None      
        )
        
        return self.EpisodeHistory
        
    def episode_history(self):
        return self.EpisodeHistory
        
    def train_on_multi_episode_history(self, multi_ep_history):
        return self.Brain.train_on_multi_episode_history(multi_ep_history)
        
            
class MultiAgent(object):

    def __init__(self, brain, nactions, alpha=0.01):
        self.Brain = brain
        self.NActions = nactions
        self.RunningReward = 0.0
        self.EpisodeReward = 0.0
        self.Alpha = alpha
        self.Reward = 0.0           # reward accunulated since last action
        self.resetRunningReward()

    def resetRunningReward(self):
        self.RunningReward = 0.0
        
    def reset(self, training=True):
        self.Training = training
        self.EpisodeReward = self.Reward = 0.0
        self.Observations = []
        self.States = []
        self.Rewards = []
        self.Actions = []
        self.ValidActions = []
        self.Values = []
        self.ProbsHistory = []
        self.ValidProbsHistory = []
        self.FirstMove = True
        self.Done = False
        #print("Agent[%d].reset()" % (id(self)%100,))
        
    def choose_action(self, observation, available_actions=None):
        self.Observations.append(observation)
        self.ValidActions.append(available_actions)
        state = observation
        if not isinstance(state, list):     # state can be a list of np.arrays
            state = [state]
        self.States.append(state)
        self.ValidActions.append(available_actions)
        probs = self.Brain.probs(state)
        valid_probs = self.Brain.policy(probs, self.Training, available_actions)
        self.ProbsHistory.append(probs)      
        self.ValidProbsHistory.append(valid_probs)      
        action = np.random.choice(self.NActions, p=valid_probs)
        self.Actions.append(action)
        return action
        
    def action(self, observation, available_actions, training=True):
        # this is reward for the previuous action
        if not self.FirstMove:
            self.Rewards.append(self.Reward)
        self.FirstMove = False
        self.EpisodeReward += self.Reward
        self.Reward = 0.0
        action = self.choose_action(observation, available_actions, training)
        #print("Agent[%d].action() -> %d" % (id(self)%100, action))
        return action

    def reward(self, reward):
        self.Reward += reward
        #print("Agent[%d].reward(%.4f) accumulated=%.4f" % (id(self)%100, reward, self.Reward))
        
    def done(self, last_observation):
        #print("Agent[%d].done()" % (id(self)%100, ))
        #print("Agent[%d].done(reward=%f)" % (id(self)%100, reward))
        #print("Agent", id(self)%10, "done:", reward)
        if not self.Done:
            if not self.FirstMove:
                self.Rewards.append(self.Reward)
            self.EpisodeReward += self.Reward
            self.Reward = 0.0
            self.RunningReward += self.Alpha*(self.EpisodeReward - self.RunningReward)
            self.Done = True
        #self.Observations.append(observation)
        
    def episode_history(self):
        valids = np.array(self.ValidActions) if self.ValidMoves[0] is not None else None
        out = {
            "rewards":          np.array(self.Rewards),
            "actions":          np.array(self.Actions),
            "valid_actions":    valids,
            "observations":     np.array(self.Observations),
            "probs":            np.array(self.ProbsHistory),
            "valid_probs":      np.array(self.ValidProbsHistory)
        }
        #print("Agent[%d].history():" % (id(self)%100,), *((k, len(lst) if lst is not None else "none") for k, lst in out.items()))
        #print("          valids:", out["valids"])
        #print("    observations:", out["observations"])
        #for obs, a, r in zip(out["observations"], out["actions"], out["rewards"]):
        #    print("    ", obs, a, r)
        #print("Multiagent: episode_history: shapes:", [(name, x.shape) for name, x in out.items()])
        return out

        
