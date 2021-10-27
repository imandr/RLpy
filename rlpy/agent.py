import random, math
import numpy as np
from .util import CallbackList

class Agent(object):

    def __init__(self, brain, alpha=0.001):
        self.Brain = brain
        self.RunningReward = None
        self.Alpha = alpha
        self.EpisodeReward = 0.0
        self.resetRunningReward()

    def resetRunningReward(self):
        self.RunningReward = 0.0
                
    def play_episode(self, env, max_steps=None, render=False, training=False, callbacks=None):

        callbacks = CallbackList.convert(callbacks)

        action_probs_history = []
        rewards_history = []
        action_history = []
        #controls_history = []
        observation_history = []
        valids_history = []
        values_history = []
        meta_history = []
        probs_history = []
        

        if False:
            print("--- beginn episode ---")
            
        self.Brain.reset_episode()

        if callbacks:
            callbacks("agent_episode_begin", self, training=training, render=render)

        tup = env.reset()
        # the env may return either just the state, or a tuple, (state, metadata)
        if isinstance(tup, tuple) and len(tup) == 2:
            state, meta = tup
        else:
            state, meta = tup, {}

        if render:
            env.render()
            
        if callbacks:
            callbacks("agent_episode_reset", self, state, meta)

        #print("p:", " ", state)
        
        self.EpisodeReward = 0.0
        done = False
        steps = 0
        
        while (not done) and (max_steps is None or steps < max_steps):
            #print("calculating probs...")
            if not isinstance(state, list):
                state = [state]                 # always a list of np arrays

            observation_history.append(state)
                
            #print("Agent.play_episode: state:", state)
            valid_actions = meta.get("valid_actions")       # None if not there
            #action, controls = self.Brain.policy(probs, means, sigmas, training, valid_actions)

            if valid_actions is not None:
                # assume it's consistent through the episode
                valids_history.append(valid_actions)

            value, probs, action = self.Brain.action(state, valid_actions, training)
            
            # probs is a structure describing the probabilities for various actions here 
            # action can be:
            # int - single discrete action
            # float - single control
            # ndarray of ints - multiple discrete actions
            # ndarray of floats - multiple controls
            # tuple (int or ndarray of ints, float or ndattay of floats)
            
            probs_history.append(probs)
            action_history.append(action)
            values_history.append(value)

            new_state, reward, done, meta = env.step(action)
            
            if callbacks:
                callbacks("agent_episode_step", self, action, new_state, reward, done, meta)

            meta_history.append(meta)

            if False:
                print("state:", state, "  probs:", probs, "  action:", action, "  reward:", reward, " new state:", new_state, "  done:", done)
            #print("p:", action, state, reward, done)
            
            state = new_state
            prev_action = action
            #prev_controls = controls
            
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
            actions = action_history,
            probs = probs_history,
            valid_actions = np.array(valids_history) if valids_history else None,
            meta = meta_history
        )

        if callbacks:
            callbacks("agent_episode_end", self, self.EpisodeReward, self.EpisodeHistory)
        
        return self.EpisodeReward, self.EpisodeHistory
        
    def episode_history(self):
        return self.EpisodeHistory
        
    def train_on_multi_episode_history(self, multi_ep_history):
        return self.Brain.train_on_multi_episode_history(multi_ep_history)
        
            
class MultiAgent(object):

    def __init__(self, brain, alpha=0.01):
        self.Brain = brain
        self.RunningReward = 0.0
        self.EpisodeReward = 0.0
        self.Alpha = alpha
        self.Reward = 0.0           # reward accunulated since last action
        self.Training = None
        self.resetRunningReward()
        self.History = []           # [(observation, action, reward)]

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
        self.ProbsHistory = []
        self.FirstMove = True
        self.Done = False
        self.PrevAction = -1
        #print("Agent[%d].reset()" % (id(self)%100,))
        self.History = []           # [(observation, action, reward)]
        
    def choose_action(self, observation, valid_actions, training):
        _, probs = self.Brain.evaluate_step(self.PrevAction, observation)
        action = self.Brain.policy(probs, training, valid_actions)
        self.ProbsHistory.append(probs)
        return action
        
    def action(self, observation, valid_actions=None):
        # this is reward for the previuous action
        
        if not isinstance(observation, list):     # make sure observation is a list of np.arrays
            observation = [observation]
        
        self.Observations.append(observation)
        if valid_actions is not None:
            self.ValidActions.append(valid_actions)
        if not self.FirstMove:
            self.Rewards.append(self.Reward)
        self.FirstMove = False
        self.EpisodeReward += self.Reward
        self.Reward = 0.0
        action = self.choose_action(observation, valid_actions, self.Training)
        self.Actions.append(action)
        #print("Agent[%d].action() -> %d" % (id(self)%100, action))
        self.PrevAction = action
        self.History.append((observation, action, None))
        return action

    def update(self, observation=None, reward=None):
        if reward: self.Reward += reward
        self.Observation = observation
        self.History.append((observation, None, reward))
        
        #print("Agent[%d].reward(%.4f) accumulated=%.4f" % (id(self)%100, reward, self.Reward))
        
    def done(self, last_observation, reward=0.0):
        #print("Agent[%d].done()" % (id(self)%100, ))
        #print("Agent[%d].done(reward=%f)" % (id(self)%100, reward))
        #print("Agent", id(self)%10, "done:", reward)
        
        self.Reward += reward
        self.EpisodeReward += self.Reward
        
        if not self.Done:
            if not self.FirstMove:
                self.Rewards.append(self.Reward)
            self.Reward = 0.0
            self.RunningReward += self.Alpha*(self.EpisodeReward - self.RunningReward)
            self.Done = True
            self.History.append((last_observation, None, reward))
        #self.Observations.append(observation)
        
    def episode_history(self):
        valids = np.array(self.ValidActions) if self.ValidActions else None
        out = {
            "nsteps":           len(self.Actions),
            "rewards":          np.array(self.Rewards),
            "actions":          np.array(self.Actions),
            "valid_actions":    valids,
            "observations":     self.Observations,              # list of lists of ndarrays
            "probs":            np.array(self.ProbsHistory),
            "full_history":     self.History
        }
        #print("Agent[%d].history():" % (id(self)%100,), *((k, len(lst) if lst is not None else "none") for k, lst in out.items()))
        #print("          valids:", out["valids"])
        #print("    observations:", out["observations"])
        #for obs, a, r in zip(out["observations"], out["actions"], out["rewards"]):
        #    print("    ", obs, a, r)
        #print("Multiagent: episode_history: shapes:", [(name, x.shape) for name, x in out.items()])
        return out

        
