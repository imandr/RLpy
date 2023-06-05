import random, math
import numpy as np
from rlpy.util import CallbackList

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

        rewards_history = []
        action_history = []
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

        #print("Agent.play_episode: state:", state)

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

            observation_history.append(state)
                
            #print("Agent.play_episode: state:", state)
            valid_actions = meta.get("valid_actions")       # None if not there
            #action, controls = self.Brain.policy(probs, means, sigmas, training, valid_actions)

            if valid_actions is not None:
                # assume it's consistent through the episode
                valids_history.append(valid_actions)

            value, probs, action = self.Brain.action(state, valid_actions, training)
            
            # probs and actions are abstract, depending on the game/brain
            
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
            #actions = np.array(actions_history, dtype=np.int),
            probs = np.array(probs_history),
            valid_actions = np.array(valids_history) if valids_history else None,
            meta = meta_history,
            actions = action_history
        )

        if callbacks:
            callbacks("agent_episode_end", self, self.EpisodeReward, self.EpisodeHistory)
        
        return self.EpisodeReward, self.EpisodeHistory
        
    def episode_history(self):
        return self.EpisodeHistory
        
    def train_on_multi_episode_history(self, multi_ep_history):
        return self.Brain.train_on_multi_episode_history(multi_ep_history)
        
            
class MultiAgent(object):

    Index = 0

    def __init__(self, brain, alpha=0.01, id=None):
        self.Brain = brain
        self.Alpha = alpha
        self.RunningReward = 0.0                # episode reward moving average, calculated using Alpha
        self.EpisodeReward = 0.0                # reward accumulated since for the episode
        self.Reward = None                      # reward accumulated since last action, None before first action, then a float
        self.Training = None
        self.resetRunningReward()
        self.History = []           # [(observation, action, reward)]
        if id is None:
            id = MultiAgent.Index
            MultiAgent.Index += 1
        self.ID = id

    def resetRunningReward(self):
        self.RunningReward = 0.0
        
    def reset(self, training=True):
        self.Training = training
        self.EpisodeReward = 0.0
        self.Reward = 0.0
        self.Observations = []
        self.Rewards = []               # Action rewards
        self.Actions = []
        self.Values = []                # values and probs used to calculate action
        self.Probs = []                 # values and probs used to calculate action
        self.ValidActions = []
        self.Done = False
        self.LastAction = None
        #print("Agent[%d].reset()" % (id(self)%100,))
        self.LastMetadata = None
        self.History = []           # [(observation, action, reward)]
        
    def action(self, observation, valid_actions=None, metadata=None):
        # this is reward for the previuous action
        
        self.Observations.append(observation)
        if valid_actions is not None:
            self.ValidActions.append(valid_actions)
        if self.LastAction is not None:
            self.Rewards.append(self.Reward)
        self.EpisodeReward += self.Reward
        value, probs, action = self.Brain.action(observation, valid_actions, self.Training)
        self.Actions.append(action)
        self.Values.append(value)
        self.Probs.append(probs)
        #print("Agent[%d].action() -> %d" % (id(self)%100, action))
        self.LastAction = action
        self.History.append((observation, action, None))
        self.Reward = 0.0
        return action

    def update(self, observation=None, reward=None, metadata=None):
        if reward:
            self.Reward += reward
        if observation is not None:
            self.Observation = observation
            self.History.append((observation, None, reward))
        
        #print("Agent[%d].reward(%.4f) accumulated=%.4f" % (id(self)%100, reward, self.Reward))
        
    def done(self, last_observation, reward=None, metadata=None):
        #print("Agent[%d].done()" % (id(self)%100, ))
        #print("Agent[%d].done(reward=%f)" % (id(self)%100, reward))
        #print("Agent", id(self)%10, "done:", reward)
        
        if reward:
            self.Reward += reward
        self.EpisodeReward += self.Reward
        
        if not self.Done:
            if self.LastAction is not None:
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
            "observations":     self.Observations,              # list of ndarrays
            "probs":            np.array(self.Probs),
            "values":           np.array(self.Values),
            "observation_history":     self.History
        }
        #print("Agent[%d].history():" % (id(self)%100,), *((k, len(lst) if lst is not None else "none") for k, lst in out.items()))
        #print("          valids:", out["valids"])
        #print("    observations:", out["observations"])
        #for obs, a, r in zip(out["observations"], out["actions"], out["rewards"]):
        #    print("    ", obs, a, r)
        #print("Multiagent: episode_history: shapes:", [(name, x.shape) for name, x in out.items()])
        return out

        
