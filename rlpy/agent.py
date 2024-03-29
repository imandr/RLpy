import random, math
import numpy as np
from rlpy.util import CallbackList
from .active_env import ActiveEnvAgent

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
            metadata = meta_history,
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
        
            
class MultiAgent(ActiveEnvAgent):
    
    # non-RNN based multiagent

    Index = 0

    def __init__(self, brain, alpha=0.01, id=None):
        self.Brain = brain
        self.Alpha = alpha
        self.EpisodeRewardMA = 0.0                # episode reward moving average, calculated using Alpha
        self.EpisodeReward = 0.0                # reward accumulated since for the episode
        self.StepReward = None                  # reward accumulated since last action, None before first action, then a float
        self.StepRecord = None                  # (observaiton, action, step reward)
        self.Training = None
        self.TrainingHistory = []           # [(observation, valid_actions, action, step reward)]
        if id is None:
            id = MultiAgent.Index
            MultiAgent.Index += 1
        self.ID = id
        self.LastStepReward = 0.0

    def reset(self, training=True):
        #print("---------------------------------------------------")
        #print("agent.reset()")
        self.Training = training
        self.EpisodeReward = 0.0
        self.StepReward = None          # will be reset to 0 by action()
        self.Values = []                # values and probs used to calculate action
        self.Probs = []                 # values and probs used to calculate action
        self.Metadata = []
        self.Done = False
        self.LastAction = None
        self.LastStepReward = 0.0
        #print("Agent[%d].reset()" % (id(self)%100,))
        self.LastMetadata = None
        self.TrainingHistory = []           # [(observation, valid_actions, action, step reward)]
        self.ActionData = None                  # (observaiton, action, valid_actions) - data used to calculate the action
        
    def action(self, observation, valid_actions=None, metadata=None):
        # this is reward for the previuous action
        #print("agent.action()")
        self.StepReward = 0.0
        value, probs, action = self.Brain.action(observation, valid_actions, self.Training)
        self.ActionData = (observation, valid_actions, action)       # init, will be complete by end_turn()
        self.Values.append(value)
        self.Probs.append(probs)
        return action

    def update(self, observation=None, reward=None, metadata=None):
        # ignore intermediate observations, can be used by RNN-based agents
        #print("agent.update()", f"reward {reward}" if reward is not None else "")
        if self.StepReward is not None and reward:
            self.StepReward += reward

    def end_turn(self):
        if self.StepReward is not None:
            self.EpisodeReward += self.StepReward
            self.TrainingHistory.append(self.ActionData + (self.StepReward,))
            self.StepReward = None

    def end_episode(self):
        self.EpisodeRewardMA += self.Alpha*(self.EpisodeReward - self.EpisodeRewardMA)
        
    def episode_history(self):
        observations, valids, actions, rewards = zip(*self.TrainingHistory)
        if all(v is None for v in valids):
            valids = None
        out = {
            "nsteps":           len(actions),
            "rewards":          np.array(rewards),
            "actions":          np.array(actions),
            "valid_actions":    valids,
            "observations":     observations,              # list of ndarrays
            "probs":            np.array(self.Probs),
            "values":           np.array(self.Values)
        }
        #print("Agent[%d].history():" % (id(self)%100,), *((k, len(lst) if lst is not None else "none") for k, lst in out.items()))
        #print("          valids:", out["valids"])
        #print("    observations:", out["observations"])
        #for obs, a, r in zip(out["observations"], out["actions"], out["rewards"]):
        #    print("    ", obs, a, r)
        #print("Multiagent: episode_history: shapes:", [(name, x.shape) for name, x in out.items()])
        return out

        
