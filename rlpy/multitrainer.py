import tensorflow as tf
import numpy as np
from .agent import MultiAgent
from .trainer import TrainerBase
import random

class MultiTrainer_Chain(TrainerBase):
    
    #
    # Train only first agent and then propagate the changes to all other changes, applying alpha^n factor to the deltas
    #
    
    def __init__(self, env, agents, replay_keep_ratio = 0.1, alpha = None, update_interval_episodes = 500):
        TrainerBase.__init__(self, agents, replay_keep_ratio)
        self.Env = env
        self.HistoryBuffer = []        # {id(brain) -> [episode_history,...]}
        self.Alpha = alpha or 0.5
        self.NextUpdate = self.UpdateInterval = update_interval_episodes
        self.Episodes = 0
        
    def train(self, target_reward=None, max_episodes=None, max_steps_per_episode=None, 
            steps_per_batch=None, episodes_per_batch=30, callbacks=[]):
            
        done = False
        episodes = 0
        while not done:
            self.Env.run(self.Agents, callbacks)
            a0 = self.Agents[0]
            h = a0.episode_history()
            #print("MultiTrainer_Chain.train(): episode rewards:", [a.EpisodeReward for a in self.Agents])
            self.HistoryBuffer.append(h)
            if len(self.HistoryBuffer) >= episodes_per_batch:
                self.HistoryBuffer = self.train_on_buffer(self.HistoryBuffer, a0.Brain, episodes_per_batch, steps_per_batch, callbacks)
            episodes += 1
            self.Episodes += 1
            if self.Episodes >= self.NextUpdate:
                alpha = self.Alpha
                b0 = self.Agents[0].Brain
                for agent in self.Agents:
                    b = agent.Brain
                    if not b is b0:
                        b.update_weights(b0, alpha)
                        alpha *= self.Alpha
                self.NextUpdate += self.UpdateInterval
            done = max_episodes is not None and episodes >= max_episodes
            #print("MultiTrainer_Chain.train: episodes=", episodes, "  self.Episodes=", self.Episodes, "  max_episodes=", max_episodes)
            maxrunning = max(a.RunningReward for a in self.Agents)
            done = done or target_reward is not None and maxrunning >= target_reward
        return maxrunning
        
class MultiTrainer_Independent(TrainerBase):
    
    #
    # Train only first agent and then propagate the changes to all other changes, applying alpha^n factor to the deltas
    #
    
    def __init__(self, env, agents, replay_keep_ratio = 0.1, alpha = 0.9, update_interval_episodes = 100):
        TrainerBase.__init__(self, agents, replay_keep_ratio)
        self.Env = env
        self.HistoryBuffers = {}        # {id(brain) -> [episode_history,...]}
        self.NextUpdate = self.UpdateInterval = update_interval_episodes
        self.Episodes = 0
        
    def train(self, target_reward=None, max_episodes=None, max_steps_per_episode=None, 
            steps_per_batch=None, episodes_per_batch=30, callbacks=[]):
            
        done = False
        episodes = 0
        while not done:
            self.Env.run(self.Agents, callbacks)
            for a in self.Agents:
                episode_history = a.episode_history()
                if episode_history["nsteps"] > 0:
                    aid = id(a)
                    history = self.HistoryBuffers.setdefault(aid, [])
                    history.append(a.episode_history())
                
                    if len(history) >= episodes_per_batch:
                        history = self.train_on_buffer(history, a.Brain, episodes_per_batch, steps_per_batch, callbacks)
                        self.HistoryBuffers[aid] = history
                
            episodes += 1
            self.Episodes += 1

            done = max_episodes is not None and episodes >= max_episodes
            #print("MultiTrainer_Chain.train: episodes=", episodes, "  self.Episodes=", self.Episodes, "  max_episodes=", max_episodes)
            maxrunning = max(a.RunningReward for a in self.Agents)
            done = done or target_reward is not None and maxrunning >= target_reward
        return maxrunning
        
class MultiTrainer_Sync(TrainerBase):
    
    #
    # Train only first agent and then propagate the changes to all other changes, applying alpha^n factor to the deltas
    #
    
    def __init__(self, env, agents, replay_keep_ratio = 0.1, alpha = None, update_interval_episodes = 100):
        TrainerBase.__init__(self, agents, replay_keep_ratio)
        self.Env = env
        self.HistoryBuffers = {}        # {id(brain) -> [episode_history,...]}
        self.Alpha = alpha or 0.7
        self.Episodes = 0
        self.WeightsCentral = None
        
    def train(self, target_reward=None, max_episodes=None, max_steps_per_episode=None, 
            steps_per_batch=None, episodes_per_batch=30, callbacks=[]):
            
        done = False
        episodes = 0
        while not done:
            self.Env.run(self.Agents, callbacks)
            for a in self.Agents:
                episode_history = a.episode_history()
                if episode_history["nsteps"] > 0:
                    aid = id(a)
                    history = self.HistoryBuffers.setdefault(aid, [])
                    history.append(a.episode_history())
                
                    if len(history) >= episodes_per_batch and random.random() < 0.5:        # to make sure agents do not always update at the same time
                        history = self.train_on_buffer(history, a.Brain, episodes_per_batch, steps_per_batch, callbacks)
                        self.HistoryBuffers[aid] = history
                        
                        brain = a.Brain
                        if self.WeightsCentral is None:
                            self.WeightsCentral = brain.get_weights()
                        else:
                            brain.update_weights(self.WeightsCentral, self.Alpha)
                            self.WeightsCentral = brain.get_weights()
                
            episodes += 1
            self.Episodes += 1

            done = max_episodes is not None and episodes >= max_episodes
            #print("MultiTrainer_Chain.train: episodes=", episodes, "  self.Episodes=", self.Episodes, "  max_episodes=", max_episodes)
            maxrunning = max(a.RunningReward for a in self.Agents)
            done = done or target_reward is not None and maxrunning >= target_reward
        return maxrunning