import tensorflow as tf
import numpy as np
from .ac import MultiAgent
from .trainer import TrainerBase
import random

class MultiTrainer_Chain(TrainerBase):
    
    #
    # Train only first agent and then propagate the changes to all other changes, applying alpha^n factor to the deltas
    #
    
    def __init__(self, env, brains, replay_keep_ratio = 0.1, alpha = 0.9, update_interval_episodes = 100):
        agents =[MultiAgent(b, env.action_space.n) for b in brains]
        TrainerBase.__init__(self, agents, replay_keep_ratio)
        self.Env = env
        self.HistoryBuffer = []        # {id(brain) -> [episode_history,...]}
        self.Alpha = alpha
        self.NextUpdate = self.UpdateInterval = update_interval_episodes
        self.Episodes = 0
        
    def propagate(self, b0, brains):
        alpha = self.Alpha
        assert b0 is self.Agents[0].Brain
        for b in brains:
            if not b is b0:
                b.update_from_brain(b0, alpha)
                print("Brain %d propagated --> %d with alpha %.3e" % (id(b0)%100, id(b)%100, alpha))
                alpha *= self.Alpha
        
    def train(self, target_reward=None, max_episodes=None, max_steps_per_episode=None, episodes_per_batch=30,
            callbacks=[]):
            
        done = False
        episodes = 0
        while not done:
            self.Env.run(self.Agents, callbacks)
            a0 = self.Agents[0]
            h = a0.episode_history()
            #print("MultiTrainer_Chain.train(): episode rewards:", [a.EpisodeReward for a in self.Agents])
            self.HistoryBuffer.append(h)
            if len(self.HistoryBuffer) >= episodes_per_batch:
                self.HistoryBuffer = self.train_on_buffer(self.HistoryBuffer, a0.Brain, episodes_per_batch, callbacks)
            episodes += 1
            self.Episodes += 1
            if self.Episodes >= self.NextUpdate:
                self.propagate(a0.Brain, [a.Brain for a in self.Agents])
                self.NextUpdate += self.UpdateInterval
            done = max_episodes is not None and episodes >= max_episodes
            #print("MultiTrainer_Chain.train: episodes=", episodes, "  self.Episodes=", self.Episodes, "  max_episodes=", max_episodes)
            maxrunning = max(a.RunningReward for a in self.Agents)
            done = done or target_reward is not None and maxrunning >= target_reward
        return maxrunning