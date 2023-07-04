import numpy as np
from .agent import MultiAgent
from .trainer import TrainerBase
from .util import CallbackList
import random

class MultitrainerBase(TrainerBase):
    
    def __init__(self, env, agents, replay_keep_ratio = 0.1):
        TrainerBase.__init__(self, replay_keep_ratio)
        self.Env = env
        self.Agents = agents

    # overridable
    def get_weights(self):
        return self.Agents[0].Brain.get_weights()
        
class MultiTrainer_Chain(MultitrainerBase):
    
    #
    # Train only first agent and then propagate the changes to all other changes, applying alpha^n factor to the deltas
    # if ring is True, update the first brain with updated weights from the last one
    #
    
    def __init__(self, env, agents, replay_keep_ratio = 0.1, alpha = None, update_interval_episodes = 500, ring=False):
        MultitrainerBase.__init__(self, env, agents, replay_keep_ratio)
        self.Alpha = alpha or 0.5
        self.NextUpdate = self.UpdateInterval = update_interval_episodes
        self.Episodes = 0
        self.Ring = ring
    
    def train(self, target_reward=None, max_episodes=None, max_steps=None,
            max_steps_per_episode=None, 
            steps_per_batch=None, episodes_per_batch=30, callbacks=None):
            
        callbacks = CallbackList.convert(callbacks)            # if callbacks is a list, convert it to the Callbacks object
            
        done = False
        episodes = 0
        while not done:
            self.Env.run(self.Agents, callbacks)
            if callbacks is not None:
                for agent in self.Agents:
                    callbacks("train_episode_end", agent, agent.EpisodeReward, agent.episode_history())
                callbacks("active_env_episode_end", self.Agents)
            for a in self.Agents:
                self.remember_episode(a.episode_history())
            episodes_trained, steps_trained = self.train_on_buffer(self.Agents[0], callbacks = callbacks, 
                episodes_per_batch = episodes_per_batch, steps_per_batch = steps_per_batch, 
                max_steps = max_steps, max_episodes = max_episodes)
            episodes += episodes_trained
            self.Episodes += episodes_trained
            if self.Episodes >= self.NextUpdate:
                alpha = self.Alpha
                b0 = self.Agents[0].Brain
                w = b0.get_weights()
                for agent in self.Agents[1:]:
                    b = agent.Brain
                    b.update_weights(w, alpha)
                    w = b.get_weights()
                if self.Ring:
                    b0.update_weights(w, alpha)
                self.NextUpdate += self.UpdateInterval
            done = max_episodes is not None and episodes >= max_episodes
            #print("MultiTrainer_Chain.train: episodes=", episodes, "  self.Episodes=", self.Episodes, "  max_episodes=", max_episodes)
            maxrunning = max(a.EpisodeRewardMA for a in self.Agents)
            done = done or target_reward is not None and maxrunning >= target_reward
        return maxrunning
        
class MultiTrainer_Ring(MultiTrainer_Chain):
    
    def __init__(self, env, agents, **args):
        MultiTrainer_Chain.__init__(self, env, agents, ring=True, **args)
        
class MultiTrainer_Independent(MultitrainerBase):
    
    #
    # Train only first agent and then propagate the changes to all other changes, applying alpha^n factor to the deltas
    #
    
    def __init__(self, env, agents, replay_keep_ratio = 0.1, alpha = 0.9, update_interval_episodes = 100):
        MultitrainerBase.__init__(self, env, agents, replay_keep_ratio)
        self.NextUpdate = self.UpdateInterval = update_interval_episodes
        self.Episodes = 0
        self.ReplayBuffers = [ReplayBuffer() for a in self.Agents]

    def train(self, target_reward=None, max_episodes=None, max_steps=None, 
            max_steps_per_episode=None, 
            steps_per_batch=None, episodes_per_batch=30, callbacks=None):

        if target_reward is None and max_episodes is None and max_steps is None:
            max_steps = 1000

        callbacks = CallbackList.convert(callbacks)            # if callbacks is a list, convert it to the Callbacks object
        done = False
        episodes = 0
        total_steps = 0
        while not done:
            self.Env.run(self.Agents, callbacks)
            for agent, replay_buffer in zip(self.Agents, self.ReplayBuffers):
                replay_buffer.remember_episode(agent.episode_history())
                episodes_trained, steps_trained = self.train_on_buffer(agent,
                            replay_buffer=replay_buffer,
                            episodes_per_batch=episodes_per_batch, steps_per_batch=steps_per_batch, callbacks=callbacks
                    )
                self.HistoryBuffers[aid] = history
                episodes += episodes_trained
                total_steps += steps_trained

            self.Episodes = episodes
            
            done = max_episodes is not None and episodes >= max_episodes or \
                max_steps is not None and total_steps >= max_steps
            #print("MultiTrainer_Chain.train: episodes=", episodes, "  self.Episodes=", self.Episodes, "  max_episodes=", max_episodes)
            maxrunning = max(a.EpisodeRewardMA for a in self.Agents)
            done = done or target_reward is not None and maxrunning >= target_reward
        return maxrunning
        
class MultiTrainer_Sync(MultitrainerBase):
    
    #
    # Train only first agent and then propagate the changes to all other changes, applying alpha^n factor to the deltas
    #
    
    def __init__(self, env, agents, replay_keep_ratio = 0.1, alpha = None, sync_frequency = 0.02):
        MultitrainerBase.__init__(self, env, agents, replay_keep_ratio)
        self.HistoryBuffers = {}        # {id(brain) -> [episode_history,...]}
        self.Alpha = alpha or 0.5
        self.Episodes = 0
        self.WeightsCentral = None
        self.SyncFrequency = sync_frequency
        
    def update_central(self, brain, alpha):
        if self.WeightsCentral is None:
            self.WeightsCentral = brain.get_weights()
        else:
            for wc, w in zip(self.WeightsCentral, brain.get_weights()):
                wc += alpha*(w-wc)

    def train(self, target_reward=None, max_episodes=None, max_steps=None,
            max_steps_per_episode=None, 
            steps_per_batch=None, episodes_per_batch=30, callbacks=None):
            
        if target_reward is None and max_episodes is None and max_steps is None:
            max_steps = 1000

        callbacks = CallbackList.convert(callbacks)            # if callbacks is a list, convert it to the Callbacks object
        # init centrals
        self.WeightsCentral = self.Agents[0].Brain.get_weights()
        for a in self.Agents:
            self.update_central(a.Brain, 0.5)
            
        done = False
        episodes = 0
        while not done:
            self.Env.run(self.Agents, callbacks)

            episodes += 1
            self.Episodes += 1
            
            for a in self.Agents:
                episode_history = a.episode_history()
                if episode_history["nsteps"] > 0:
                    aid = id(a)
                    history = self.HistoryBuffers.setdefault(aid, [])
                    history.append(a.episode_history())
                
                    if len(history) >= episodes_per_batch and random.random() < 0.5:        # to make sure agents do not always update at the same time
                        history = self.train_on_buffer(history, a, 
                                episodes_per_batch=episodes_per_batch, 
                                steps_per_batch=steps_per_batch, callbacks=callbacks
                        )
                        self.HistoryBuffers[aid] = history
                        
                        if random.random() < self.SyncFrequency:
                            rmin = rmax = None
                            for agent in self.Agents:
                                r = agent.EpisodeRewardMA
                                if r is not None:
                                    if rmin is None:
                                        rmin = rmax = r
                                    else:
                                        rmin = min(r, rmin)
                                        rmax = max(r, rmax)
                            if rmax is None or rmin == rmax:
                                alpha = 0.5
                            else:
                                alpha = (a.EpisodeRewardMA - rmin)/(rmax - rmin)
                            alpha = alpha * 0.8 + 0.1
                            
                            self.update_central(a.Brain, alpha)
                            a.Brain.update_weights(self.WeightsCentral, 1-alpha)
                            
                            callbacks("agent_synced", a, alpha)

            done = max_episodes is not None and episodes >= max_episodes
            #print("MultiTrainer_Chain.train: episodes=", episodes, "  self.Episodes=", self.Episodes, "  max_episodes=", max_episodes)
            maxrunning = max(a.EpisodeRewardMA for a in self.Agents)
            done = done or target_reward is not None and maxrunning >= target_reward
        return maxrunning