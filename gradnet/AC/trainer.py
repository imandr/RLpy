#
# Based on sample code provided with Keras documentation
#

import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class TrainerBase(object):
    
    def __init__(self, agents, keep_ratio):
        self.Agents = agents
        self.KeepRatio = keep_ratio

    def train_on_buffer(self, buf, brain, episodes_per_batch, steps_per_batch, callbacks):
        bufsize = len(buf)
        if episodes_per_batch is None and steps_per_batch is None:
            episodes_per_batch = 100
        kept_episodes = []
        #print("train_on_buffer: history length:", len(buf), "    episodes_per_batch:", episodes_per_batch, "  keep ratio:", self.KeepRatio)
        total_steps = 0
        while buf:
            batch = []
            batch_steps = 0
            done = False
            while buf and not done:
                episode = buf.pop()
                n = len(episode["actions"])
                batch.append(episode)
                
                batch_steps += n
                done = (
                    (steps_per_batch is not None and batch_steps >= steps_per_batch)
                    or (episodes_per_batch is not None and len(batch) >= episodes_per_batch)
                )
                
            if done:
                batch_steps, stats = brain.train_on_multi_episode_history(batch)
                total_steps += batch_steps
                for callback in callbacks:
                    if hasattr(callback, "train_batch_end"):
                        callback.train_batch_end(brain, self.Agents, len(batch), batch_steps, stats)
                for episode in batch:
                    if random.random() < self.KeepRatio:
                        kept_episodes.append(episode)
            else:
                # buf exhausted
                assert not buf
                kept_episodes += batch
        #print("train_on_buffer: buffer size:", bufsize, "->", len(kept_episodes))
        return kept_episodes 

class Trainer(TrainerBase):
    
    def __init__(self, agent, alpha=0.01, replay_ratio = 0.1):
        TrainerBase.__init__(self, agent, alpha)
        self.Agent = agent
        self.Alpha = alpha       # smooth constant for running reward
        self.KeepRatio = replay_ratio
        self.HistoryBuffer = []
        
    def callbacks(self, callbacks, method, *params, **args):
        for callback in callbacks:
            if hasattr(callback, method):
                getattr(callback, method)(*params, **args)
        
        
    def train(self, env, target_reward=None, max_episodes=None, max_steps_per_episode=None,
            episodes_per_batch=None, steps_per_batch=None, callbacks=[]):

        self.HistoryBuffer = []
        rewards_history = []

        episodes = 0
        while max_episodes is None or episodes < max_episodes:
            self.callbacks(callbacks, "before_train_episode", self.Agent)
            history = self.Agent.play_episode(env, max_steps_per_episode, training=True)
            episode_reward = self.Agent.EpisodeReward
            running_reward = self.Agent.RunningReward
            self.HistoryBuffer.append(history)
            episodes += 1
            rewards_history.append(episode_reward)
            
            self.callbacks(callbacks, "train_episode_end", self.Agent, episode_reward, history)

            self.HistoryBuffer = self.train_on_buffer(self.HistoryBuffer, self.Agent.Brain, 
                    episodes_per_batch, steps_per_batch, callbacks)   
            
            if target_reward is not None and running_reward >= target_reward:
                break
                
        return episodes, running_reward, rewards_history

if __name__ == "__main__":
    import gym
    
    #
    # Example usage
    #
    
    # Configuration parameters for the whole setup
    seed = 42
    gamma = 0.99  # Discount factor for past rewards
    max_steps_per_episode = 10000
    env = gym.make("CartPole-v0")  # Create the environment
    env.seed(seed)

    num_inputs = 4
    num_actions = 2
    num_hidden = 128

    brain = Brain(num_inputs, num_actions, alpha = 0.01, beta=0.005)
    trainer = Trainer()

    nepisodes, reward_achieved, reward_history = trainer.train(env, brain, 195, 
                max_steps_per_episode=max_steps_per_episode)

    print("Running reward achieved:", reward_achieved, "after", nepisodes, "of training")

