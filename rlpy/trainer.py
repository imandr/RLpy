import random
from .util import CallbackList


class ReplayBuffer(object):
    
    def __init__(self, keep_ratio=0.1):
        self.KeepRatio = keep_ratio
        self.Buffer = []

    def make_batch(self, min_episodes=None, min_steps=None):
        assert min_episodes or min_steps, "Either min_episodes or min_steps needs to be specified"
        kept_episodes = []
        batch = []
        batch_steps = 0
        for episode in self.Buffer:
            nsteps = len(episode["actions"])
            batch.append(episode)
            batch_steps += nsteps
            if random.random() < self.KeepRatio:
                kept_episodes.append(episode)
            if (min_steps is None or batch_steps >= min_steps) and (min_episodes is None or len(batch) >= min_episodes):
                break
        else:
            return None         # not enough episodes
        self.Buffer = self.Buffer[len(batch):] + kept_episodes
        return batch
        
    def append(self, episode):
        self.Buffer.append(episode)

    def batches(self, min_episodes=None, min_steps=None):
        while True:
            batch = self.make_batch(min_episodes, min_steps)
            if batch is None:
                break
            yield batch


class TrainerBase(object):
    
    def __init__(self, keep_ratio=0.1):
        self.ReplayBuffer = ReplayBuffer(keep_ratio)
        
    def remember_episode(self, episode):
        self.ReplayBuffer.append(episode)

    def batches(self, min_episodes=None, min_steps=None):
        return self.ReplayBuffer.batches(min_episodes, min_steps)
        
    def train_on_buffer(self, agent, replay_buffer=None, callbacks = None, episodes_per_batch = None, steps_per_batch = None, max_steps = None, max_episodes = None):
        replay_buffer = replay_buffer or self.ReplayBuffer
        if callbacks:
            callbacks = CallbackList(callbacks)
        brain = agent.Brain
        if episodes_per_batch is None and steps_per_batch is None:
            episodes_per_batch = 10
        steps_trained = 0
        episodes_trained = 0
        for batch in replay_buffer.batches(episodes_per_batch, steps_per_batch):
            batch_steps, stats = brain.train_on_multi_episode_history(batch)
            steps_trained += batch_steps
            episodes_trained += len(batch)
            if callbacks:
                callbacks("train_batch_end", agent, len(batch), batch_steps, stats)
            if max_steps is not None and steps_trained >= max_steps \
                    or max_episodes is not None and episodes_trained >= max_episodes:
                break
        return episodes_trained, steps_trained

class Trainer(TrainerBase):
    
    def __init__(self, agent, replay_ratio = 0.1):
        TrainerBase.__init__(self, replay_ratio)
        self.Agent = agent

    def train(self, env, max_steps_per_episode=None,
            target_reward=None, max_episodes=None, max_steps=None,
            episodes_per_batch=None, steps_per_batch=None, 
            callbacks=None
        ):
            
        if target_reward is None and max_episodes is None and max_steps is None:
            max_steps = 1000
            
        if episodes_per_batch is None and steps_per_batch is None:
            steps_per_batch = 100
            
        callbacks = CallbackList.convert(callbacks)

        self.HistoryBuffer = []
        rewards_history = []

        episodes = 0
        total_steps = 0
        brain = self.Agent.Brain
        
        while (max_episodes is None or episodes < max_episodes) \
                and (max_steps is None or total_steps < max_steps) \
                and (target_reward is None or self.Agent.RunningReward is None or self.Agent.RunningReward < target_reward):

            episode_reward, history = self.Agent.play_episode(env, max_steps_per_episode, training=True, callbacks=callbacks)
            if callbacks is not None:
                callbacks("train_episode_end", self.Agent, episode_reward, history)
            rewards_history.append(episode_reward)
            self.remember_episode(history)
            
            episodes_trained, steps_trained = self.train_on_buffer(self.Agent, callbacks = callbacks, 
                episodes_per_batch = episodes_per_batch, steps_per_batch = steps_per_batch, 
                max_steps = max_steps, max_episodes = max_episodes)

            episodes += episodes_trained
            total_steps += steps_trained
            
            if max_steps is not None and total_steps >= max_steps \
                    or max_episodes is not None and episodes >= max_episodes:
                break

        return episodes, self.Agent.RunningReward, rewards_history


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

