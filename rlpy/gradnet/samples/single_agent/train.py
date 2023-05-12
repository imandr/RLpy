from tank_target_env import TankTargetEnv
from cartpole_env import CartPoleEnv
from cartpole_cont import CartPoleContinuousEnv
from walker_env import WalkerEnv
from blackjack import SimpleBlackJackEnv
from game11 import Game11Env
from game21 import Game21Env
from ascention import AscentionEnv
from cluster_analysis import ClusterEnv
from cluster_analysis_k import ClusterKEnv
from sequence_env import SequenceEnv
from single_agent_ttt_env import SingleAgentTicTacToeEnv
from counter import CounterEnv
from rlpy.gradnet.AC import Brain, RNNBrain, BrainContinuous, BrainDiscrete, BrainMixed
from rlpy import Agent, Trainer, Callback
from util import Monitor
import numpy as np
import sys, getopt, math
from gradnet.optimizers import get_optimizer

Usage = """
python train.py [-l <file>] [-s <file>] [-w <file>] <environment name>
  -s <file>    - save weights into the file
  -l <file>    - load weights from the file in the beginning
  -w <file>    - equivalent to -s <file> -l <file>
  environment name - "cartpole", "tanks" or "walker" or any suitable gym environment name
"""

class MovingAverage(object):
    def __init__(self, alpha=0.1):
        self.Alpha = alpha
        self.Value = None
        
    def __call__(self, x):
        if self.Value is None:
            self.Value = x
        self.Value += self.Alpha*(x-self.Value)
        return self.Value

EnvParams = {
    "clusters":    {
        "gamma":    0.5,
        "target":   10000.0,
        "max_steps_per_episode":    300,
        "beta":     None,
        "entropy_weight":   0.1,
        "cutoff":   5,
        "hidden":   300,
        "episodes_between_tests":   5,
        "test_episodes":        1,
    },
    "clusters_k":    {
        "critic_weight":    1.0,
        "gamma":    0.99,
        "target":   1000.0,
        "max_steps_per_episode":    300,
        "beta":     None,
        "entropy_weight":   0.01,
        "cutoff":   10,
        "hidden":   200,
        "episodes_between_tests":   10,
        "test_episodes":        1,
    },
    "mnist":    {
        "gamma":    0.0,
        "target":   95,
        "beta":     None,
        "entropy_weight":   0.01,
        "cutoff":   5,
        "hidden":   500
    },
    "tanks":    {
        "target":   9.5,
        "beta":     0.5,
        "entropy_weight":   0.01,
        "cutoff":   10
    },
    "ttt":    {
        "hidden":   400,
        "target":   1000.0,
        "cutoff":   10,
        "entropy_weight":   0.01,
        "invalid_action_weight":    100.0,
        "gamma": -1.0
    },
    "walker":   {
        "gamma":    0.99,
        "cutoff":   10,
        "beta":     None,
        "target":   9.5,
        "max_episodes":     10000,
        "hidden":   500,
        #"critic_weight":    1.0,
        #"actor_weight":    0.5
    },
    "LunarLander-v2":  {
        "target":           195.0,
        "max_steps_per_episode":    200,
        "learning_rate":    0.01,
        "critic_weight":    1.0,
        "entropy_weight":   0.01,
        "actor_weight":     1.0,
        "gamma":            0.95,
        "cutoff":           10,
        "beta":             None
    },
    "CartPole-v0":  {
        "brain":  BrainDiscrete,
        "target":   195.0,
        "max_steps_per_episode":    200,
        "learning_rate":    0.001,
        "critic_weight":    0.5,
        "entropy_weight":   0.01,
        "actor_weight":    1.0,
    },
    "air_defense":  {
        "brain":  None,
        "gamma":    0.9,
        "target":   4.99,
        "max_steps_per_episode":    200,
        "learning_rate":    0.01,
        "critic_weight":    0.5,
        "entropy_weight":   0.0001,
        "actor_weight":    1.0,
        #"cutoff":           1
    },
    "cartpole_mixed":  {
        "brain":  None,
        "gamma":    0.9,
        "target":   4.99,
        "max_steps_per_episode":    500,
        "learning_rate":    0.01,
        "critic_weight":    0.5,
        "entropy_weight":   0.0001,
        "actor_weight":    1.0,
        #"cutoff":           1
    },
    "cartpole":  {
        "brain":  BrainDiscrete,
        "gamma":    0.9,
        "target":   4.99,
        "max_steps_per_episode":    500,
        "learning_rate":    0.01,
        "critic_weight":    0.5,
        "entropy_weight":   0.0001,
        "actor_weight":    1.0,
        #"cutoff":           1
    },
    "cartpole_cont":  {
        "brain":  BrainContinuous,
        "gamma":    0.9,
        "target":   4.99,
        "max_steps_per_episode":    500,
        "learning_rate":    0.01,
        "critic_weight":    0.5,
        "entropy_weight":   0.0001,
        "actor_weight":    1.0,
        #"cutoff":           1
    },
    "sequence":  {
        "brain":  RNNBrain,
        "gamma":    0.9,
        "target":   100,
        "max_steps_per_episode":    50,
        "learning_rate":    0.01,
        "entropy_weight":   0.01,
        "cutoff":           10
    },
    "game11":  {
        "rnn":  True,
        "gamma":    1.0,
        "target":   11.0,
        "max_steps_per_episode":    200,
        "learning_rate":    0.01,
        "critic_weight":    1.0,
        "entropy_weight":   0.01,
        "actor_weight":     1.0,
        "cutoff":           100
    },
    "blackjack":  {
        "rnn":  True,
        "gamma":    1.0,
        "target":   21.0,
        "max_steps_per_episode":    200,
        "learning_rate":    0.01,
        "critic_weight":    1.0,
        "entropy_weight":   0.001,
        "actor_weight":     1.0,
        "cutoff":           100,
        "hidden":           200
    },
    "game21":  {
        "rnn":  True,
        "gamma":    1.0,
        "target":   21.0,
        "max_steps_per_episode":    200,
        "learning_rate":    0.01,
        "critic_weight":    1.0,
        "entropy_weight":   0.01,
        "actor_weight":     1.0,
        "cutoff":           100
    },
    "ascention":  {
        "rnn":  True,
        "gamma":    0.99,
        "target":   1.0,
        "max_steps_per_episode":    200,
        "learning_rate":    0.01,
        "critic_weight":    1.0,
        "entropy_weight":   0.1,
        "actor_weight":     1.0,
        "cutoff":           100
    },
    "counter":  {
        "rnn":  True,
        "gamma":    1.0,
        "target":   0.95,
        "max_steps_per_episode":    200,
        "learning_rate":    0.01,
        "critic_weight":    1.0,
        "entropy_weight":   0.01,
        "actor_weight":     0.5,
        "cutoff":           100
    },
    "*-saved":    {       # default parameters
        "brain":    BrainDiscrete,
        "continuous":   False,
        "hidden":   200,
        "rnn":  False,
        "gamma":    0.99,
        "epsilon":  0.0,
        "cutoff":   1,
        "beta":     None,
        "learning_rate":    0.01,
        "entropy_weight":   0.001,
        "critic_weight":    0.5,
        "actor_weight":    1.0,
        "invalid_action_weight":    5.0,
        "max_steps_per_episode":    100,
        "max_episodes":     100000,
        "steps_per_batch":  20,
        "episodes_between_tests":   1000,
        "test_episodes":        10
    },
    "*":    {       # default parameters
        "brain":    BrainDiscrete,
        "continuous":   False,
        "hidden":   200,
        "rnn":  False,
        "gamma":    0.99,
        "epsilon":  0.0,
        "cutoff":   None,
        "beta":     1.0,
        "learning_rate":    0.01,
        "entropy_weight":   0.001,
        "critic_weight":    0.5,
        "actor_weight":    1.0,
        "invalid_action_weight":    5.0,
        "max_steps_per_episode":    100,
        "max_episodes":     100000,
        "steps_per_batch":  20,
        "episodes_between_tests":   1000,
        "test_episodes":        3
    }
}

np.set_printoptions(precision=4, suppress=True, linewidth=200)

opts, args = getopt.getopt(sys.argv[1:], "w:s:l:")
if not args:
    print(Usage)
    sys.exit(2)
    
opts = dict(opts)
load_from = opts.get("-l") or opts.get("-w")
save_to = opts.get("-s") or opts.get("-w")
env_name = args[0]

params = EnvParams["*"]
params.update(EnvParams.get(env_name, {}))

print(f"Running {env_name} environment with the following parameters:")
for k, v in sorted(params.items()):
    print("  ", k,"=",v)

gamma = params["gamma"]
comment = params.get("comment", "")
learning_rate = params["learning_rate"]
cutoff = params["cutoff"]
max_steps_per_episode = params["max_steps_per_episode"]
port = 8989
hidden = params["hidden"]
with_rnn = params["rnn"]
continuous = params["continuous"]
beta = params["beta"]
epsilon = params["epsilon"]
target = params.get("target")
max_episodes = params["max_episodes"]
steps_per_batch = params["steps_per_batch"]

episodes_between_tests = params["episodes_between_tests"]
test_episodes = params["test_episodes"]

brain_class = params["brain"]

entropy_weight = params["entropy_weight"]
critic_weight = params["critic_weight"]
actor_weight = params["actor_weight"]
invalid_action_weight = params["invalid_action_weight"]

if env_name == "tanks":
    env = TankTargetEnv()
elif env_name == "walker":
    env = WalkerEnv()
elif env_name == "air_defense":
    from air_defense import AirDefenseEnv
    env = AirDefenseEnv()
elif env_name == "ttt":
    env = SingleAgentTicTacToeEnv()
elif env_name == "cartpole_mixed":
    from cartpole_mixed import CartPoleMixedEnv
    env = CartPoleMixedEnv()
elif env_name == "cartpole":
    env = CartPoleEnv()
elif env_name == "cartpole_cont":
    env = CartPoleContinuousEnv()
elif env_name == "blackjack":
    env = SimpleBlackJackEnv()
elif env_name == "ascention":
    env = AscentionEnv()
elif env_name == "counter":
    env = CounterEnv()
elif env_name == "sequence":
    env = SequenceEnv(10, 5)
elif env_name == "mnist":
    from mnist_env import MNISTEnv
    env = MNISTEnv()
elif env_name == "game11":
    env = Game11Env()
elif env_name == "game21":
    env = Game21Env()
elif env_name == "clusters":
    env = ClusterEnv(7,7)
elif env_name == "clusters_k":
    env = ClusterKEnv(4, 4)
else:
    import gym
    env = gym.make(env_name)
    
optimizer = get_optimizer("adagrad", learning_rate=learning_rate, gamma=1.0)   #, momentum=0.5)
#optimizer = get_optimizer("stretch", learning_rate=learning_rate)   #, momentum=0.5)

    
monitor = Monitor("monitor.csv", 
    title = "Actor-Criric Reinforced Learning",
    metadata = dict(
        rnn=with_rnn,
        gamma=gamma,
        comment = comment,
        environment = env_name,
        environment_reward = "",
        learning_rate = learning_rate,
        brain = "AC",
        cutoff = cutoff,
        beta = beta,
        steps_per_episode = max_steps_per_episode,
        optimizer = optimizer.__class__.__name__,
        entropy_weight = entropy_weight,
        critic_weight = critic_weight,
        hidden_layers = hidden,
        max_steps_per_episode = max_steps_per_episode
    ),
    plots=[
        [
            {
                "label":        "running average training reward",
                "line_width":   2.0
            },
            {
                "label":        "training eposode reward",
                "line_width":   1.0
            }
        ],
        [
            {   "label":    "actor loss"   }
        ],
        [
            {   "label":    "critic loss"   }
        ],
        [
            {   "label":    "entropy", "line_width": 1.0   },
            {   "label":    "entropy MA"   }
        ],
        [
            {   "label":    "average reward"},
            {   "label":    "average return"},
            {   "label":    "average value"}
            #{   "label":    "average advantage"}
        ]
    ]
)

monitor.start_server(port)

class SaveCallback(Callback):
    
    def __init__(self, save_to):
        Callback.__init__(self)
        self.BestReward = None
        self.SaveTo = save_to

    def train_batch_end(self, brain, agent, batch_episodes, total_steps, losses):
        running_reward = agent.RunningReward
        if self.BestReward is None:
            self.BestReward = running_reward
        elif running_reward > self.BestReward and self.SaveTo:
            brain.save(self.SaveTo)
            print("Model weights saved to", self.SaveTo, "with best running reward", self.BestReward)
            self.BestReward = running_reward

class UpdateMonitorCallback(Callback):
    
    TestInterval = 1000
    ReportInterval = 10
    
    def __init__(self, monitor):
        Callback.__init__(self)
        self.NextTest = self.TestInterval
        self.NextReport = self.ReportInterval
        self.Episodes = 0
        self.Steps = 0
        self.Monitor = monitor
        self.EntropyMA = MovingAverage()
        self.AvgValueMA = MovingAverage()
        self.AvgReturnMA = MovingAverage()
        self.AvgRewardMA = MovingAverage()
        self.AvgAdvantageMA = MovingAverage()
        self.AvgTestRewardMA = MovingAverage()

    def train_batch_end(self, brain, agent, batch_episodes, steps, stats):
        self.Episodes += batch_episodes
        self.Steps += steps
        running_reward = agent.RunningReward
        entropy = stats["entropy"]
            
        mon_data = {
                "running average training reward":   running_reward,
                "training eposode reward":   agent.EpisodeReward,
                "critic loss":  stats["critic_loss"],
                "actor loss":  stats["actor_loss"],
                "entropy":  entropy,
                "entropy MA":  self.EntropyMA(entropy),
                "invalid action loss":  stats["invalid_action_loss"],
                "average value":    self.AvgValueMA(stats["average_value"]),
                "average return":    self.AvgReturnMA(stats["average_return"]),
                "average reward":    self.AvgRewardMA(stats["average_reward"])
                #"average advantage":    self.AvgAdvantageMA(stats["average_advantage"])
            }        
            
        self.Monitor.add(self.Steps, mon_data)
        

class PrintCallback(Callback):
    
    PlayInterval = 1000
    ReportInterval = 100
    
    def __init__(self):
        Callback.__init__(self)
        self.NextPlay = self.PlayInterval
        self.NextReport = self.ReportInterval
        self.Episodes = 0

    def train_batch_end(self, brain, agent, batch_episodes, total_steps, stats):
        #print("End of batch. Episodes:", batch_episodes, "   steps:", total_steps)
        running_reward = agent.RunningReward
        self.Episodes += batch_episodes
        if self.Episodes >= self.NextReport:
            print(
                ("Episode: %6d  running reward: %8.4f. Losses per step: actor: %8.4f, critic:%8.4f, entropy:%8.4f." +
                    "  Average: value:%8.4f, return:%8.4f, normaized entropy:%8.4f") % (self.Episodes, running_reward,
                            stats["actor_loss"], stats["critic_loss"], stats["entropy_loss"],
                            stats["average_value"], stats["average_return"], stats["entropy"]#, stats["average_advantage"]
                        )
            )
            #print("   rms(grads):", [math.sqrt(g2) for g2 in stats["average_grad_squared"]])
            self.NextReport += self.ReportInterval
            
class TestCallback(Callback):
    
    def __init__(self, monitor=None, test_interval = 1000, test_episodes = 10, render=True):
        Callback.__init__(self, fire_interval = test_interval)
        self.TestEpisodes = test_episodes
        self.Render = render
        self.Monitor = monitor
        
    def train_batch_end(self, brain, agent, batch_episodes, total_steps, stats):
        sum_reward = 0
        for _ in range(self.TestEpisodes):
            agent.play_episode(env, max_steps=max_steps_per_episode, render=self.Render, training=False)
            test_reward = agent.EpisodeReward
            sum_reward += test_reward
            print("test reward:", test_reward)
        avg_test_reward = sum_reward/self.TestEpisodes
        print("Average test reward:", type(avg_test_reward), avg_test_reward)
    

model = None
if hasattr(env, "create_model"):
    model = env.create_model(hidden)
    print("Using model created by the environment")

print("optimizer:", optimizer)


print("Env observation space:", env.observation_space, env.observation_space.shape)

if brain_class is None:
    brain = BrainMixed.from_env(env, 
        cutoff=cutoff, beta=beta, gamma=gamma,
        hidden=hidden,
        optimizer = optimizer,
        actor_weight = actor_weight,
        critic_weight = critic_weight,
        entropy_weight = entropy_weight,
        invalid_action_weight = invalid_action_weight
    )
else:
    brain = brain_class(env.observation_space.shape, env.NActions, model=model,
        cutoff=cutoff, beta=beta, gamma=gamma,
        hidden=hidden,
        optimizer = optimizer,
        actor_weight = actor_weight,
        critic_weight = critic_weight,
        entropy_weight = entropy_weight,
        invalid_action_weight = invalid_action_weight
        )

if load_from:
    brain.load(load_from)
    print("Model weights loaded from", load_from)

agent = Agent(brain)
cb = PrintCallback()
mcb = UpdateMonitorCallback(monitor)
test_cb = TestCallback(monitor, test_interval=episodes_between_tests, test_episodes=test_episodes)
save_cb = SaveCallback(save_to)

trainer = Trainer(agent, replay_ratio=0.1)

episodes, running_reward, rewards_history = trainer.train(env, target, 
    min_episodes=100, max_episodes=max_episodes, max_steps_per_episode=max_steps_per_episode, 
    steps_per_batch=steps_per_batch, callbacks=[cb, save_cb, mcb, test_cb])

print("--- training ended ---")
print("  episodes:      ", episodes)
print("  running reward:", running_reward)




