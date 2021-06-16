from tank_target_env import TankTargetEnv
from walker_env import WalkerEnv
from AC import Brain, Agent, Trainer
from util import Monitor
import numpy as np
from tensorflow import keras
import sys, getopt

Usage = """
python train.py [-l <file>] [-s <file>] [-w <file>] <environment name>
  -s <file>    - save weights into the file
  -l <file>    - load weights from the file in the beginning
  -w <file>    - equivalent to -s <file> -l <file>
  environment name - either "tanks" or "walker" or any suitable gym environment name
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
    "tanks":    {
        "target":   9.5,
        "beta":     0.5,
        "cutoff":   None
    },
    "walker":   {
        "gamma":    0.9,
        "cutoff":   None,
        "beta":     0.5,
        "entropy_weight":   0.002,
        "target":   9.5,
        "max_episodes":     10000
    },
    "CartPole-v0":  {
        "target":   195.0,
        "max_steps_per_episode":    200
    },
    "*":    {       # default parameters
        "gamma":    0.99,
        "epsilon":  0.0,
        "cutoff":   1,
        "beta":     None,
        "learning_rate":    0.01,
        "entropy_weight":   0.01,
        "critic_weight":    0.5,
        "max_steps_per_episode":    100,
        "max_episodes":     2000
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
hidden = 200
beta = params["beta"]
epsilon = params["epsilon"]
target = params.get("target")
max_episodes = params["max_episodes"]


entropy_weight = params["entropy_weight"]
critic_weight = params["critic_weight"]

optimizer = keras.optimizers.Adagrad(learning_rate=learning_rate)

if env_name == "tanks":
    env = TankTargetEnv()
elif env_name == "walker":
    env = WalkerEnv()
else:
    import gym
    env = gym.make(env_name)
    
monitor = Monitor("monitor.csv", 
    title = "Actor-Criric Reinforced Learning",
    metadata = dict(
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
                "label":        "running average training score",
                "line_width":   2.0
            }            
        ],
        [
            {   "label":    "critic loss"   },
            {   "label":    "actor loss"   }
        ],
        [
            {   "label":    "entropy", "line_width": 1.0   },
            {   "label":    "entropy MA"   }
        ],
        [
            {   "label":    "average reward"},
            {   "label":    "average return"},
            {   "label":    "average value"},
            {   "label":    "average advantage"}
        ]
    ]
)

monitor.start_server(port)

class SaveCallback(object):
    
    def __init__(self, save_to):
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

class UpdateMonitorCallback(object):
    
    PlayInterval = 200
    ReportInterval = 10
    
    def __init__(self, monitor):
        self.NextPlay = self.PlayInterval
        self.NextReport = self.ReportInterval
        self.Episodes = 0
        self.Monitor = monitor
        self.EntropyMA = MovingAverage()
        self.AvgValueMA = MovingAverage()
        self.AvgReturnMA = MovingAverage()
        self.AvgRewardMA = MovingAverage()
        self.AvgAdvantageMA = MovingAverage()

    def train_batch_end(self, brain, agent, batch_episodes, total_steps, stats):
        self.Episodes += batch_episodes
        running_reward = agent.RunningReward
        entropy = -stats["entropy"]
        self.Monitor.add(self.Episodes, {
                "running average training score":   running_reward,
                "critic loss":  stats["critic"],
                "actor loss":  stats["actor"],
                "entropy":  entropy,
                "entropy MA":  self.EntropyMA(entropy),
                "invalid action loss":  stats["invalid_action"],
                "average value":    self.AvgValueMA(stats["average_value"]),
                "average return":    self.AvgReturnMA(stats["average_return"]),
                "average reward":    self.AvgRewardMA(stats["average_reward"]),
                "average advantage":    self.AvgAdvantageMA(stats["average_advantage"])
            })

class Callback(object):
    
    PlayInterval = 200
    ReportInterval = 10
    
    def __init__(self):
        self.NextPlay = self.PlayInterval
        self.NextReport = self.ReportInterval
        self.Episodes = 0

    def train_batch_end(self, brain, agent, batch_episodes, total_steps, stats):
        running_reward = agent.RunningReward
        self.Episodes += batch_episodes
        if self.Episodes >= self.NextReport:
            print(
                ("Episode: %6d  running reward: %8.4f. Losses: actor: %8.4f, critic:%8.4f, entropy:%8.4f." +
                    "  Average: value:%8.4f, return:%8.4f, advantage:%8.4f") % (self.Episodes, running_reward,
                            stats["actor"], stats["critic"], stats["entropy"],
                            stats["average_value"], stats["average_return"], stats["average_advantage"]
                        )
            )
            self.NextReport += self.ReportInterval
        if self.Episodes >= self.NextPlay:
            for _ in range(3):
                data = agent.play_episode(env, max_steps=max_steps_per_episode, render=True, training=False)
                test_reward = agent.EpisodeReward
                print("test reward:", test_reward)
            self.NextPlay += self.PlayInterval

        
    

num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.n

model = None
if hasattr(env, "create_model"):
    model = env.create_model(hidden)
    print("Using model created by the environment")
    model.summary()
    
brain = Brain((num_inputs,), num_actions, model=model, 
    learning_rate=learning_rate, 
    cutoff=cutoff, beta=beta, gamma=gamma,
    optimizer=optimizer, hidden=hidden,
    critic_weight = critic_weight,
    entropy_weight = entropy_weight
    )

if load_from:
    brain.load(load_from)
    print("Model weights loaded from", load_from)

agent = Agent(brain, num_actions)
cb = Callback()
mcb = UpdateMonitorCallback(monitor)
save_cb = SaveCallback(save_to)
trainer = Trainer(agent, replay_ratio=0.1)

trainer.train(env, target, max_episodes=max_episodes, max_steps_per_episode=max_steps_per_episode, callbacks=[cb, save_cb, mcb])

print("--- training ended ---")




