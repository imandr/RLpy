from AC import Brain, MultiTrainer_Chain, ActiveEnvironment
import numpy as np, getopt, sys, time
from util import Monitor, MovingAverage
from tensorflow import keras


opts, args = getopt.getopt(sys.argv[1:], "w:s:l:")
opts = dict(opts)
load_from = opts.get("-l") or opts.get("-w")
save_to = opts.get("-s") or opts.get("-w")

np.set_printoptions(precision=4, suppress=True)

#env = ActiveGymEnvironment("CartPole-v0")
#env = ActiveGymEnvironment("Acrobot-v1")
#env = ActiveEnvironment.from_gym_env(MountainCarEnv(), time_limit=200)
env = ActiveEnvironment.from_gym_env("CartPole-v0", time_limit=200)

cutoff = None
gamma = 0.99
beta = 0.5
comment = ""
learning_rate = 0.01
max_steps_per_episode = 300
port = 8989
hidden = 200

entropy_weight = 0.0
critic_weight = 0.5
invalid_action_weight = 10.0
cross_training = 0.01

optimizer = keras.optimizers.Adagrad(learning_rate=learning_rate)

brain = Brain(env.observation_space.shape, env.action_space.n, gamma=gamma, cutoff=cutoff, learning_rate=learning_rate, entropy_weight=entropy_weight,
    optimizer=optimizer, hidden=hidden, beta = beta,
    critic_weight=critic_weight, invalid_action_weight=invalid_action_weight)

trainer = MultiTrainer_Chain(env, [brain], alpha=cross_training)

monitor = Monitor("monitor.csv", 
    title = "Cartpole, Active Environment",
    metadata = dict(
        gamma=gamma,
        trainer=trainer.__class__.__name__,
        comment = comment,
        environment = "Active(CartPole-v0)",
        learning_rate = learning_rate,
        brain = "AC",
        cutoff = cutoff,
        beta = beta,
        optimizer = optimizer.__class__.__name__,
        entropy_weight = entropy_weight,
        critic_weight = critic_weight,
        hidden_layers = hidden,
        max_steps_per_episode = max_steps_per_episode,
        invalid_action_weight = invalid_action_weight,
        cross_training=cross_training
    ),
    plots=[
        [
            {
                "label":        "running reward",
                "line_width":   1.0
            }
        ],
        [
            {   "label":    "entropy",  "line_width": 0.5   },
            {   "label":    "entropy MA",  "line_width": 1.5   },
        ],
        [
            {   "label":    "critic loss MA"   },
            {   "label":    "actor loss"   },
            #{   "label":    "invalid action loss"   }
        ]
    ]
)


monitor.start_server(port)

class SaveCallback(object):
    
    def __init__(self, save_to):
        self.BestEntropy = None
        self.EntropyMA = None
        self.NImprovements = 3
        self.SaveTo = save_to
        self.Alpha = 0.1

    def train_batch_end(self, brain, agents, batch_episodes, batch_steps, avg_losses):
        entropy = -avg_losses["entropy"]
        if self.BestEntropy is None:
            self.BestEntropy = self.EntropyMA = entropy
        else:
            self.EntropyMA += self.Alpha * (entropy-self.EntropyMA)
        #print("SaveCallback: entropy:", entropy, "  MA:", self.EntropyMA, "  BestEntropy:", self.BestEntropy, "  NI:", self.NImprovements)
        if self.EntropyMA < self.BestEntropy and self.SaveTo:
            self.BestEntropy = self.EntropyMA
            self.NImprovements -= 1
            if self.NImprovements <= 0:
                brain.save(self.SaveTo)
                print("Model saved to", self.SaveTo, "with entropy MA", self.EntropyMA)

class UpdateMonitorCallback(object):
    
    def __init__(self, monitor):
        self.Episodes = 0
        self.Monitor = monitor
        self.EntropyMA = MovingAverage()
        self.CriticMA = MovingAverage()
        self.RunningScoreMA = MovingAverage()

    def train_batch_end(self, brain, agents, batch_episodes, batch_steps, avg_losses):
        self.Episodes += batch_episodes
        for a in agents:
            rs_ma = self.RunningScoreMA(a.RunningReward)
        entropy = -avg_losses["entropy"]
        ema = self.EntropyMA(entropy)
        clossma = self.CriticMA(avg_losses["critic"])
        #print("UpdateMonitorCallback: avg losses:", [(name, value) for name, value in avg_losses.items()])
        self.Monitor.add(self.Episodes, {
                "running reward":   rs_ma,
                "critic loss MA":  clossma,
                "actor loss":  avg_losses["actor"],
                "entropy MA":  ema,
                "entropy":  entropy
                #"invalid action loss":  avg_losses["invalid_action"]
            })

class WatchCallback(object):
    
    def __init__(self):
        self.Episodes = 0
        self.TotalSteps = 0
        self.TStart = time.time()
        
    def train_batch_end(self, brain, agents, batch_episodes, batch_steps, avg_losses):
        agent = agents[0]
        self.Episodes += batch_episodes
        print("End of training batch: episodes: %7d, last episode reward: %10.3f, running_reward:%10.3f" % (self.Episodes, agent.EpisodeReward, agent.RunningReward))
        self.TotalSteps += batch_steps
        print("Traning rate:", self.TotalSteps/(time.time() - self.TStart), "steps/second", 
            "   ", self.Episodes/(time.time() - self.TStart), "episodes/second")
        
if load_from:
    [b.load(load_from) for b in brains]
    print("Model loaded from", load_from)

mon_cb = UpdateMonitorCallback(monitor)
save_cb =  SaveCallback(save_to)
watch_cb = WatchCallback()

target = 190

agent = trainer.Agents[0]

for epoch in range(1000):
    running_reward = trainer.train(target, max_episodes = 100, max_steps_per_episode = 200, episodes_per_batch=10, callbacks=[mon_cb, watch_cb, save_cb])
    print("====== testing ======")
    for episode in range(3):
        env.run(trainer.Agents, training=False, render=True)
        print("Test episode reward: %10.3f" % agent.EpisodeReward)
    if target is not None and running_reward > target:
        print("===== training is complete ======")
        break
