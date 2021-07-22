from tank_duel_env import TankDuelEnv
from AC import Brain, MultiTrainer_Chain
import numpy as np, getopt, sys
from util import Monitor, Smoothie
from tensorflow import keras
import time


opts, args = getopt.getopt(sys.argv[1:], "w:s:l:")
opts = dict(opts)
load_from = opts.get("-l") or opts.get("-w")
save_to = opts.get("-s") or opts.get("-w")

np.set_printoptions(precision=4, suppress=True)

duel = True
hit_target = True
env = TankDuelEnv(duel=duel, target=hit_target)

cutoff = None
beta = 0.5
gamma = 0.99
comment = ""
learning_rate = 0.01
max_steps_per_episode = 300
port = 8989
hidden = 200

entropy_weight = 0.01
critic_weight = 0.5
invalid_action_weight = 10.0
cross_training = 0.2

optimizer = keras.optimizers.Adagrad(learning_rate=learning_rate)

brains = [Brain(env.observation_space.shape, env.action_space.n, gamma=gamma, cutoff=cutoff, learning_rate=learning_rate, entropy_weight=entropy_weight,
    optimizer=optimizer, hidden=hidden,
    critic_weight=critic_weight, invalid_action_weight=invalid_action_weight) for _ in (0,1)
]
    
trainer = MultiTrainer_Chain(env, brains, alpha=cross_training)

monitor = Monitor("monitor.csv", 
    title = "Multitank",
    metadata = dict(
        gamma=gamma,
        trainer=trainer.__class__.__name__,
        hit_target = hit_target,
        duel = duel,
        comment = comment,
        environment = str(env),
        learning_rate = learning_rate,
        brain = "AC",
        cutoff = cutoff,
        beta = beta, 
        steps_per_episode = max_steps_per_episode,
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
                "label":        "running reward 0",
                "line_width":   1.0
            },            
            {
                "label":        "running reward 1",
                "line_width":   1.0
            }
        ],
        [
            {   "label":    "entropy low",  "line_width": 1.0   },
            {   "label":    "entropy MA",  "line_width": 1.5   },
            {   "label":    "entropy high",  "line_width": 1.0   }
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

    def train_batch_end(self, brain, agents, batch_episodes, total_steps, avg_losses):
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
        self.EntropySmoothie = Smoothie(0.01)
        self.CriticSmoothie = Smoothie(0.01)
        self.RunningScoreSmoothie = Smoothie(0.01)

    def train_batch_end(self, brain, agents, batch_episodes, total_steps, avg_losses):
        self.Episodes += batch_episodes
        for a in agents:
            rs_low, rs_ma, rs_high = self.RunningScoreSmoothie(a.RunningReward)
        entropy = -avg_losses["entropy"]
        elow, ema, ehigh = self.EntropySmoothie(entropy)
        _, clossma, _ = self.CriticSmoothie(avg_losses["critic"])
        #print("UpdateMonitorCallback: avg losses:", [(name, value) for name, value in avg_losses.items()])
        self.Monitor.add(self.Episodes, {
                "running reward 0":   agents[0].RunningReward,
                "running reward 1":   agents[1].RunningReward,
                "critic loss MA":  clossma,
                "actor loss":  avg_losses["actor"],
                "entropy low":  elow,
                "entropy high":  ehigh,
                "entropy MA":  ema,
                #"invalid action loss":  avg_losses["invalid_action"]
            })

class WatchCallback(object):

    def __init__(self):
        self.TotalSteps = 0
        self.TotalEpisodes = 0
        self.TStart = time.time()

    def train_batch_end(self, brain, agents, batch_episodes, batch_steps, avg_losses):
        self.TotalSteps += batch_steps
        self.TotalEpisodes += batch_episodes
        print("Traning rate:", self.TotalSteps/(time.time() - self.TStart))


if load_from:
    [b.load(load_from) for b in brains]
    print("Model loaded from", load_from)

mon_cb = UpdateMonitorCallback(monitor)
save_cb =  SaveCallback(save_to)

for epoch in range(1000):
    trainer.train(max_episodes = 200, episodes_per_batch=5, callbacks=[mon_cb, save_cb])
    print(f"====== end of training run. total episodes: {trainer.Episodes} ======")
    print("====== testing ======")
    scores = np.zeros((2,))
    for episode in range(5):
        env.run(trainer.Agents, training=False, render=True)
        winner = np.argmax([a.EpisodeReward for a in trainer.Agents])
        if trainer.Agents[1-winner].EpisodeReward != trainer.Agents[winner].EpisodeReward:
            scores[winner] += 1
    print("score: %d:%d" % tuple(scores))
        
