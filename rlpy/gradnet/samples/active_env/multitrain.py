from rlpy.gradnet.AC import Brain
from rlpy import MultiTrainer_Chain, ActiveEnvironment, MultiAgent, MultiTrainer_Independent, MultiTrainer_Sync, Callback
import numpy as np, getopt, sys
from util import Monitor, Smoothie
import time
import gradnet


opts, args = getopt.getopt(sys.argv[1:], "w:s:l:b:n:a:")
opts = dict(opts)
load_from = opts.get("-l") or opts.get("-w")
save_to = opts.get("-s") or opts.get("-w")
brain_mode = opts.get("-b", "chain")        # or chain or sync
nagents = int(opts.get("-n", 1))
env_name = args[0]

alpha = None
if "-a" in opts:
    alpha = float(opts["-a"])

np.set_printoptions(precision=4, suppress=True)

cutoff = 10
beta = 0.5
gamma = 0.99
comment = ""
learning_rate = 0.01
max_steps_per_episode = 300
port = 8989
hidden = 300

entropy_weight = 0.01
critic_weight = 0.5
invalid_action_weight = 10.0
cross_training = 0.0

if env_name == "duel":
    from tank_duel_env import TankDuelEnv
    duel = True
    hit_target = True
    compete = True
    brain_mode = "chain"
    env = TankDuelEnv(duel=duel, target=hit_target, compete=compete)
    nagents = 2
elif env_name == "tanks_single":
    from tank_target_env import TankTargetEnv
    genv = TankTargetEnv()
    env = ActiveEnvironment.from_gym_env(genv, max_steps_per_episode)
    nagents = 1
elif env_name == "ttt":
    from ttt_env import TicTacToeEnv
    env = TicTacToeEnv()
    nagents = 2
    gamma = 1.0
    cutoff = 100
    critic_weight = 1.0
    alpha = 0.5
    brain_mode="share"
else:
    import gym
    env = ActiveEnvironment.from_gym_env(gym.make(env_name))
    brain_mode="share"
    nagents = 1

optimizer = gradnet.optimizers.get_optimizer("adagrad", learning_rate=learning_rate) 

if brain_mode == "share":
    brain = Brain(env.observation_space, env.action_space, gamma=gamma, cutoff=cutoff, learning_rate=learning_rate, entropy_weight=entropy_weight,
        optimizer=optimizer, hidden=hidden,
        critic_weight=critic_weight, invalid_action_weight=invalid_action_weight)
    agents = [MultiAgent(brain) for _ in range(nagents)]
    trainer = MultiTrainer_Independent(env, agents)
elif brain_mode in ("sync", "chain"):
    brains = [Brain(env.observation_space, env.action_space, gamma=gamma, cutoff=cutoff, learning_rate=learning_rate, entropy_weight=entropy_weight,
        optimizer=optimizer, hidden=hidden,
        critic_weight=critic_weight, invalid_action_weight=invalid_action_weight) for _ in range(nagents)
    ]
    b0 = brains[0]
    for brain in brains:
        if not brain is b0:
            brain.set_weights(b0.get_weights())
    agents = [MultiAgent(brain) for brain in brains]
    if brain_mode == "chain":
        trainer = MultiTrainer_Chain(env, agents, alpha=alpha)
    else:
        trainer = MultiTrainer_Sync(env, agents, alpha=alpha)
        
    
plots=[
    [
        {
            "label":        f"running reward {i}",
            "line_width":   1.0
        } for i in range(nagents)
    ],
    [
        {   "label":    "entropy moving average",  "line_width": 1.5   }
    ],
    [
        {   "label":    "critic loss MA"   },
        {   "label":    "actor loss"   },
        #{   "label":    "invalid action loss"   }
    ]
]


monitor = Monitor("monitor.csv", 
    title = "Multitank",
    metadata = dict(
        gamma=gamma,
        trainer=trainer.__class__.__name__,
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
    ), plots=plots
)


monitor.start_server(port)

class SaveCallback(Callback):
    
    def __init__(self, save_to):
        Callback.__init__(self)
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

class UpdateMonitorCallback(Callback):
    
    def __init__(self, monitor):
        Callback.__init__(self, fire_interval=10)
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
        _, clossma, _ = self.CriticSmoothie(avg_losses["critic_loss"])
        #print("UpdateMonitorCallback: avg losses:", [(name, value) for name, value in avg_losses.items()])
        data = {
                "critic loss MA":  clossma,
                "actor loss":  avg_losses["actor_loss"],
                "entropy moving average":  -ema,
                #"invalid action loss":  avg_losses["invalid_action"]
            }
        for i, agent in enumerate(agents):
            data[f"running reward {i}"] = agent.RunningReward
        
        self.Monitor.add(self.Episodes, data)

class WeightSyncCallback(Callback):
    
    def agent_synced(self, agent, alpha):
        print("agent %d synced with alpha=%.3f" % (id(agent)%101, alpha))

if load_from:
    [b.load(load_from) for b in brains]
    print("Model loaded from", load_from)

mon_cb = UpdateMonitorCallback(monitor)
save_cb =  SaveCallback(save_to)

for epoch in range(1000):
    trainer.train(max_episodes = 200, episodes_per_batch=5, callbacks=[mon_cb, save_cb, WeightSyncCallback()])
    print(f"====== end of training run. total episodes: {trainer.Episodes} ======")
    print("====== testing ======")
    scores = np.zeros((nagents,))
    for episode in range(5):
        env.run(trainer.Agents, training=False, render=True)
        winner = np.argmax([a.EpisodeReward for a in trainer.Agents])
        loser = np.argmin([a.EpisodeReward for a in trainer.Agents])
        if trainer.Agents[winner].EpisodeReward > trainer.Agents[loser].EpisodeReward:
            scores[winner] += 1
    print("scores", scores)

        
