from rlpy import MultiAgent, Callback, BrainDiscrete
import numpy as np, getopt, sys
from util import Monitor, Smoothie
import time
import gradnet


opts, args = getopt.getopt(sys.argv[1:], "l:b:n:a:q")
opts = dict(opts)
nagents = int(opts.get("-n", 1))
env_name = args[0]

alpha = None
if "-a" in opts:
    alpha = float(opts["-a"])

do_render = "-q" not in opts

load_from = opts.get("-l")

np.set_printoptions(precision=4, suppress=True)

cutoff = None
beta = 1.0
gamma = 0.99
comment = ""
learning_rate = 0.01
max_steps_per_episode = 300
port = 8989
hidden = 300

entropy_weight = 0.01
critic_weight = 1.0
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
    max_steps_per_episode = 200
    beta = 0.9
    hidden = 400
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

brain = BrainDiscrete(env.observation_space.shape, env.action_space.n, gamma=gamma, 
    learning_rate=learning_rate, entropy_weight=entropy_weight,        
    cutoff=cutoff, beta=beta,
    hidden=hidden,
    critic_weight=critic_weight, invalid_action_weight=invalid_action_weight)

if load_from:
    brain.load_weights(load_from)


agents = [MultiAgent(brain) for _ in range(nagents)]


while True:
    env.run(agents, training=False, render=do_render)

        
