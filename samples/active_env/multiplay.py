from gradnet import ModelClient

from rlpy import MultiAgent, BrainDiscrete
import numpy as np, getopt, sys
from util import Monitor, Smoothie
import time, os.path
import gradnet
from gradnet import ModelClient

np.set_printoptions(precision=4, suppress=True)

Usage = """
python multitrain.py <env> <model server URL>
"""

opts, args = getopt.getopt(sys.argv[1:], "vr")
opts = dict(opts)

do_render = "-r" in opts
verbose = "-v" in opts
env_name = args[0]
model_server_url = args[1]
model_client = ModelClient(env_name, model_server_url)

max_steps_per_episode = 300
if env_name == "duel":
    from tank_duel_env import TankDuelEnv
    brain_mode = "chain"
    env = TankDuelEnv(duel=False)
    nagents = 2
    alpha = 0.2
    hidden = 500
else:
    import gym
    env = ActiveEnvironment.from_gym_env(gym.make(env_name))
    brain_mode="share"
    nagents = 1

brain = BrainDiscrete(env.observation_space.shape, env.action_space.n, hidden=hidden)
agents = [MultiAgent(brain, id=i) for i in range(nagents)]

while True:
    weights = None
    while weights is None:
        time.sleep(1.0)
        weights = model_client.get_weights()
    brain.set_weights(weights)
    env.run(agents, training=False, render=do_render)
