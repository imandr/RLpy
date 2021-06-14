import numpy as np
import getopt, sys, random
from envs import make_env

np.set_printoptions(precision=3)

opts, args = getopt.getopt(sys.argv[1:], "vn:l:")
opts = dict(opts)
n_agents = int(opts.get("-n", 2))
do_render = "-v" in opts
load_from = opts.get("-l")

env = make_env(args[0], n_agents)
actions_shape = env.ActionShape
num_actions = env.NumActions

while True:
    print("--------------------------------------------")
    obs = env.reset()
    if do_render:
        env.render()
        print (obs)
    done = [False]*n_agents
    score = 0.0
    while not all(done):
        actions = [random.randint(0, num_actions-1) for _ in range(n_agents)]
        obs, reward, done, info = env.step(actions)
        if do_render:
            env.render()
            print (actions, obs, reward, done)
        score += reward
    print("score:", score)
