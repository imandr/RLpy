import numpy as np
import getopt, sys, random
from envs import make_env

np.set_printoptions(precision=3)

opts, args = getopt.getopt(sys.argv[1:], "vn:l:")
opts = dict(opts)
num_tests = int(opts.get("-n", 100))
do_render = "-v" in opts
load_from = opts.get("-l")

env = make_env(args[0])
num_actions = env.action_space.n

for t in range(num_tests):
    print("--------------------------------------------")
    obs = env.reset()
    if do_render:
        env.render()
        print (obs)
    done = False
    score = 0.0
    while not done:
        action = random.randint(0, num_actions-1)
        obs, reward, done, info = env.step(action)
        if do_render:
            env.render()
            print (obs, reward, done)
        score += reward
    print("score:", score)
