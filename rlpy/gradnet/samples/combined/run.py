from testenv import TestEnv
from rlpy import Agent
from ac_combined import Brain

env = TestEnv()
brain = Brain(env.ObservationSpec, env.ActionSpec)
agent = Agent(brain)

reward, history = agent.play_episode(env, max_steps=10)

print("--- episode reward=", reward, "  history: ---")
for k, lst in history.items():
    if isinstance(lst, list):
        print(k,":")
        for x in lst:
            print("   ", x)
    else:
        print(k,":", lst)
        
brain.train_on_multi_episode_history([history])