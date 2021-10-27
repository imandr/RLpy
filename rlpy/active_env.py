from .util import CallbackList

class ActiveEnvAgent(object):
    #
    # Abstract class defining the Active Environment Agent interface
    #
    
    def __init__(self):
        pass
        
    def reset(self, training):
        # reset the agent state and get ready for the episode in training or test mode
        pass
        
    def update(self, observation=None, reward=0.0):
        # the environment may notify the agent about its changes by calling this method zero or more times between agent's actions
        # the arguments, both optional, can be used to pass new obsservation and/or an incremental reward received by the agent
        pass

    def action(self, observation, valid_actions=None):
        # the environment asks the agent for the next action for the observation
        raise NotImplementedError()
        
    def done(self, reward=0.0, observation=None):
        # the environment signals the end of the episode. Depending on the environment, it may also provide the last observation,
        # corresponding to the end state for the agent. Optional reward is the last reward received by the agent in the episode.
        pass
        

class ActiveEnvironment(object):

    def __init__(self, name=None, max_turns=None, action_space=None, observation_space=None):
        self.MaxTurns = max_turns
        self.Name = name
        self.action_space = action_space
        self.observation_space = observation_space
        
    def __str__(self):
        return self.Name or self.__class__.__name__
        
    @staticmethod
    def from_gym_env(name_or_env, time_limit=None):
        return ActiveFromGymEnvironment(name_or_env, time_limit)

    def run(self, agents, callbacks=None, training=True, render=False, max_turns = None):
        
        callbacks = CallbackList.convert(callbacks)
        
        if max_turns is None:   max_turns = self.MaxTurns

        self.reset(agents, training)
        callbacks("active_env_begin_episode", self, agents)
        all_done = False
        T = self.MaxTurns
        while not all_done and (T is None or T > 0):
            all_done = self.turn()
            #print("ActiveEnv.run(): all_done=", all_done)
            if render:  self.render()
            callbacks("active_env_end_turn", self, agents)
            if T is not None:
                T -= 1
                if not all_done and T <= 0:
                    for a in agents:
                        a.done(None)

        callbacks("active_env_end_episode", self, agents)
                
    def reset(self, agents, training):        # overridable
        pass
        
    def turn(self):       # overridable
        return False    # true if all agents are done
                
class ActiveFromGymEnvironment(ActiveEnvironment):
    
    def __init__(self, name_or_env, time_limit=None):
        if isinstance(name_or_env, str): 
            import gym
            gym_env = gym.make(name_or_env)
            name = name_or_env
        else:
            gym_env = name_or_env
            name = str(gym_env)
        self.MaxSteps = time_limit
        self.GEnv = gym_env
        ActiveEnvironment.__init__(self, name=name, max_turns=time_limit, 
            action_space=gym_env.action_space, observation_space = gym_env.observation_space)
        
        self.Observation = None
        self.Agent = None
        
    def reset(self, agents, training):
        assert len(agents) == 1, "Gym environments work with single action only"
        self.Training = training
        self.Agent = agents[0]
        self.Observation = self.GEnv.reset()
        self.Agent.reset(training)
        
    def turn(self):
        action = self.Agent.action(self.Observation)
        self.Observation, reward, done, meta = self.GEnv.step(action)
        if done:
            self.Agent.done(self.Observation, reward)
        else:
            self.Agent.update(reward=reward)
        return done
        
    def render(self):
        self.GEnv.render()
    
        