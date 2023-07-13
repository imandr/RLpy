from .util import CallbackList

class ActiveEnvAgent(object):
    #
    # Abstract class defining the Active Environment Agent interface
    #
    
    def __init__(self):
        pass
        
    def reset(self, training):
        # reset the agent state for new episode
        pass
        
    def action(self, observation, valid_actions=None):
        # ends previous turn and begins new one. Rewards accumulated since the previous action() will be associated
        # with the previous action
        # the environment asks the agent for the next action for the observation
        # called once per turn
        # returns the action to take for the turn
        raise NotImplementedError()
        
    def update(self, observation=None, reward=0.0):
        # the environment may notify the agent about its changes by calling this method zero or more times between agent's actions
        # the arguments, both optional, can be used to pass new observation and/or an incremental reward received by the agent
        # intermediate observations received between action and end_turn() can be used by RRN-based agents
        # rewards will be accumulated and associated with the action at end_turn()
        pass

    def end_turn(self):
        # Rewards accumulated since the previous action() will be associated with the previous action
        # Reward accumulation will be reset
        pass
        
    def end_episode(self):
        # the environment signals the end of the episode
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
        for a in agents:
            a.reset(training)
        callbacks("active_env_begin_episode", self, agents)
        end_episode = False
        T = self.MaxTurns
        while not end_episode and (T is None or T > 0):
            end_episode = self.turn()
            for a in agents:
                a.end_turn()
            #print("ActiveEnv.run(): all_done=", all_done)
            if render:  self.render()
            callbacks("active_env_end_turn", self, agents)

        # make sure done() is sent
        for a in agents:
            a.end_episode()

        callbacks("active_env_end_episode", self, agents, training)
                
    def reset(self, agents, training):        # overridable
        pass
        
    def turn(self):       # overridable
        return False      # true if this is the end of the episode
                
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
