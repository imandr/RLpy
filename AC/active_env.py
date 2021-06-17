class ActiveEnvAgent(object):
    #
    # Abstract class defining the Active Environment Agent interface
    #
    
    def __init__(self):
        pass
        
    def reset(self, training):
        # reset the agent state and get ready for the episode in training or test mode
        pass
        
    def update(self, observation):
        # the environment may notify the agent about its changes by calling this method zero or more times between agent's actions
        pass

    def reward(self, reward):
        # this function will also be called when the agent receives reward, which may be 0 or more times between the agent's actions
        # the reward is the reward collected since last "reward" or "action" call
        pass
        
    def action(self, observation, available_actions):
        # the environment asks the agent for the next action for the observation
        raise NotImplementedError()
        
    def done(self, observation=None):
        # the environment signals the end of the episode. Depending on the environment, it may also provide the last observation,
        # corresponding to the end state for the agent
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
        return ActiveGymEnvironment(name_or_env, time_limit)

    def run(self, agents, callbacks=[], training=True, render=False, max_turns = None):
        if max_turns is None:   max_turns = self.MaxTurns

        if not isinstance(callbacks, list):
            callbacks = [callbacks]
        
        self.reset(agents)
        all_done = False
        T = self.MaxTurns
        while not all_done and (T is None or T > 0):
            all_done = self.turn(training)
            if render:  self.render()
            for cb in callbacks:
                if hasattr(cb, "end_turn"):
                    cb.end_turn(self, agents, {})
            if T is not None:
                T -= 1
                if not all_done and T <= 0:
                    for a in agents:
                        a.done(None)

        for cb in callbacks:
            if hasattr(cb, "end_episode"):
                cb.end_episode(self, agents, {})
                
    def reset(self, agents):        # overridable
        pass
        
    def turn(self, training):       # overridable
        return False    # true if all agents are done
                
class ActiveGymEnvironment(ActiveEnvironment):
    
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

        self.State = None
        self.Agent = None
        
    def reset(self, agents):
        assert len(agents) == 1, "Gym environments work with single action only"
        self.Agent = agents[0]
        self.State = self.GEnv.reset()
        self.Agent.reset()
        
    def turn(self, training):
        action = self.Agent.action(self.State, None, training=training)
        self.State, reward, done, meta = self.GEnv.step(action)
        self.Agent.reward(reward)
        if done:
            self.Agent.done(self.State)
        return done
        
    def render(self):
        self.GEnv.render()
    
        