class Callback(object):
    
    def __init__(self, fire_rate=1.0, fire_interval=None):
        self.FireRate = fire_rate
        self.NextFire = self.FireInterval = fire_interval
        self.FireLevel = 0.0
        self.FireCount = 0
        
    def __call__(self, event, *params, **args):
        self.FireCount += 1
        self.FireLevel += self.FireRate
        if self.FireInterval is not None and self.FireCount < self.NextFire:
            return
        if self.FireLevel < 1.0:
            return

        if hasattr(self, event):
            getattr(self, event)(*params, **args)

        self.FireLevel -= 1.0
        if self.FireInterval is not None:
            self.NextFire += self.FireInterval
            
    #
    # Optional methods:
    #

    def train_episode_end(self, agent, episode_reward, episode_history):
        pass
        
    def train_batch_end(self, agent, n_episodes, total_steps, stats):
        pass
            
    # 
    # Active env only:
    #
    def active_env_episode_end(self, agents):
        # assume each agent has agent.EpisodeReward and agent.episode_history()
        pass

class CallbackList(object):
    
    #
    # Sends event data to list of Callback objects and to list of callback functions
    #
    
    def __init__(self, *sources):
        lst = []

        for src in sources:
            if src is not None:
                if isinstance(src, CallbackList):
                    lst += src.Callbacks
                elif isinstance(src, (list, tuple)):
                    lst += list(src)
                elif isinstance(src, Callback):
                    lst.append(src)
                else:
                    raise ValueError("Can not add %s to CallbackList" % (src,))

        self.Callbacks = lst

    def add(self, *callbacks):
        self.Callbacks += list(callbacks)

    def __add__(self, other):
        return CallbackList(self, other)

    def __call__(self, event, *params, **args):
        for cb in self.Callbacks:
            cb(event, *params, **args)

    @staticmethod
    def convert(arg):
        return CallbackList(arg)

