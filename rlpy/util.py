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

class CallbackList(object):
    
    #
    # Sends event data to list of Callback objects and to list of callback functions
    #
    
    def __init__(self, *callbacks):
        self.Callbacks = list(callbacks)
        
    def add(self, *callbacks):
        self.Callbacks += list(callbacks)
        
    def __call__(self, event, *params, **args):
        for cb in self.Callbacks:
            cb(event, *params, **args)
            
    @staticmethod
    def convert(arg):
        if isinstance(arg, CallbackList):
            return arg
        elif isinstance(arg, (list, tuple)):
            return CallbackList(*arg)
        elif arg is None:
            return CallbackList()       # empty
        else:
            raise ValueError("Can not convert %s to CallbackList" % (arg,))


