from pythreader import Primitive, PyThread, synchronized, TaskQueue
from webpie import WPApp, WPHandler
import time, os.path, numpy as np, io, traceback, sys, json
from lib import to_str, to_bytes
from gradnet import serialize_weights, deserialize_weights

class Model(Primitive):
    
    IdleTime = 30*60
    
    def __init__(self, name, save_dir, alpha, params=None):
        Primitive.__init__(self)
        self.Name = name
        self.Params = None
        self.SaveFile = save_dir + "/" + name + "_params.npz"
        self.LastActivity = 0
        self.Alpha = alpha

    @synchronized
    def get(self):
        if self.Params is None and os.path.isfile(self.SaveFile):
            loaded = np.load(self.SaveFile)
            self.Params = [loded[k] for k in loaded]
        self.LastActivity = time.time()
        return self.Params
        
    @synchronized
    def set(self, params):
        self.LastActivity = time.time()
        self.Params = params
        self.save()

    @synchronized
    def save(self):
        if self.Params is not None:
            np.savez(self.SaveFile, *self.Params)

    @synchronized
    def update(self, params, alpha=None):
        alpha = alpha or self.Alpha
        if isinstance(params, bytes):
            params = deserialize_weights(params)
        self.LastActivity = time.time()
        old_params = self.get()
        #print("Model.get: old_params:", old_params)
        if old_params is None:
            self.Params = params
        else:
            self.Params = []
            for old, new in zip(old_params, params):
                self.Params.append(old + alpha * (new - old))
        return self.Params
    
    @synchronized
    def reset(self):
        last_params = self.Params
        self.Params = None
        try:    os.remove(self.SaveFile)
        except: pass
        return last_params
    
    @synchronized
    def offload_if_idle(self):
        if time.time() > self.LastActivity + self.IdleTime and self.Params is not None:
            np.savez(self.SaveFile, *self.Params)
            self.Params = None
            
class Handler(WPHandler):
    
    Alpha = 0.2                 # for now
    
    def model(self, request, model, alpha=None):

        if request.method == "GET":
            model = self.App.model(model, create=False)
            if model is None:
                return 404, "Not found"
            else:
                return 200, serialize_weights(model.get() or [])

        elif request.method == "DELETE":
            model = self.App.model(model)
            if model is None:
                return 404, "Not found"
            else:
                model.reset()
                return 200, "OK"

        elif request.method == "POST":
            model = self.App.model(model)
            model.set(deserialize_weights(request.body))
            return 200, serialize_weights(model.get())
            
        elif request.method == "PUT":
            if alpha is not None:
                alpha = float(alpha)
            model = self.App.model(model)
            #print("handler: PUT: body:", request.body)
            params = model.update(deserialize_weights(request.body), alpha=alpha)
            #print("handler: PUT: params:", params)
            return 200, serialize_weights(params) if params else b''
            
        else:
            return 400, "Unsupported method"
    
    def models(self, request, relpath):
        return 200, json.dumps(list(self.App.models())), "text/json"
        
    
class App(WPApp):
    
    def __init__(self, save_dir, alpha):
        WPApp.__init__(self, Handler)
        self.Alpha = alpha
        self.SaveDir = save_dir
        self.Models = {}
    
    @synchronized
    def model(self, name, create=True):
        model = self.Models.get(name)
        if model is None and create:
            model = self.Models[name] = Model(name, self.SaveDir, self.Alpha)
        return model
        
    def models(self):
        return self.Models.keys()

if __name__ == "__main__":
    import getopt
    opts, args = getopt.getopt(sys.argv[1:], "a:s:p:")
    opts = dict(opts)
    alpha = float(opts.get("-a", 0.5))
    storage = opts.get("-s", "models")
    port = int(opts.get("-p", 8888))
    App(storage, alpha).run_server(port, logging=True, log_file="-")