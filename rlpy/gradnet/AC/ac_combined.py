import random, math
import numpy as np
from gradnet import Model, Input, Loss
from gradnet.layers import Dense, LSTM, Concatenate, Flatten
from gradnet.optimizers import get_optimizer
from gradnet.activations import get_activation
from gradnet.losses import get_loss

try:
    from numba import jit
    print("numba imported")
except Exception as e:
    # no numba
    print("numba not imported:", e)
    def jit(**args):
        def decorator(f):
            def decorated(*params, **args):
                return f(*params, **args)
            return decorated
        return decorator
        
try:
    from gym import spaces
    gym_imported = True
except:
    gym_imported = False
        
@jit(nopython=True)
def _calc_retruns(gamma, rewards, value_weights, values):
    prev_ret = 0.0
    prev_value = 0.0
    prev_w = 0.0
    T = len(rewards)
    t = T-1
    returns = np.empty((T,))
    for r, w, v in list(zip(rewards, value_weights, values))[::-1]:
        ret = r + gamma*(prev_ret*(1-prev_w) + prev_value*prev_w)
        returns[t] = ret
        prev_ret = ret
        prev_value = v
        prev_w = w
        t -= 1
    return returns
    
def entropy_loss(_, probs, data):
        valid_mask=data.get("valid_actions")
        p = np.clip(probs, 1e-5, None)
        values = -np.sum(p*np.log(p), axis=-1)
        grads = np.log(p)+1.0
        if valid_mask is not None:
            grads = grads * valid_mask
        n = probs.shape[-1]
        logn = math.log(n)
        k = abs(logn-1.0)/n
        return values/logn, grads*k

def actor_loss_single_action(_, probs, values, returns, actions):
        probs_shape = probs.shape
        mb = probs_shape[0]
        na = probs_shape[-1]
        
        probs = probs.reshape((-1, probs_shape[-1]))
        values = values.reshape((-1,))
        actions = actions.reshape((-1,))
        returns = returns.reshape((-1,))

        values_shape = values.shape
        
        print("actor_loss_single_action: values shape: ", values.shape)
        print("                          returns shape:", returns.shape)
        print("                          actions shape:", actions.shape)
        
        #print("ActorLoss: probs:",probs.shape)
        #print("          values:", values.shape)
        print("actor_loss_single_action: actions:", actions)
        action_mask = np.eye(na)[actions]
        action_probs = np.sum(action_mask*probs, axis=-1)
        advantages = returns - values
        #print("ActorLoss: rewards:", rewards.shape, "   values:", values.shape, "   probs:", probs.shape, "   advantages:", advantages.shape,
        #    "   action_probs:", action_probs.shape)
        loss_grads = -advantages/np.clip(action_probs, 1.e-5, None)  
        #print("ActorLoss: shapes:", action_mask.shape, loss_grads.shape)
        grads = action_mask * loss_grads[:,None]

        grads = grads.reshape(probs_shape)
        
        losses = -advantages*np.log(np.clip(action_probs, 1.e-5, None)).reshape(values_shape)          # not really loss values
        print("actor_loss_single_action: losses:", losses)
        return losses, [grads, None]


def combined_actor_loss(_, values, *params):
    values = values[:,0]
    print("combined_actor_loss: values.shape:", values.shape)
    probs_inputs, data = params[:-1], params[-1]
    losses = np.zeros(values.shape)
    all_grads = [None]     # None for grads dL/dvalues
    actions = data["actions"]       
    returns = data["returns"]
    for i, probs in enumerate(probs_inputs):
        l, (grads, _) = actor_loss_single_action(_, probs, values, returns, actions[:,i])
        losses += l
        all_grads.append(grads)
    return losses, all_grads
    

def invalid_actions_loss(_, probs, data):
    valid_mask=data.get("valid_actions")
    if valid_mask is not None:
        losses = np.mean(probs*probs*(1-valid_mask), axis=-1)
        grads = 2*(1-valid_mask)*probs
    else:
        losses = np.zeros((len(probs),))
        grads = None
    #print("invalid_actions_loss: losses:", losses, "   grads:", grads)
    return losses, grads
    
    
def critic_loss(_, values, data):
    returns = data["returns"]
    d = values - returns[:,None]
    losses = d*d
    grads = 2*d
    return losses, grads
    
#
# continuous actions losses
#
    
def cont_entropy_loss(_, sigmas, data):
        # means, sigmas: [mb, ncontrols]
        sigmas = np.clip(sigmas, 1e-3, None)
        ncontrols = sigmas.shape[-1]
        sprod = np.prod(sigmas, axis=-1)**(2/ncontrols)
        entropy = ncontrols/2 * np.log(2*math.pi*math.e*sprod)
        grads = 1/sigmas
        return entropy, [grads]

def cont_actor_loss(_, means, sigmas, values, data):        
        sigmas = np.clip(sigmas, 1e-3, None)
        controls = data["controls"]
        logprobs = -math.log(2*math.pi)/2 - np.log(sigmas) - ((controls-means)/sigmas)**2/2
        returns = data["returns"][:,None]
        #print("actor_loss: logprobs:", logprobs.shape)
        #print("            controls:", controls.shape)
        #print("            returns:", data["returns"].shape)
        advantages = returns - values
        #print("            advantages:", advantages.shape)
        losses = -advantages * logprobs
        grads_sigmas = -advantages * (((controls-means)/sigmas)**2 - 1)/sigmas
        grads_means = -advantages * (controls-means)/sigmas**2
        #print("            grads means:", grads_means.shape)
        return losses, [grads_means, grads_sigmas, None]
    
class Brain(object):
    #
    # Non-recurrent Brain model:
    #   inputs: 
    #       state:                  [mb, ...]
    #
    #   outputs:
    #       value:                  [mb, 1]
    #       probs:                  [mb, nactions] probabilities, dummy but present if NActions == 0
    #       means                   [mb, ncontrols] control means, dummy but present if NControls == 0
    #       sigmas                  [mb, ncontrols] control sigmas, dummy but present if NControls == 0
    # 
    def __init__(self, observation_space, action_space, model=None, recurrent=False,
            optimizer=None, hidden=128, gamma=0.99, 
            with_rnn = False,
            learning_rate = 0.001,       # learning rate for the common part and the Actor
            entropy_weight = 0.01,
            invalid_action_weight = 0.5,
            critic_weight = 1.0, actor_weight=1.0,
            cutoff = 1, beta = None     # if cutoff is not None, beta is ignored
                                        # if cutoff is None and beta is None, entropy is used
                                        # otherwise beta is used
        ):
        
        #
        # observation space:
        #   int
        #   Box             
        #   tuple of Box'es   
        #
        
        self.ObservationShapes = None

        if isinstance(observation_space, int):
            self.ObservationShapes = [(observation_space,)]
        elif isinstance(observation_space, spaces.Box):
            self.ObservationShapes = [observation_space.shape]
        elif isinstance(observation_space, spaces.Tuple):
            if all(isinstance(x, spaces.Box) for x in observation_space):
                self.ObservationShapes = [(x.shape,) for x in observation_space]
        if self.ObservationShapes is None:
            raise ValueError("Unsupported observation space specification: %s" % (observation_spec,))
            
        #
        # action space:
        #   int or Discrete        - single discrete action
        #   [ints] or MultiDiscrete   - discrete action dimensions
        #   Box             - controls
        #   Tuple(int, Discrete, MultiDiscrete or[ints], Box or int)
        #   tuple(int, Discrete, MultiDiscrete or[ints], Box or int)
        #   
        
        discrete = cont = self.DiscreteDims = None
        self.NControls = 0        
        self.ControlBounds = None
        
        if isinstance(action_space, (spaces.Discrete, spaces.MultiDiscrete)):
            discrete = action_space
        elif isinstance(action_space, spaces.Box):
            cont = action_space
        elif isinstance(action_space, (spaces.Tuple, tuple)):
            discrete, cont = action_space
        else:
            raise ValueError("Unsupported vaue type for action specification: %s" % (action_spec,))
            
        if discrete is not None:
            if isinstance(discrete, int):
                self.DiscreteDims = [discrete]
            elif isinstance(discrete, list):
                self.DiscreteDims = discrete
            elif isinstance(discrete, spaces.Discrete):
                self.DiscreteDims = [discrete.n]
            elif isinstance(discrete, spaces.MultiDiscrete):
                self.DiscreteDims = list(discrete.nvec)
            else:
                raise ValueError("Unrecognized discrete actions space:", discrete)
                
        if cont is not None:
            if isinstance(cont, spaces.Box):
                assert len(cont.shape) == 1
                self.NControls = cont.shape[0]
                self.ControlBounds = cont.low, cont.high
            elif isinstance(cont, int):
                self.NControls = cont
            else:
                raise ValueError("Unrecognized continuous actions space:", cont)
        
        if not self.DiscreteDims and not self.NControls:
            raise ValueError("Unsupported vaue type for action specification: %s" % (action_spec,))
            
        print("Creating brain for:")
        print("    observation shapes:", self.ObservationShapes)
        print("    discrete dims:     ", self.DiscreteDims)
        print("    controls:          ", self.NControls)
        
        self.Recurrent = recurrent
            
        self.Optimizer = optimizer or get_optimizer("adagrad", learning_rate=learning_rate) 
                #optimizer("SGD", learning_rate=learning_rate, momentum=0.5)
        self.Cutoff = cutoff
        self.Beta = beta
        self.Gamma = gamma
        self.FutureValueCutoff = cutoff
        
        self.EntropyWeight = entropy_weight
        self.InvalidActionWeight = invalid_action_weight
        self.CriticWeight = critic_weight
        
        if model is None:   
            model = self.default_model(self.ObservationShapes, self.DiscreteDims, self.NControls, hidden) if not with_rnn \
                    else self.default_rnn_model(self.ObservationShapes, self.NActions, hidden)
                    
        model = self.add_losses(model, critic_weight=critic_weight, actor_weight=actor_weight, entropy_weight=entropy_weight, 
                    invalid_action_weight=invalid_action_weight)
        model.compile(optimizer=self.Optimizer)
            
        if True:
            print("model losses and weights:")
            for name, (loss, weight) in model.Losses.items():
                print(f"{name}: {loss} * {weight}")

        self.Model = model
        
    def weights_as_dict(self):
        out = {}
        for l in self.Model.layers:
            w = l.get_weights()
            if w is not None:
                if isinstance(w, list):
                    for i, wi in enumerate(w):
                        out[l.Name+f"_{i}"] = wi
                else:
                    out[l.Name] = w
        return out

    def copy_weights_from_keras_brain(self, kbrain):
        kmodel = kbrain.Model
        weights = {}
        for l in kmodel.layers:
            name = l.name
            weights[name] = l.get_weights()

        for l in self.Model.layers:
            name = l.Name
            if name is not None:
                w = weights.get(name)
                if w is None:
                    print(f"warning: the source model has no weights for layer {name}")
                else:
                    l.set_weights(w)
                    
    def get_weights(self):
        return self.Model.get_weights()
        
    def set_weights(self, weights):
        self.Model.set_weights(weights)
        
    def update_weights(self, source, alpha):
        # source can be either: Brain, Model, list of weights
        if isinstance(source, Brain):
            source = source.Model
        old = self.Model.get_weights()
        self.Model.update_weights(source, alpha)
        return old
                    
    def default_model(self, input_shapes, discrete_dims, num_controls, hidden):

        inputs = [Input(input_shape, name=f"input_{i}") for i, input_shape in enumerate(input_shapes)]

        if len(inputs) == 1:
            inp = inputs[0]
        else:
            flattened = [Flatten()(inp) if len(inp.Shape)>1 else inp for inp in inputs]
            inp = Concatenate()(*flattened)

        common1 = Dense(hidden, activation="relu", name="common1")(inp)
        common = Dense(hidden//2, activation="relu", name="common")(common1)

        critic1 = Dense(hidden//5, name="critic1", activation="relu")(common)
        value = Dense(1, name="critic")(critic1)
        
        prob_layers = []
        means = sigmas = None
        
        out_layers = [value]
        
        for i, num_actions in enumerate(discrete_dims or []):
            a = Dense(max(hidden//5, num_actions*5), activation="relu", name=f"action_hidden_{i}")(common)
            probs = Dense(num_actions, name=f"action_{i}", activation="softmax")(a)
            prob_layers.append(probs)
            out_layers.append(probs)

        if num_controls > 0:
            a = Dense(max(hidden//5, num_controls*10), activation="relu", name="controls_hidden")(common)
            means = Dense(num_controls, activation="linear", name="means")(common)
            sigmas = Dense(num_controls, activation="softplus", name="sigmas")(common)
            out_layers.append(means)
            out_layers.append(sigmas)
            
        model = Model(inputs, out_layers)
        
        model["value"] = value
        model["prob_layers"] = prob_layers
        model["means"] = means
        model["sigmas"] = sigmas
        
        return model

    def add_losses(self, model, critic_weight=1.0, actor_weight=1.0, entropy_weight=0.01, invalid_action_weight=10.0, **unused):

        model.add_loss(Loss(critic_loss, model["value"]),                   critic_weight, name="critic_loss")
        prob_layers = model["prob_layers"]
        if prob_layers:
            model   \
            .add_loss(Loss(combined_actor_loss, model["value"], *prob_layers),    actor_weight, name=f"actor_loss")
            #.add_loss(Loss(combined_entropy_loss, *prob_layers),                  entropy_weight, name=f"entropy_loss") \
            #.add_loss(Loss(invalid_actions_loss, *prob_layers),          invalid_action_weight, name=f"invalid_action_loss")    \
        
        if self.NControls:
            model   \
            .add_loss(Loss(cont_actor_loss,   model["means"], model["sigmas"], model["value"]),    actor_weight, name="cont_actor_loss")     \
            #.add_loss(Loss(cont_entropy_loss, model["sigmas"]),                  entropy_weight, name="cont_entropy_loss")

        return model
        
    def reset_episode(self):    # overridable
        self.Model.reset_state()
        
    def save(self, filename):
        self.Model.save_weights(filename)
        
    def load(self, filename):
        self.Model.load_weights(filename)
        
    def serialize_weights(self):
        weights = self.Model.get_weights()
        header=["%d" % (len(weights),)]
        for w in weights:
            shape_text = ",".join(str(w.shape))
            line = "%s/%s" % (w.dtype.str, shape_text)
            header.append(line)
        header = ":".join(header)
        out = [header.encode("utf-8")]
        for w in weights:
            out.append(w.data)
        return b''.join(out)
            
    @staticmethod
    def deserialize_weights(buf):
        def read_until(buf, i, end):
            c = b''
            out = b''
            while i < len(buf) and c != end:
                c = buf[i]
                i += 1
                if c != end:
                    out += c
            if c != end:
                raise ValueError("end of buffer reached")
            return i, out
        i = 0
        i, nweights = read_until(buf, i, b':')
        nweights = int(nweights)
        weight_descs = []
        for j in range(nweights):
            i, desc = read_until(buf, i, b':')
            words = desc.split(b'/', 1)
            assert len(words) == 2
            dt = np.dtype(words[0])
            shape = tuple(int(n) for n in words[1].split(b','))
            weight_descs.append(dt, shape)
            
        weights = []
        for dt, shape in weight_descs:
            n = math.prod(shape)
            w = np.frombuffer(buf, dt, n, offset=i).reshape(shape)
            i += len(w.buffer)
            weights.append(w)
        return weights
        
    def set_weights_from_serialized(self, buf):
        self.Model.set_weights(Brain.deserialize_weights(buf))
                
    def calculate_future_returns_with_cutoff(self, rewards, probs, values):
        #
        # calculates furure rewards for each step of the (single) episode history:
        #
        #  future_rewards(t) = sum[i=[0...cutoff)](r[t+i]*gamma**i) + v[t+cutoff]
        #  if cutoff == 1: future_rewards[t] = r[t] + v[t+1]
        #
        # returns: rets[t] - np.array
        #
        
        cutoff = self.Cutoff
        gamma_powers = self.Gamma**np.arange(cutoff)

        out = []
        T = len(rewards)
        #print("T=", T)
        rets = np.empty((T,), dtype=np.float32)
        for t in range(T):
            ret = 0.0
            jmax = T-t if cutoff is None else min(T-t, cutoff)
            ret = np.sum(gamma_powers[:jmax]*rewards[t:t+jmax])
            #print("t=", t, "   T=", T, "   cutoff=", cutoff,"   jmax=", jmax, "   discounted:", ret)
            if t+cutoff < T:
                #print("     +V:", vals[t+cutoff])
                ret += values[t+cutoff]*self.Gamma**cutoff
            #print("t:", t, "   ret:", ret)
            rets[t] = ret
        if False:
            print("calculate_future_returns_with_cutoff: gamma=", self.Gamma, "  cutoff=", cutoff)
            print("  rewards:", rewards)
            print("  powers: ", gamma_powers)
            print("  returns:", rets)
        return rets
        
    def calculate_future_returns(self, rewards, probs, values):
        #
        # calculates furure rewards for each step of the (single) episode history:
        #
        # returns: rets[i,t] - list of np.arrays
        #
        
        if self.Cutoff is not None:
            return self.calculate_future_returns_with_cutoff(rewards, probs, values)
        
        if self.Beta is None:
            value_weights = -np.sum(probs*np.log(np.clip(probs, 1e-5, None)), axis=-1)/math.log(self.NActions) # entropy for each step normalized to 1
        else:
            value_weights = np.ones((len(rewards),)) * self.Beta
        return _calc_retruns(self.Gamma, rewards, value_weights, values)

        
    def valid_probs(self, probs, valid_masks):
        if valid_masks is not None and valid_masks[0] is not None:
            probs = valid_masks * probs
            probs = probs/np.sum(probs, axis=-1, keepdims=True)
        return probs
        
    def add_losses_from_episode(self, h):
        #
        # actions is list of:
        #   ints            - single action environments
        #   [int,...]       - multiple discrete actions environments
        #   ([ints], [float])   - multi-action, multi-control environments
        #
        # observations is list of lists:
        #   [ndarray, ...]
        #

        rewards = h["rewards"]
        observations = h["observations"]
        actions = h["actions"]
        valids = h["valid_actions"]
        T = len(actions)

        #print("Brain.add_losses_from_episode: steps:", T)

        # transpose observations history
        xcolumns = [np.array(column) for column in zip(*observations)]

        #print("episode observations shape:", observations.shape)
        #print("add_losses: reset_state()")
        self.Model.reset_state()
        values, probs, means, sigmas = self.evaluate_batch(xcolumns)
        valid_probs = self.valid_probs(probs, valids)
        returns = self.calculate_future_returns(rewards, valid_probs, values)
        h["returns"] = returns
        data = {
            "returns":  returns,
        }

        if False:
            print("--------- Brain.add_losses_from_episode: history:")
            print("        T:    ", T)
            print("   valies:    ", values)
            print("  actions:    ", actions)
            print("  rewards:    ", rewards)
            print("  returns:    ", returns)
            print("   valids:    ", valids)


        loss_values = self.Model.backprop(y_=returns[:,None], data=h)
        if False:
            print("--------- Brain.add_losses_from_episode: episode:")
            print("    probs:    ", probs)
            print("  rewards:    ", rewards)
            print("  returns:    ", returns)
            print("   values:    ", values)
            #print("  entropy:    ", loss_values["entropy_loss"])
            #print(" critic loss per step:", loss_values["critic_loss"]/T)

        
        #print("add_losses_from_episode: stats.keys:", list(stats.keys()))
        stats = {name:0.0 for name in ("critic_loss", "actor_loss", "entropy_loss", "invalid_action_loss")}
        for name, values in loss_values.items():
            stats[name] = np.sum(values)
        stats["sum_values"] = np.sum(values)
        stats["sum_rewards"] = np.sum(rewards)
        stats["sum_returns"] = np.sum(returns)
        
        return stats        
        
    def train_on_multi_episode_history(self, multi_ep_history):
        #
        # multi_ep_history is dictionary list of dictionaries{"observations":..., "actions":..., "returns":...}, one entry per episode
        #
        
        #print("--- train_on_history: episodes:", len(multi_ep_history))
        
        assert isinstance(multi_ep_history, list)
        assert all(isinstance(h, dict) for h in multi_ep_history)
        
        #log_n_actions = math.log(self.NActions)
        
        total_steps = 0
        
        sum_values = 0.0
        sum_advantages = 0.0
        sum_returns = sum_rewards = 0.0
        
        actor_losses = invalid_action_losses = critic_losses = entropy_losses = 0.0
        self.Model.reset_losses()
        
        for h in multi_ep_history:
            T = h["steps"]
            if T:       # episode was not empty
                stats = self.add_losses_from_episode(h)
                actor_losses += stats["actor_loss"]
                invalid_action_losses += stats["invalid_action_loss"]
                critic_losses += stats["critic_loss"]
                entropy_losses += stats["entropy_loss"]
                total_steps += T
        
                sum_values += stats["sum_values"]
                sum_returns += stats["sum_returns"] 
                sum_rewards += stats["sum_rewards"] 
        
        if False:
            print("Train on episodes:", len(multi_ep_history), "   steps:", total_steps)
            print("    Losses per step: critic:", critic_losses/total_steps, 
                 "  actor:", actor_losses/total_steps, 
                 "  invalid:", invalid_action_losses/total_steps, 
                 "  entropy:", entropy_losses/total_steps)

        if False:
            for g in self.Model.layer_gradients():
                if np.sum(g*g) > 1e10:
                    for l in self.Model.layers:
                        pgrads = l.PGradSum
                        if pgrads is not None:
                            for g in pgrads:
                                if np.sum(g*g) > 1e10:
                                    print(f"AC.Brain.train_on_multi_episode_history: Layer: {l}, g:{g}")
        grads2 = [np.mean(g*g)/total_steps for g in self.Model.layer_gradients()]
        self.Model.apply_deltas()
        
        stats = dict(
            actor_loss = actor_losses/total_steps,
            critic_loss = critic_losses/total_steps,
            entropy_loss = entropy_losses/total_steps,
            entropy = entropy_losses/total_steps,
            invalid_action_loss = invalid_action_losses/total_steps,
            average_reward = sum_rewards/total_steps,
            average_return = sum_returns/total_steps,
            average_value = sum_values/total_steps,
            average_grad_squared = grads2
        )
            
        return total_steps, stats
        
    def policy(self, action_probs, means, sigmas, valid_masks, training):
        actions = controls = None
        if action_probs:
            actions = []
            if valid_masks is None:
                valid_masks = [None]*len(action_probs)
            for probs, valids in zip(action_probs, valid_masks):
                probs = np.squeeze(probs)
                if not training:
                    # make it a bit more greedy
                    probs = probs**2
                    probs = probs/np.sum(probs)

                if valids is not None:
                    probs = self.valid_probs(probs, valids)

                action = np.random.choice(len(probs), p=probs)
                actions.append(action)
            if len(actions) == 1:
                actions = actions[0]
        if means is not None and len(means):
            assert len(means) == len(sigmas)
            controls = np.random.normal(means, sigmas)
            if self.ControlBounds:
                low, high = self.ControlBounds
                controls = [np.clip(x, l, h) for x, l, h in zip(controls, low, high)]
                #print("clipped: low/clipped/high:", low, controls, high)
            
            
        return actions, controls

    def action(self, observation, valid_masks, training=True):
        if not isinstance(observation, list):
            observation = [observation]

        values, action_probs, means, sigmas = self.evaluate_batch(
            [s[None,...] for s in observation]
        )
        
        if self.DiscreteDims:
            action_probs = [a[0] for a in action_probs]

        if self.NControls:
            means = means[0]
            sigmas = sigmas[0]
            
        value = values[0]

        #print("Brain.action: value, action_probs, means, sigmas = ", value, action_probs, means, sigmas)
        actions, controls = self.policy(action_probs, means, sigmas, valid_masks, training)
        
        if self.NControls == 0:
            env_action = actions
        elif not self.DiscreteDims:
            env_action = controls
        else:
            env_action = (actions, controls)
        
        return value, action_probs, means, sigmas, actions, controls, env_action
        
    def evaluate_batch(self, observations, reset=False):
        if not isinstance(observations, list):
            observations = [observations]
        mbsize = len(observations[0])
        if reset:
            self.Model.reset_states()
        outputs = self.Model.compute(observations)
        means = sigmas = action_probs = None
        values, outputs = outputs[0], outputs[1:]
        if self.DiscreteDims:
            action_probs = outputs[:len(self.DiscreteDims)]
        if self.NControls:
            means, sigmas = outputs[-2:]
        
        return values[:,0], action_probs, means, sigmas        # convert values from [mb,1] to [mb]
        
    evaluate_sequence = evaluate_batch
        
class RNNBrain(Brain):
    #
    # Recurrent Brain model:
    #   inputs: 
    #       state:                  [mb, t, ...]
    #
    #   output:
    #       value                   [mb, t, 1]
    #       probs                   [mb, t, nactions] probabilities, dummy but present if NActions == 0
    #       means                   [mb, t, ncontrols] control means, dummy but present if NControls == 0
    #       sigmas                  [mb, t, ncontrols] control sigmas, dummy but present if NControls == 0
    #
    
    def default_model(self, input_shape, n_actions, hidden):
        #
        # call for 1 step probs and actions:
        #   probs, value = model.call(observation[:][None,:])       # +dim for t
        #   probs = probs[0,:]        -> probs[a]
        #   value = value[0,0]        -> value (scalar)
        #
        # call for the episode:
        #   probs, values = model.call(observations[:,:])
        #   probs = probs[:,:]        -> probs[t,a]
        #   values = values[:,0]      -> value[t]
        #

        assert len(input_shape) == 1
        n_inputs = input_shape[0]

        obs = Input((None, n_inputs))                   
        prev_action = Input((None, n_actions))                   
        obs_actions = Concatenate()(obs, prev_action)
        rnn = LSTM(hidden, return_sequences=True)(obs_actions)        
        obs_rnn = Concatenate()(obs, rnn)
        probs = Dense(n_actions, activation="softmax", name="probs")(obs_rnn)
        values = Dense(1, name="values")(obs_rnn)

        model = Model([prev_action, obs], [probs, values])
        model["value"] = values
        model["probs"] = probs
    
        return model

    def evaluate_single(self, action, state):
        if not isinstance(state, list):
            state = [state]
        states = [s[None, ...] for s in state]        # add t dimension
        probs, values = self.evaluate_many([action], states)
        #print("RNNBrain.model: evaluate_single: probs, values:", probs[0], values[0])
        return probs[0], values[0]

    def evaluate_many(self, states):
        if not isinstance(states, list):
            states = [states]
        inputs = states        
        inputs = [s[None, ...] for s in inputs]        # add minibatch dimension
        #print("RNNBrain.model: evaluate_many: states:", states)
        outputs = self.Model.compute(inputs)
        values = outputs[0]
        probs = means = sigmas = None
        probs, values = self.Model.compute(inputs)          
        #for l in self.Model.links():
        #    print(l,":",l.Layer, ": Y=", None if l.Y is None else l.Y.shape)
        return probs[0], values[0,:,0]
        
        
        
