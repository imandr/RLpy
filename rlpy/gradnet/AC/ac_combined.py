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

def actor_loss(_, probs, values, data):
        probs_shape = probs.shape
        mb = probs_shape[0]
        na = probs_shape[-1]
        values_shape = values.shape
        
        returns = data["returns"]
        actions = data["actions"]

        probs = probs.reshape((-1, probs_shape[-1]))
        values = values.reshape((-1,))
        actions = actions.reshape((-1,))
        returns = returns.reshape((-1,))
        
        #print("ActorLoss: probs:",probs.shape)
        #print("          values:", values.shape)
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
        return losses, [grads, None]

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
        controls = data["actions"]
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
    
    
    def __init__(self, observation_shapes, num_actions=0, num_controls=0, model=None, recurrent=False,
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
        
        if isinstance(observation_shapes, int):
            observation_shapes = [(observation_shapes,)]
        elif isinstance(observation_shapes, tuple):
            observation_shapes = [observation_shapes]
        elif not isinstance(observation_shapes, list):
            raise ValueError("Unsupported vaue type for observation_shapes: %s" % (type(observation_shapes),))
        
        self.ObservationShapes = observation_shapes
        self.NActions = num_actions      
        self.NControls = num_controls      
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
        
        # ActionVectors[action] = [0,0,0,....,1,...] for actions and [all zeros] for action=-1
        self.ActionVectors = np.concatenate(
            [np.eye(self.NActions), np.zeros((1,self.NActions))],
            axis = 0
        )

        if model is None:   
            model = self.default_model(self.ObservationShapes, self.NActions, self.NControls, hidden) if not with_rnn \
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
                    
    def default_model(self, input_shapes, num_actions, num_controls, hidden):

        inputs = [Input(input_shape, name=f"input_{i}") for i, input_shape in enumerate(input_shapes)]

        if len(inputs) == 1:
            inp = inputs[0]
        else:
            flattened = [Flatten()(inp) if len(inp.Shape)>1 else inp for inp in inputs]
            inp = Concatenate()(*flattened)

        common1 = Dense(hidden, activation="relu", name="common1")(inp)
        common = Dense(max(hidden//2, num_actions*5), activation="relu", name="common")(common1)

        critic1 = Dense(hidden//5, name="critic1", activation="relu")(common)
        value = Dense(1, name="critic")(critic1)

        if num_actions > 0:
            action1 = Dense(max(hidden//5, num_actions*5), activation="relu", name="action1")(common)
            probs = Dense(num_actions, name="action", activation="softmax")(action1)
        else:
            probs = Constant()
            
        if num_controls > 0:
            means = Dense(num_controls, activation="linear", name="means")(common)
            sigmas = Dense(num_controls, activation="linear", name="sigmas")(common)
        else:
            means = sigmas = Constant()
            
        model = Model(inputs, [value, probs, means, sigmas])
        
        model["value"] = value
        model["probs"] = probs
        model["means"] = value
        model["sigmas"] = probs
        
        return model

    def add_losses(self, model, critic_weight=1.0, actor_weight=1.0, entropy_weight=0.01, invalid_action_weight=10.0, **unused):

        model.add_loss(Loss(critic_loss, model["value"]),                   critic_weight, name="critic_loss")
            
        if self.NActions > 0:
            model   \
            .add_loss(Loss(actor_loss, model["probs"], model["value"]),    actor_weight, name="actor_loss")     \
            .add_loss(Loss(entropy_loss, model["probs"]),                  entropy_weight, name="entropy_loss") \
            .add_loss(Loss(invalid_actions_loss, model["probs"]),          invalid_action_weight, name="invalid_action_loss")
        else:
            model   \
            .add_loss(Loss("zero", model["value"]),  actor_weight, name="actor_loss")     \
            .add_loss(Loss("zero", model["value"]),  entropy_weight, name="entropy_loss") \
            .add_loss(Loss("zero", model["value"]),  invalid_action_weight, name="invalid_action_loss")
            
        if self.NControls > 0:
            model   \
            .add_loss(Loss(cont_actor_loss,   model["means"], model["sigmas"], model["value"]),    actor_weight, name="cont_actor_loss")     \
            .add_loss(Loss(cont_entropy_loss, model["sigmas"]),                  entropy_weight, name="cont_entropy_loss")
        else:
            model   \
            .add_loss(Loss("zero", model["Value"]),    actor_weight, name="cont_actor_loss")     \
            .add_loss(Loss("zero", model["value"]),                  entropy_weight, name="cont_entropy_loss")

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
        prev_actions = np.roll(actions, 1)
        prev_actions[0] = -1
        probs, values = self.evaluate_many(prev_actions, xcolumns)

        valid_probs = self.valid_probs(probs, valids)
        returns = self.calculate_future_returns(rewards, valid_probs, values)
        h["returns"] = returns

        loss_values = self.Model.backprop(y_=returns[:,None], data=h)
        if False:
            print("--------- Brain.add_losses_from_episode: episode:")
            print("    probs:    ", probs)
            print("  rewards:    ", rewards)
            print("  returns:    ", returns)
            print("   values:    ", values)
            print("  entropy:    ", loss_values["entropy_loss"])
            print(" critic loss per step:", loss_values["critic_loss"]/T)

        
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
            if len(h.get("actions")):       # episode was not empty
                stats = self.add_losses_from_episode(h)
                actor_losses += stats["actor_loss"]
                invalid_action_losses += stats["invalid_action_loss"]
                critic_losses += stats["critic_loss"]
                entropy_losses += stats["entropy_loss"]
                T = len(h["actions"])
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
        
    def evaluate_step(self, prev_action, prev_controls, state, reset=False):
        if not isinstance(state, list):
            state = [state]
        if prev_controls is None and self.NControls:
            prev_controls = np.zeros((self.NControls,))
        # add mb dimension and run as minibatch
        values, probs, means, sigmas = self.evaluate_batch(
            self.ActionVectors[prev_action][None,...], 
            prev_controls[None,...] if prev_controls is not None else None, 
            [s[None,...] for s in state]
        )
        # remove mb dimension
        return values[0], probs[0], means[0], sigmas[0]
        
    def policy(self, prev_action, state, training, valid_actions):
        # prev_action here is a tuple (prev_discreet_action, [prev_controls]), or None if this is first step in the episode
        #
        # returns tuple:
        # (discreete action or None, [controls])
        #

        if prev_action == None:
            prev_discreet_action = prev_controls = None
        else:
            prev_discreet_action, prev_controls = prev_action
        
        value, action_probs, means, sigmas = self.evaluate_single(prev_discreet_action, prev_controls, state)
        
        action = -1
        controls = None
        
        if len(action_probs):
            probs = np.squeeze(probs)
            if not training:
                # make it a bit more greedy
                probs = probs**2
                probs = probs/np.sum(probs)

            if valid_actions is not None:
                probs = self.valid_probs(probs, valid_actions)

            action = np.random.choice(self.NActions, p=probs)
        
        if len(means):
            assert len(means) == len(sigmas)
            controls = np.random.normal(means, sigmas)
            
        return action, controls
        
    def evaluate_sequence(self, prev_actions, prev_controls, states, reset=False):
        #
        # inputs: prev_action: [t]                  -1 if no previous action
        #         prev_controls: [t, ncontrols]     
        #         state: list of [t, ...]
        #
        if not isinstance(state, list):
            state = [state]
            
        return self.evaluate_batch(
            prev_actions, 
            prev_controls, 
            state,
            reset=reset
        )
        
    def evaluate_batch(self, prev_actions, prev_controls, states, reset=False):
        if not isinstance(states, list):
            states = [states]
        mbsize = len(states[0])
        inputs = []
        if self.NActions:
            inputs.append(self.ActionVectors[prev_actions])
        if self.NControls:
            inputs.append(prev_controls)
        inputs += states
        if reset:
            self.Model.reset_states()
        values, probs, means, sigmas = self.Model.compute(inputs)       
        # remove minibatch dimension 
        return values[:,0], probs, means, sigmas        # convert values from [mb,1] to [mb]
        
class RNNBrain(Brain):
    #
    # Recurrent Brain model:
    #   inputs: 
    #       prev_action_vectors:    [mb, t, nactions] - optional, present only if NActions > 0
    #       prev_controls:          [mb, t, ncontrols] - optional, present only if NControls > 0
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

    def evaluate_many(self, actions, states):
        if not isinstance(states, list):
            states = [states]
        inputs = [self.ActionVectors[actions]] + states        
        inputs = [s[None, ...] for s in inputs]        # add minibatch dimension
        #print("RNNBrain.model: evaluate_many: states:", states)
        probs, values = self.Model.compute(inputs)          
        #for l in self.Model.links():
        #    print(l,":",l.Layer, ": Y=", None if l.Y is None else l.Y.shape)
        return probs[0], values[0,:,0]
        
        
        
