import random, math
import numpy as np
from gradnet import Model, Input, Loss
from gradnet.layers import Dense, LSTM, Concatenate, Flatten
from gradnet.optimizers import get_optimizer
from gradnet.activations import get_activation
from gradnet.losses import get_loss
from gym import spaces

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
        
@jit(nopython=True)
def _calc_retruns(gamma, rewards, value_weights, values):
    prev_ret = 0.0
    prev_value = 0.0
    prev_w = 0.0
    T = len(rewards)
    t = T-1
    returns = np.empty((T,))
    #print("_calc_retruns:")
    #print("  rewards:      ", rewards)
    #print("  values:       ", values)
    #print("  value_weights:", value_weights)
    for r, w, v in list(zip(rewards, value_weights, values))[::-1]:
        ret = r + gamma*(prev_ret*(1-prev_w) + prev_value*prev_w)
        returns[t] = ret
        prev_ret = ret
        prev_value = v
        prev_w = w
        t -= 1
    return returns
    


def entropy_controls_loss(_, sigmas, data):
        # means, sigmas: [mb, ncontrols]
        sigmas = np.clip(sigmas, 1e-3, None)
        ncontrols = sigmas.shape[-1]
        sprod = np.prod(sigmas, axis=-1)**(2/ncontrols)
        entropy = ncontrols/2 * np.log(2*math.pi*math.e*sprod)
        grads = 1/sigmas
        return entropy, [grads]

def actor_controls_loss(_, means, sigmas, values, data):        
        sigmas = np.clip(sigmas, 1e-3, None)
        controls = np.array(data["controls"])
        #print("actor_controls_loss:")
        #print("            controls:", controls)
        #print("            means:   ", means)
        #print("            sigmas:  ", sigmas)
        logprobs = -math.log(2*math.pi)/2 - np.log(sigmas) - ((controls-means)/sigmas)**2/2
        returns = data["returns"][:,None]
        #print("actor_loss: logprobs:", logprobs.shape)
        #print("            controls:", controls.shape)
        #print("            returns:", data["returns"].shape)
        advantages = returns - values
        #print("            advantages:", advantages)
        losses = -advantages * logprobs
        grads_sigmas = -advantages * (((controls-means)/sigmas)**2 - 1)/sigmas
        grads_means = -advantages * (controls-means)/sigmas**2
        #print("                 lossed:", grads_means)
        #print("            grads means:", grads_means)
        #print("           grads sigmas:", grads_sigmas)
        return losses, [grads_means, grads_sigmas, None]


def entropy_actions_loss(_, probs, data):
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

def actor_actions_loss(_, probs, values, data):
        probs_shape = probs.shape
        mb = probs_shape[0]
        na = probs_shape[-1]
        values_shape = values.shape
        
        returns = data["returns"]
        actions = data["actions"]

        #print("ActorLoss: returns.shape:", returns.shape, "   values.shape:", values.shape)

        probs = probs.reshape((-1, probs_shape[-1]))
        values = values.reshape((-1,))
        #actions = actions.reshape((-1,))
        returns = returns.reshape((-1,))
        
        #print("ActorLoss: probs:",probs.shape)
        #print("          values:", values.shape)
        action_mask = np.eye(na)[actions]
        action_probs = np.sum(action_mask*probs, axis=-1)
        advantages = returns - values
        #print("ActorLoss: rewards:", rewards.shape, "   values:", values.shape, "   probs:", probs.shape, "   advantages:", advantages.shape,
        #    "   action_probs:", action_probs.shape)
        #print("ActorLoss: advantages.shape:", advantages.shape, "   action_probs.shape:", action_probs.shape)
        loss_grads = -advantages/np.clip(action_probs, 1.e-5, None)  
        #print("ActorLoss: action_mask.shape:", action_mask.shape, "   loss_grads.shape:", loss_grads.shape)
        grads = action_mask * loss_grads[:,None]

        grads = grads.reshape(probs_shape)
        
        losses = -advantages*np.log(np.clip(action_probs, 1.e-5, None)).reshape(values_shape)          # not really loss values
        
        #print("ActorLoss: losses:", losses)
        #print("ActorLoss: grads:", grads)
        
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
    
class Brain(object):
    
    def __init__(self, input_shape=None, num_actions=None, model=None, optimizer=None, hidden=128, gamma=0.99, 
            with_rnn = False,
            learning_rate = 0.001,       # learning rate for the common part and the Actor
            entropy_weight = 0.01,
            invalid_action_weight = 0.5,
            critic_weight = 1.0, actor_weight=1.0,
            cutoff = None, beta = None     # if cutoff is not None, beta is ignored
                                        # if cutoff is None and beta is None, entropy is used
                                        # otherwise beta is used
        ):
        
        #print("Brain(input_shape=", input_shape,"  num_actions=", num_actions, ")")
        
        if isinstance(input_shape, int):
            input_shape = (input_shape,)
        self.NActions = num_actions
        self.Optimizer = optimizer or get_optimizer("adagrad", learning_rate=learning_rate) 
                #optimizer("SGD", learning_rate=learning_rate, momentum=0.5)
        self.Cutoff = cutoff
        self.Beta = beta
        self.Gamma = gamma
        self.MeanRet = 0.0
        self.RetSTD =  1.0
        self.NormAlpha = 0.001
        self.FutureValueCutoff = cutoff

        self.InvalidActionWeight = invalid_action_weight
        self.CriticWeight = critic_weight
        self.ActorWeight = actor_weight
        self.EntropyWeight = entropy_weight

        if model is None:   model = self.create_model(input_shape, hidden)
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
        source_weights = source.get_weights()
        old = self.Model.get_weights()
        self.Model.update_weights(source_weights, alpha)
        return old
                    
    def reset_episode(self):    # overridable
        self.Model.reset_state()
        
    def evaluate_sequence(self, states):
        print("Brain.evaluate_many: states:", type(states), len(states), states)
        probs, values = self.Model.compute(states)
        return values[:,0], probs
        
    def action(self, state, valid_actions, training):
        probs, value = self.evaluate_state(state)
        action = self.policy(probs, training, valid_actions)
        return value, probs, action

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
        if True:
            print("calculate_future_returns_with_cutoff: gamma=", self.Gamma, "  cutoff=", cutoff)
            print("  rewards:", rewards)
            print("  powers: ", gamma_powers)
        rets = np.empty((T,), dtype=np.float32)
        for t in range(T):
            ret = 0.0
            jmax = T-t if cutoff is None else min(T-t, cutoff)
            ret = np.sum(gamma_powers[:jmax]*rewards[t:t+jmax])
            #print("t=", t, "   T=", T, "   cutoff=", cutoff,"   jmax=", jmax, "   discounted:", ret)
            if t+cutoff < T:
                #print("     +V:", vals[t+cutoff])
                ret += values[t+cutoff]*self.Gamma**cutoff
            print("t:", t, "   ret:", ret)
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
        observations = h["observations"]        # list of observatons: [(state1, state2, ...)]
        
        actions = h["actions"]
        valids = h["valid_actions"]
        T = len(actions)

        #print("Brain.add_losses_from_episode: observations:", type(observations), observations)

        # transpose observations history
        states_by_column = [np.array(column) for column in zip(*observations)]
        prev_actions = np.roll(actions, 1)
        prev_actions[0] = -1

        #print("episode observations shape:", observations.shape)
        #print("add_losses: reset_state()")
        self.Model.reset_state()
        probs, values = self.evaluate_states(observations)
        valid_probs = self.valid_probs(probs, valids)
        returns = self.calculate_future_returns(rewards, valid_probs, values)
        h["returns"] = returns

        h = self.format_episode_history(h)

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
        stats = {name:np.sum(values) for name, values in loss_values.items()}
        stats["sum_values"] = np.sum(values)
        stats["sum_rewards"] = np.sum(rewards)
        stats["sum_returns"] = np.sum(returns)

        stats = self.compile_losses_from_episode(stats)

        return stats        
        
    def train_on_multi_episode_history(self, multi_ep_history):
        #
        # multi_ep_history is dictionary list of dictionaries{"observations":..., "actions":..., "returns":...}, one entry per episode
        #
        
        #print("--- train_on_history ---")
        
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
            
                #print("train_on_multi_episode_history: h:")
                #for name, data in h.items():
                #    print("   ", name, data)
            
            
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
            
        # per_episode_losses is a dictionary:
        #
        # { loss_name -> [loss_value(episode_i), ...]}
        #
        return total_steps, stats


    # overridable
    def create_model(self, input_shape, num_actions, hidden):
        model = self.default_model(input_shape, num_actions, hidden)
        model   \
            .add_loss(Loss(critic_loss, model["value"]),                   self.CriticWeight, name="critic_loss")  \
            .add_loss(Loss(actor_loss, model["probs"], model["value"]),    self.ActorWeight, name="actor_loss")     \
            .add_loss(Loss(entropy_loss, model["probs"]),                  self.EntropyWeight, name="entropy_loss") \
            .add_loss(Loss(invalid_actions_loss, model["probs"]),          self.InvalidActionWeight, name="invalid_action_loss")   \
            .compile(optimizer=self.Optimizer)
        return model

    # 
    # Overridables
    #
    
    def policy(self, probs, training, valid_actions=None):
        """Produce action based on the probabilities vector

        Returns one of:
            integer - action for discrete environments
            ndarray - controls vector for continuous environments
            (int, ndarray) - for mixed
        """
        raise NotImplementedError()
    
    def default_model(self, input_shape, num_actions, hidden):
        inp = Input(input_shape, name="input")
        common1 = Dense(hidden, activation="relu", name="common1")(inp)
        common = Dense(hidden//2, activation="relu", name="common")(common1)

        #action1 = Dense(max(hidden//5, num_actions*5), activation="relu", name="action1")(common)
        probs = Dense(num_actions, name="action", activation="softmax")(common)
        
        #critic1 = Dense(hidden//5, name="critic1", activation="relu")(common)
        value = Dense(1, name="critic")(common)

        model = Model([inp], [probs, value])
        
        model["value"] = value
        model["probs"] = probs
        
        return model
        
    def compile_losses_from_episode(self, losses):
        return losses
        
    def format_episode_history(self, history):
        return history

    def evaluate_states(self, states):
        """Evaluates multiple states
        
        Input:
            states: list or ndarray of states
        
        Returns:
            (action probabilities[], values[])
            action probabilities: [Nstates, Nactions]
            values: [NStates]
        """
        #print("Brain.evaluate_many: states:", type(states), len(states))
        # transpose states
        states = np.array(states)
        probs, values = self.Model.compute(states)
        return probs, values[:,0]

    def evaluate_state(self, state):
        """Evaluate single state.
        
        Returns:
            tuple (action probabilities, value)
        """
        probs, values = self.evaluate_states([state])
        return probs[0,:], values[0]

    def evaluate_sequence(self, states):
        """Evaluates sequence of states. Same as evaluate_states(), except for RNN-based models.
        """
        return self.evaluate_states(states)

class BrainMixed(Brain):
    
    def __init__(self, observation_shape, nactions, ncontrols, **args):
        self.NActions = nactions
        self.NControls = ncontrols
        Brain.__init__(self, observation_shape, nactions, **args)
        #print("BrainContinuous(): NControls:", self.NControls)
        
    @staticmethod
    def from_env(env, **args):
        ncontrols = env.NControls if hasattr(env, "NControls") else 0
        nactions = env.NActions if hasattr(env, "NActions") else 0
        return BrainMixed(env.observation_space.shape, nactions, ncontrols, **args)
            
    def create_model(self, input_shape, hidden):
        model = self.default_model(input_shape, self.NActions, self.NControls, hidden)
        model.add_loss(Loss(critic_loss, model["value"]),                   self.CriticWeight, name="critic_loss")

        if self.NActions:
            #print('create_model: model["probs"]:', model["action_probs"], '   model["value"]:', model["value"])
            model \
                .add_loss(Loss(actor_actions_loss, model["action_probs"], model["value"]), self.ActorWeight, name="actor_loss.actions")     \
                .add_loss(Loss(invalid_actions_loss, model["action_probs"]),               self.InvalidActionWeight, name="invalid_action_loss")   \
                .add_loss(Loss(entropy_actions_loss, model["action_probs"]),               self.EntropyWeight, name="entropy_loss.actions")

        if self.NControls:
            model \
                .add_loss(Loss(actor_controls_loss, model["means"], model["sigmas"], model["value"]),    
                                                self.ActorWeight, name="actor_loss.controls")     \
                .add_loss(Loss(entropy_controls_loss, model["sigmas"]),             self.EntropyWeight, name="entropy_loss.controls")
            if not self.NActions:
                model.add_loss(Loss("zero", model["means"]),                        1.0, name="invalid_action_loss")
        model.compile(optimizer=self.Optimizer)
        return model

    def default_model(self, input_shape, num_actions, num_controls, hidden):
        inp = Input(input_shape, name="input")
        common1 = Dense(hidden, activation="relu", name="common1")(inp)
        common = Dense(hidden//2, activation="relu", name="common")(common1)

        value = Dense(1, name="critic")(common)
        
        sigmas = means = probs = action_probs = control_probs = None
        
        if num_actions == 0:
            sigmas = Dense(num_controls, name="sigmas", activation="softplus")(common)
            means = Dense(num_controls, name="means", activation="linear")(common)
            control_probs = probs = Concatenate()(means, sigmas)
        elif num_controls == 0:
            action_probs = probs = Dense(num_actions, name="action", activation="softmax")(common)
        else:
            sigmas = Dense(num_controls, name="sigmas", activation="softplus")(common)
            means = Dense(num_controls, name="means", activation="linear")(common)
            action_probs = Dense(num_actions, name="action", activation="softmax")(common)
            control_probs = Concatenate()(means, sigmas)
            probs = Concatenate()(means, sigmas, action_probs)

        model = Model([inp], [probs, value])
        
        model["value"] = value
        model["means"] = means
        model["sigmas"] = sigmas
        model["action_probs"] = action_probs
        model["control_probs"] = control_probs
        model["probs"] = probs
        
        return model

    def policy(self, probs, training, valid_actions=None):
        action = controls = None
        if self.NControls:
            means = probs[:self.NControls]
            sigmas = probs[self.NControls:self.NControls*2]
            controls = np.random.normal(means, sigmas)
        
        if self.NActions:
            action_probs = probs[self.NControls*2:]
            if valid_actions is not None:
                action_probs = self.valid_probs(action_probs, valid_actions)
            action = np.random.choice(self.NActions, p=action_probs)

        if controls is None:
            return action
        elif action is None:
            return controls
        else:
            return (action, controls)

    """
    def evaluate_state(self, state):
        if not isinstance(state, (tuple, list)):
            state = [state]
        state = [s[None,...] for s in state]        # add minibatch dimension
        #print("evaluate_single: state shapes:", [s.shape for s in state])
        #print("Brain.evaluate_state() calling Model.compute() with state:", state)
        probs, values = self.Model.compute(state)
        probs = probs[0,:]
        value = values[0,0] 
        #print("Brain.evaluate_state(): returning probs:", probs, "  value:", value)
        return probs, value

    def evaluate_states(self, states):
        #print("Brain.evaluate_many: states:", type(states), len(states))
        # transpose states
        states = np.array(states)
        probs, values = self.Model.compute(states)
        return probs, values[:,0]
    """
            
    def compile_losses_from_episode(self, losses):
        out = {}
        for name, value in losses.items():
            if '.' in name:
                prefix, _ = name.split('.', 1)
            else:
                prefix = name
            out[prefix] = out.get(prefix, 0.0) + value
        #print("compile_losses_from_episode: out:", out)
        return out

    def format_episode_history(self, history):
        # split (action, controls) tuples into separate items
        out = history
        if self.NControls:
            out = history.copy()
            if self.NActions:
                out = history.copy()
                out["actions"] = [a[0] for a in history["actions"]]
                out["controls"] = [a[1] for a in history["actions"]]
            else:
                out["controls"] = history["actions"]            
        return out

class BrainDiscrete(BrainMixed):

    def __init__(self, observation_space, nactions, **args):
        BrainMixed.__init__(self, observation_space, nactions, 0, **args)

    def policy(self, action_probs, training, valid_actions=None):
        if valid_actions is not None:
            action_probs = self.valid_probs(action_probs, valid_actions)
        return np.random.choice(self.NActions, p=action_probs)

class BrainContinuous(BrainMixed):

    def __init__(self, observation_space, ncontrols, **args):
        BrainMixed.__init__(self, observation_space, 0, ncontrols, **args)

    def policy(self, probs, training, valid_actions=None):
        means = probs[:self.NControls]
        sigmas = probs[self.NControls:self.NControls*2]
        return np.random.normal(means, sigmas)

class RNNBrain(Brain):
    
    def default_model(self, input_shape, n_actions, n_controls, hidden):
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
        rnn = LSTM(hidden, return_sequences=True)(obs)
        dense = Dense(hidden, activation="relu", name="dense_base")(obs)
        common = Dense(hidden//2, activation="relu", name="common")(Concatenate()(rnn, dense))
        value = Dense(1, name="critic")(common)
        
        sigmas = means = probs = action_probs = control_probs = None

        if num_actions > 0:
            action_probs = Dense(n_actions, name="action", activation="softmax")(common)
            
        if num_controls > 0:
            sigmas = Dense(n_controls, name="sigmas", activation="softplus")(common)
            means = Dense(n_controls, name="means", activation="linear")(common)
            control_probs = Concatenate()(means, sigmas)
        
        if control_probs is not None and action_probs is not None:
            probs = Concatenate()(control_probs, action_probs)
        elif control_probs is not None:
            probs = control_probs
        else:
            probs = action_probs

        model["value"] = value
        model["means"] = means
        model["sigmas"] = sigmas
        model["action_probs"] = action_probs
        model["control_probs"] = control_probs
        model["probs"] = probs
    
        return model

    def evaluate_states(self, states):
        """Evaluates a set of out-of-sequence states
        """
        states = [s[None, ...] for s in states]        # add t dimension
        probs, values = self.Model.compute(states)          
        return probs[:,0], values[:,0,0]

    def evaluate_sequence(self, sequence):
        probs, values = self.Model.compute([sequence])  # add minibatch dimension
        return probs[0], values[0,:,0]

        
