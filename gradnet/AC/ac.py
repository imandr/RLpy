import random, math
import numpy as np
from gradnet import Model, Input, Loss
from gradnet.layers import Dense, LSTM, Concatenate
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
    
class EntropyLoss(Loss):
    
    def compute(self, data):
        probs = self.Inputs[0].Y
        p = np.clip(probs, 1e-5, None)
        values = -np.sum(p*np.log(p), axis=-1)
        grads = np.log(p)+1.0
        n = probs.shape[-1]
        logn = math.log(n)
        k = abs(logn-1.0)/n
        #print("Entropy loss: k=", k)
        self.Grads = [grads*k]
        self.Values = values/logn  
        return self.value

class ActorLoss(Loss):
    
    def compute(self, data):
        returns = data["returns"]
        actions = data["actions"]
        probs, values = self.Inputs[0].Y, self.Inputs[1].Y
        probs_shape = probs.shape
        mb = probs_shape[0]
        na = probs_shape[-1]
        values_shape = values.shape
        
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
        
        self.Grads = [grads, None]
        self.Values = -advantages*np.log(np.clip(action_probs, 1.e-5, None)).reshape(values_shape)          # not really loss values

        if False:
            print("ActorLoss:")
            #print("      probs:", probs)
            #print("    actions:", actions)
            print("    returns:", returns)
            print("     values:", values)
            print("   probs[a]:", action_probs)
            print("  advatages:", advantages)
            print("  logplosses: ", self.Values)
            
            #print("action_mask:", action_mask)
            #print(" loss_grads:", loss_grads)
            #print("      grads:", grads)


        return self.value

class InvalidActionLoss(Loss):
    
    def compute(self, data):
        probs = self.Inputs[0].Y
        valid_mask = data.get("valids")
        if valid_mask is not None:
            self.Values = np.mean(probs*probs*(1-valid_mask), axis=-1)
            self.Grads = [2*(1-valid_mask)*probs]
        else:
            self.Values = np.zeros((len(probs),))
            self.Grads = [None]
        return self.value

class Brain(object):
    
    def __init__(self, input_shape=None, num_actions=None, model=None, optimizer=None, hidden=128, gamma=0.99, 
            with_rnn = False,
            learning_rate = 0.001,       # learning rate for the common part and the Actor
            entropy_weight = 0.01,
            invalid_action_weight = 0.5,
            critic_weight = 1.0, actor_weight=1.0,
            cutoff = 1, beta = None     # if cutoff is not None, beta is ignored
                                        # if cutoff is None and beta is None, entropy is used
                                        # otherwise beta is used
        ):
        if isinstance(input_shape, int):
            input_shape = (input_shape,)
        self.NActions = num_actions
        self.Optimizer = optimizer or get_optimizer("SGD", learning_rate=learning_rate) 
                #optimizer("SGD", learning_rate=learning_rate, momentum=0.5)
        self.Cutoff = cutoff
        self.Beta = beta
        self.Gamma = gamma
        self.MeanRet = 0.0
        self.RetSTD =  1.0
        self.NormAlpha = 0.001
        self.FutureValueCutoff = cutoff
        
        self.EntropyWeight = entropy_weight
        self.InvalidActionWeight = invalid_action_weight
        self.CriticWeight = critic_weight

        if model is None:   
            model = self.default_model(input_shape, num_actions, hidden) if not with_rnn \
                    else self.default_rnn_model(input_shape, num_actions, hidden)

        model.add_loss(ActorLoss(model["probs"], model["value"]), actor_weight, name="actor_loss")
        model.add_loss(get_loss("mse")(model["value"]), critic_weight, name="critic_loss")
        model.add_loss(EntropyLoss(model["probs"]), entropy_weight, name="entropy_loss")
        model.add_loss(InvalidActionLoss(model["probs"]), invalid_action_weight, name="invalid_action_loss")
        model.compile(optimizer=self.Optimizer)

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
        
    def default_model(self, input_shape, num_actions, hidden):
        inp = Input(input_shape, name="input")
        common1 = Dense(hidden, activation="relu", name="common1")(inp)
        common = Dense(hidden//2, activation="relu", name="common")(common1)

        #action1 = Dense(max(hidden//5, num_actions//2), activation="relu", name="action1")(common)
        probs = Dense(num_actions, name="action", activation="softmax")(common)
        
        #critic1 = Dense(hidden//5, name="critic1", activation="relu")(common)
        value = Dense(1, name="critic")(common)

        model = Model([inp], [probs, value])
        
        model["value"] = value
        model["probs"] = probs
        
        return model
        
    def reset_episode(self):    # overridable
        self.Model.reset_state()
        
    def policy(self, probs, training, valid_actions=None):
        
        probs = np.squeeze(probs)
        if not training:
            # make it a bit more greedy
            probs = probs*probs
            probs = probs/np.sum(probs)

        if valid_actions is not None:
            probs = self.valid_probs(probs, valid_actions)

        action = np.random.choice(self.NActions, p=probs)
        return action
        
    def evaluate_single(self, state):
        if not isinstance(state, list):
            state = [state]
        state = [s[None,...] for s in state]        # add t dimension
        #print("evaluate_single: state shapes:", [s.shape for s in state])
        probs, values = self.Model.compute(state)        
        return probs[0,:], values[0,0]
        
    def evaluate_many(self, states):
        probs, values = self.Model.compute(states)
        return probs, values[:,0]
        
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
        valids = h["valids"]
        T = len(actions)

        # transpose observations history
        xcolumns = [np.array(column) for column in zip(*observations)]

        #print("episode observations shape:", observations.shape)
        #print("add_losses: reset_state()")
        self.Model.reset_state()
        probs, values = self.evaluate_many(xcolumns)

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
        stats = {name:np.sum(values) for name, values in loss_values.items()}
        stats["sum_values"] = np.sum(values)
        stats["sum_rewards"] = np.sum(rewards)
        stats["sum_returns"] = np.sum(returns)
        
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
        
class RNNBrain(Brain):
    
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
        rnn = LSTM(hidden, return_sequences=True)(obs)
        dense = Dense(hidden, activation="relu", name="dense_base")(obs)
        concatenated = Concatenate()(rnn, dense)
        common = Dense(hidden, activation="relu", name="common")(concatenated)
        probs = Dense(n_actions, activation="softmax")(common)
        values = Dense(1, name="dense_value")(concatenated)

        model = Model(obs, [probs, values])

        model["value"] = values
        model["probs"] = probs
    
        return model

    def evaluate_single(self, state):
        if not isinstance(state, list):
            state = [state]
        state = [s[None, None, ...] for s in state]        # add minibatch and t dimension
        #print("evaluate_single: state shapes:", [s.shape for s in state])
        probs, values = self.Model.compute(state)        
        return probs[0,0,:], values[0,0,0]

    def evaluate_many(self, states):
        if not isinstance(states, list):
            states = [states]
        states = [s[None, ...] for s in states]        # add minibatch and t dimension
        probs, values = self.Model.compute(states)          # add minibatch dimension
        #for l in self.Model.links():
        #    print(l,":",l.Layer, ": Y=", None if l.Y is None else l.Y.shape)
        return probs[0], values[0,:,0]
        
        
        
