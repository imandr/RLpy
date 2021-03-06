#
# Based on sample code provided with Keras documentation
#

import random, math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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
        


class Brain(object):
    
    def __init__(self, input_shape=None, num_actions=None, model=None, optimizer=None, hidden=128, gamma=0.99, 
            learning_rate = 0.001,       # learning rate for the common part and the Actor
            entropy_weight = 0.01,
            invalid_action_weight = 0.5,
            critic_weight = 1.0,
            cutoff = 1, beta = None     # if cutoff is not None, beta is ignored
                                        # if cutoff is None and beta is None, entropy is used
                                        # otherwise beta is used
        ):
        if isinstance(input_shape, int):
            input_shape = (input_shape,)
        self.NActions = num_actions
        if model is None:   
            model = self.default_model(input_shape, num_actions, hidden)
        self.Model = model
        if optimizer is None:   optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
        self.Optimizer = optimizer
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
        
    def __call__(self, inputs):
        return self.Model(inputs)       # returns tensors
        
    def get_weights(self):
        return self.Model.get_weights()
        
    def weights_as_dict(self):
        out = {}
        for l in self.Model.layers:
            w = l.get_weights()
            if w is not None:
                if isinstance(w, list):
                    for i, wi in enumerate(w):
                        out[l.name+f"_{i}"] = wi
                else:
                    out[l.name] = w
        return out

    def set_weights(self, weights):
        self.Model.set_weights(weights)

    def update_weights(self, target_weights, alpha=1.0):
        my_w = self.get_weights()
        self.set_weights([w0 + alpha*(w1-w0) for w0, w1 in zip(my_w, target_weights)])        
        
    def update_from_brain(self, other, alpha=1.0):
        self.update_weights(other.get_weights(), alpha)
        
    def default_model(self, input_shape, num_actions, hidden):
        inputs = layers.Input(shape=input_shape, name="input")
        common1 = layers.Dense(hidden, activation="relu", name="common1")(inputs)
        common = layers.Dense(hidden//2, activation="relu", name="common")(common1)

        #action1 = layers.Dense(max(hidden//5, num_actions*2), activation="relu", name="action1")(common)
        action = layers.Dense(num_actions, activation="softmax", name="action")(common)
        
        #critic1 = layers.Dense(hidden//5, name="critic1", activation="softplus")(common)
        critic = layers.Dense(1, name="critic")(common)

        return keras.Model(inputs=[inputs], outputs=[action, critic])
        
    def reset_episode(self):    # overridable
        pass
        
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
        
    def probs(self, state):
        # add the batch dimension and make sure it's a list
        if isinstance(state, (list, tuple)):
            # multi-input model
            state = [s[None,...] for s in state]
        else:
            state = [state[None,...]]
        probs, _ = self.Model(state)
        return probs.numpy()[0]
        
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
    
    def __calculate_future_returns(self, rewards, probs, values):
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
        returns = []

        prev_ret = 0.0
        prev_value = 0.0
        prev_w = 0.0
        T = len(rewards)
        t = T-1
        returns = np.empty((T,))
        for r, w, v in list(zip(rewards, value_weights, values))[::-1]:
            ret = r + self.Gamma*(prev_ret*(1-prev_w) + prev_value*prev_w)
            returns[t] = ret
            prev_ret = ret
            prev_value = v
            prev_w = w
            t -= 1
        assert t == -1
        return returns
        
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

    def advantages_gae(self, rewards, probs, values):
        T = len(rewards)
        deltas = np.empty_like(rewards)
        deltas[:-1] = rewards[:-1] + self.Gamma * values[1:]
        deltas[-1] = rewards[-1]
        gamma_lambda_powers = (self.Gamma*self.Lambda)**np.arange(len(rewards))
        return np.array([np.sum(gamma_lambda_powers[:T-t]*deltas[t:]) for t in range(T)])

    def valid_probs(self, probs, valid_masks):
        if valid_masks is not None and valid_masks[0] is not None:
            probs = valid_masks * probs
            probs = probs/np.sum(probs, axis=-1, keepdims=True)
        return probs
        
    def losses_from_episode(self, h):
        critic_losses = []
        actor_losses = []
        invalid_action_losses = []
        entropy_losses = []

        log_n_actions = math.log(self.NActions)
        
        rewards = h["rewards"]
        observations = h["observations"]
        actions = h["actions"]
        valids = h["valids"]
        T = len(actions)

        # transpose observations history
        xcolumns = [np.array(column) for column in zip(*observations)]

        #print("episode observations shape:", observations.shape)
        probs, values = self.Model(xcolumns)
        values = values[:,0]
        values_numpy = values.numpy()
        sum_values = np.sum(values_numpy)
        
        valid_probs = self.valid_probs(probs.numpy(), valids)
        
        returns = self.calculate_future_returns(rewards, valid_probs, values_numpy)
        sum_returns = np.sum(returns)
        sum_rewards = np.sum(rewards)
                        
        #
        # Critic losses
        #
        diff = values - returns
        episode_ctiric_loss = tf.reduce_sum(diff*diff)
        if False:
            print("rewards:", rewards)
            print("returns:", returns)
            print("values: ", values_numpy)
            print("diffs:  ", diff.numpy())
            print("critic loss per step:", episode_ctiric_loss.numpy()/T)  #, " <==============" if episode_ctiric_loss.numpy() > 1.0 else "")
        critic_losses.append(episode_ctiric_loss)

        #
        # Actor losses
        #
        advantages = (returns - values).numpy()
        sum_advantages = np.sum(advantages)
        #print(returns.shape, values.shape, diffs.shape)
        action_mask = np.zeros(probs.shape)
        for i, action in enumerate(actions):
            action_mask[i, action] = 1.0
        #print(logprobs.shape, action_mask.shape, diffs.shape)
        action_probs = tf.reduce_sum(probs*action_mask, axis=-1)
        logprobs = tf.math.log(tf.clip_by_value(action_probs, 1e-5, 1-1e-5))
        problosses = -logprobs * advantages
        episode_actor_loss = tf.reduce_sum(problosses)
        actor_losses.append(episode_actor_loss)
        
        #
        # Invalid actions loss - used to unconditionally reduce the probability of invalid actions
        #
        
        if valids is not None:  
            # invalid action losses
            eposide_invalid_action_loss = tf.reduce_sum(probs*probs*(1-valids))
            invalid_action_losses.append(eposide_invalid_action_loss)
        #    
        # Entropy loss - used prevent "bad" actions probabilities to go to complete zero, to encourage exploration 
        #
        entropy_per_step = -tf.reduce_sum(probs*tf.math.log(tf.clip_by_value(probs, 1e-5, 1.0)), axis=-1)            
        episode_entropy_loss = -tf.reduce_sum(entropy_per_step)
        entropy_losses.append(episode_entropy_loss)
        
        if False:
            entropy = entropy_per_step.numpy()
            print("--------- Brain.train_on_multi_episode_history: episode:")
            #print("  rewards:    ", rewards)
            print("  returns:    ", returns)
            print("  values:     ", values_numpy)
            print("  probs[a]:   ", action_probs.numpy())
            print("  advantages: ", advantages)
            print("  logplosses: ", problosses.numpy())
            #print("  entropy:    ", entropy)
        
        
        return dict(
            entropy=entropy_losses,
            critic=critic_losses,
            actor=actor_losses,
            invalids=invalid_action_losses,
            sum_values = sum_values,
            sum_advantages = sum_advantages,
            sum_rewards = sum_rewards,
            sum_returns = sum_returns
        )
        
    def train_on_multi_episode_history(self, multi_ep_history):
        #
        # multi_ep_history is dictionary list of dictionaries{"observations":..., "actions":..., "returns":...}, one entry per episode
        #
        
        #print("--- train_on_history ---")
        
        assert isinstance(multi_ep_history, list)
        assert all(isinstance(h, dict) for h in multi_ep_history)
        
        huber_loss = keras.losses.Huber()
        mse = keras.losses.MeanSquaredError()
        total_steps = 0
        
        sum_values = 0.0
        sum_advantages = 0.0
        sum_returns = sum_rewards = 0.0
        
        with tf.GradientTape() as tape:
            
            critic_losses = []
            actor_losses = []
            invalid_action_losses = []
            entropy_losses = []
            
            for h in multi_ep_history:
                stats = self.losses_from_episode(h)
                actor_losses += stats["actor"]
                invalid_action_losses += stats["invalids"]
                critic_losses += stats["critic"]
                entropy_losses += stats["entropy"]
                T = len(h["actions"])
                total_steps += T
                
                sum_values += stats["sum_values"]
                sum_advantages += stats["sum_advantages"]
                sum_returns += stats["sum_returns"] 
                sum_rewards += stats["sum_rewards"] 
                
            actor_loss = sum(actor_losses)
            critic_loss = sum(critic_losses)
            invalid_action_loss = sum(invalid_action_losses) if invalid_action_losses else tf.convert_to_tensor(0.0)
            entropy_loss = sum(entropy_losses)
            total_loss = (
                actor_loss 
                + critic_loss*self.CriticWeight 
                + invalid_action_loss*self.InvalidActionWeight 
                + entropy_loss*self.EntropyWeight
            )

            if True:
                print("Train on episodes:", len(multi_ep_history), "   steps:", total_steps)
                print("    Losses per step: critic:", critic_loss.numpy()/total_steps, 
                     "  actor:", actor_loss.numpy()/total_steps, 
                     "  invalid:", invalid_action_loss.numpy()/total_steps, 
                     "  entropy:", entropy_loss.numpy()/total_steps)
            
            grads = tape.gradient(total_loss, self.Model.trainable_variables)

        grads2 = [np.mean(g.numpy()**2)/total_steps for g in grads]

        self.Optimizer.apply_gradients(zip(grads, self.Model.trainable_variables))
        averages = dict(
            actor = actor_loss.numpy()/total_steps,
            critic = critic_loss.numpy()/total_steps,
            entropy = entropy_loss.numpy()/total_steps,
            invalid_action = invalid_action_loss.numpy()/total_steps,
            average_reward = sum_rewards/total_steps,
            average_return = sum_returns/total_steps,
            average_value = sum_values/total_steps,
            average_advantage = sum_advantages/total_steps,
            average_grad_squared = grads2
        )
            
        # per_episode_losses is a dictionary:
        #
        # { loss_name -> [loss_value(episode_i), ...]}
        #
        return total_steps, averages
        
