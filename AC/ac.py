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
            actor_lag = 0.2,       # critic rate slow down (1.0 means no slow down)
            entropy_weight = 0.01,
            invalid_action_weight = 0.5,
            critic_weight = 1.0,
            epsilon = 0.0,
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
        if optimizer is None:   optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.Optimizer = optimizer
        self.Cutoff = cutoff
        self.Beta = beta
        self.OffPolicyProbability = epsilon
        
        self.ActorLag = actor_lag
        
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
        common = layers.Dense(hidden/2, activation="relu", name="common")(common1)

        action1 = layers.Dense(max(hidden/5, num_actions)/2, activation="relu", name="action1")(common)
        action = layers.Dense(num_actions, activation="softmax", name="action")(action1)
        
        critic1 = layers.Dense(hidden/5, name="critic1", activation="softplus")(common)
        critic = layers.Dense(1, name="critic")(critic1)

        return keras.Model(inputs=[inputs], outputs=[action, critic])
        
    def policy(self, probs, training):
        probs = np.squeeze(probs)
        if training:
            action = np.random.choice(self.NActions, p=probs)
        else:
            # not training - make it a bit more greedy
            probs = probs*probs
            action = np.random.choice(self.NActions, p=probs/np.sum(probs))
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

        
    def valid_probs(self, probs, valid_masks):
        if valid_masks is not None and valid_masks[0] is not None:
            probs = valid_masks * probs
            probs = probs/np.sum(probs, axis=-1, keepdims=True)
        return probs
            
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
        log_n_actions = math.log(self.NActions)
        
        sum_values = 0.0
        sum_advantages = 0.0
        sum_returns = sum_rewards = 0.0

        with tf.GradientTape() as tape:
            
            critic_losses = []
            actor_losses = []
            invalid_action_losses = []
            entropy_losses = []
            
            for h in multi_ep_history:
                rewards = h["rewards"]
                observations = h["observations"]
                actions = h["actions"]
                valids = h["valids"]
                T = len(actions)
                total_steps += T
                                
                #print("episode observations shape:", observations.shape)
                probs, values = self.Model(observations)
                values = values[:,0]
                if False:
                    print("probs:", probs.numpy())
                values_numpy = values.numpy()
                sum_values += np.sum(values_numpy)
                
                valid_probs = self.valid_probs(probs.numpy(), valids)
                
                returns = self.calculate_future_returns(rewards, valid_probs, values_numpy)
                sum_returns += np.sum(returns)
                sum_rewards += np.sum(rewards)
                                
                # critic losses
                #print("mse(",returns[:,None], values,")")
                diff = values - returns
                episode_ctiric_loss = tf.reduce_sum(diff*diff)
                if False:
                    print("rewards:", rewards)
                    print("returns:", returns)
                    print("values: ", values_numpy)
                    print("diffs:  ", diff.numpy())
                    print("critic loss:", episode_ctiric_loss.numpy())  #, " <==============" if episode_ctiric_loss.numpy() > 1.0 else "")
                critic_losses.append(episode_ctiric_loss)

                # actor losses
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
                
                # entropy
                entropy = -(np.sum(valid_probs*np.log(np.clip(valid_probs, 1e-5, None)), axis=-1)/log_n_actions)
                if False:
                    print("--------- Brain.train_on_multi_episode_history: episode:")
                    print("  rewards:    ", rewards)
                    print("  returns:    ", returns)
                    print("  values:     ", values_numpy)
                    print("  probs[a]:   ", action_probs.numpy())
                    print("  advantages: ", advantages)
                    print("  logplosses: ", problosses.numpy())
                    print("  entropy:    ", entropy)
                
                
                if valids is not None:  
                    # invalid action losses
                    eposide_invalid_action_loss = tf.reduce_sum(probs*probs*(1-valids))
                    invalid_action_losses.append(eposide_invalid_action_loss)
                else:
                    eposide_invalid_action_loss = tf.convert_to_tensor(0.0)
                invalid_action_losses.append(eposide_invalid_action_loss)
                    
                # entropy loss
                entropy = -tf.reduce_sum(probs*tf.math.log(tf.clip_by_value(probs, 1e-5, 1.0)))/log_n_actions        # so that = 1 if the p distribution is uniform
                episode_entropy_loss = -entropy
                entropy_losses.append(episode_entropy_loss)
                
            actor_loss = sum(actor_losses)
            critic_loss = sum(critic_losses)
            invalid_action_loss = sum(invalid_action_losses)
            entropy_loss = sum(entropy_losses)
            total_loss = (
                actor_loss 
                + critic_loss*self.CriticWeight 
                + invalid_action_loss*self.InvalidActionWeight 
                + entropy_loss*self.EntropyWeight
            )

            if False:
                print("Losses: critic:", critic_loss.numpy()/total_steps, 
                    "  actor:", actor_loss.numpy()/total_steps, 
                    "  invalid:", invalid_action_loss.numpy()/total_steps, 
                    "  entropy:", entropy_loss.numpy()/total_steps)
            
            grads = tape.gradient(total_loss, self.Model.trainable_variables)

        self.Optimizer.apply_gradients(zip(grads, self.Model.trainable_variables))
        averages = dict(
            actor = actor_loss.numpy()/total_steps,
            critic = critic_loss.numpy()/total_steps,
            entropy = entropy_loss.numpy()/total_steps,
            invalid_action = invalid_action_loss.numpy()/total_steps,
            average_reward = sum_rewards/total_steps,
            average_return = sum_returns/total_steps,
            average_value = sum_values/total_steps,
            average_advantage = sum_advantages/total_steps
        )
            
        # per_episode_losses is a dictionary:
        #
        # { loss_name -> [loss_value(episode_i), ...]}
        #
        return total_steps, averages
        
class Agent(object):

    def __init__(self, brain, nactions, alpha=0.01):
        self.Brain = brain
        self.NActions = nactions
        self.RunningReward = None
        self.Alpha = alpha
        self.EpisodeReward = 0.0
        self.resetRunningReward()

    def resetRunningReward(self):
        self.RunningReward = None
                
    def play_episode(self, env, max_steps=None, render=False, training=False):
        action_probs_history = []
        rewards_history = []
        action_history = []
        observation_history = []
        valids_history = []
        
        state = env.reset()
        #print("p:", " ", state)
        self.EpisodeReward = 0.0
        done = False
        steps = 0
        
        while (not done) and (max_steps is None or steps < max_steps):
            #print("calculating probs...")
            if not isinstance(state, list):
                state = [state]                 # always a list of np arrays
            probs = self.Brain.probs(state)
            action = self.Brain.policy(probs, training)
            observation_history.append(state)
            action_history.append(action)
            action_probs_history.append(probs)
            
            state, reward, done, meta = env.step(action)
            #print("p:", action, state, reward, done)
            if render:
                env.render()
            
            rewards_history.append(reward)
            self.EpisodeReward += reward
            steps += 1
            #print("Agent.play_episode: done:", done)
        #print("returning")
        
        if self.RunningReward is None:
            self.RunningReward = self.EpisodeReward
        else:
            self.RunningReward += self.Alpha*(self.EpisodeReward - self.RunningReward)
        if observation_history:
            obs0 = observation_history[0]
            if isinstance(obs0, list):
                # multi-array observation
                observation_history = [np.array(column) for column in zip(*observation_history)]
            else:
                observation_history = np.array(observation_history)
        self.EpisodeHistory = dict(
            rewards = np.array(rewards_history),
            observations = observation_history,
            actions = np.array(action_history),
            probs = np.array(action_probs_history),
            episode_reward = self.EpisodeReward,
            valids = None       # not used for now
        )
        
        return self.EpisodeHistory
        
    def episode_history(self):
        return self.EpisodeHistory
        
    def train_on_multi_episode_history(self, multi_ep_history):
        return self.Brain.train_on_multi_episode_history(multi_ep_history)
        
            
class MultiAgent(object):

    def __init__(self, brain, nactions, alpha=0.01):
        self.Brain = brain
        self.NActions = nactions
        self.RunningReward = 0.0
        self.EpisodeReward = 0.0
        self.Alpha = alpha
        self.Reward = 0.0           # reward accunulated since last action
        self.resetRunningReward()

    def resetRunningReward(self):
        self.RunningReward = 0.0
        
    def reset(self, training=True):
        self.EpisodeReward = self.Reward = 0.0
        self.Observations = []
        self.States = []
        self.Rewards = []
        self.Actions = []
        self.ValidMoves = []
        self.Values = []
        self.ProbsHistory = []
        self.FirstMove = True
        self.Done = False
        #print("Agent[%d].reset()" % (id(self)%100,))
        
    def choose_action(self, observation, available_actions, training):
        self.Observations.append(observation)
        self.ValidMoves.append(available_actions)
        state = observation
        if not isinstance(state, list):     # state can be a list of np.arrays
            state = [state]
        state = [tf.expand_dims(tf.convert_to_tensor(s), 0) for s in state]
        #print("state:", state)
        self.States.append(state)
        action_probs, critic_value = self.Brain(state)
        masked_probs = action_probs = action_probs.numpy()[0]
        critic_value = critic_value.numpy()[0,0]
        
        #action_probs = action_probs[0]
        #critic_value = critic_value[0,0]
        
        #print("action_probs, critic_value:", action_probs.numpy(), critic_value.numpy())
        
        self.ProbsHistory.append(action_probs)      
        self.Values.append(critic_value)            

        if available_actions is not None:
            masked_probs = action_probs * available_actions
            masked_probs = masked_probs/np.sum(masked_probs)
            if not training:            # for now
                masked_probs = masked_probs*masked_probs
                masked_probs /= np.sum(masked_probs)
                #print("non-training policy")
        action = np.random.choice(self.NActions, p=masked_probs)
            
        self.Actions.append(action)
        #print("Agent", id(self)%10, " obs:", observation, "   probs:", action_probs, "   action:", action, "  masked probs:", masked_probs, "   value:", critic_value)    #, "  available:", available_actions, "    masked probs:", masked_probs, "   value:", critic_value.numpy())
        return action
        
    def action(self, observation, available_actions, training=True):
        # this is reward for the previuous action
        if not self.FirstMove:
            self.Rewards.append(self.Reward)
        self.FirstMove = False
        self.EpisodeReward += self.Reward
        self.Reward = 0.0
        action = self.choose_action(observation, available_actions, training)
        #print("Agent[%d].action() -> %d" % (id(self)%100, action))
        return action

    def reward(self, reward):
        self.Reward += reward
        #print("Agent[%d].reward(%.4f) accumulated=%.4f" % (id(self)%100, reward, self.Reward))
        
    def done(self, last_observation):
        #print("Agent[%d].done()" % (id(self)%100, ))
        #print("Agent[%d].done(reward=%f)" % (id(self)%100, reward))
        #print("Agent", id(self)%10, "done:", reward)
        if not self.Done:
            if not self.FirstMove:
                self.Rewards.append(self.Reward)
            self.EpisodeReward += self.Reward
            self.Reward = 0.0
            self.RunningReward += self.Alpha*(self.EpisodeReward - self.RunningReward)
            self.Done = True
        #self.Observations.append(observation)
        
    def episode_history(self):
        valids = np.array(self.ValidMoves) if self.ValidMoves[0] is not None else None
        out = {
            "rewards":          np.array(self.Rewards),
            "actions":          np.array(self.Actions),
            "valids":           valids,
            "observations":     np.array(self.Observations) #,
            #"episode_reward":   self.EpisodeReward
        }
        #print("Agent[%d].history():" % (id(self)%100,), *((k, len(lst) if lst is not None else "none") for k, lst in out.items()))
        #print("          valids:", out["valids"])
        #print("    observations:", out["observations"])
        #for obs, a, r in zip(out["observations"], out["actions"], out["rewards"]):
        #    print("    ", obs, a, r)
        #print("Multiagent: episode_history: shapes:", [(name, x.shape) for name, x in out.items()])
        return out

        
