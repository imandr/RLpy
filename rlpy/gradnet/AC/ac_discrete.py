import random, math
import numpy as np
from gradnet import Model, Input, Loss
from gradnet.layers import Dense, LSTM, Concatenate, Flatten
from gradnet.optimizers import get_optimizer
from gradnet.activations import get_activation
from gradnet.losses import get_loss
from rlpy.gradnet.AC import Brain
from gym import spaces

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

class BrainDiscrete(Brain):
    
    def __init__(self, observation_space, nactions, **args):
        Brain.__init__(self, observation_space, nactions, **args)
        self.NActions = nactions
        #print("BrainContinuous(): NControls:", self.NControls)
            
    def create_model(self, input_shape, num_actions, hidden):
        model = self.default_model(input_shape, num_actions, hidden)
        model   \
            .add_loss(Loss(critic_loss, model["value"]),                   self.CriticWeight, name="critic_loss")  \
            .add_loss(Loss(actor_loss, model["probs"], model["value"]),    self.ActorWeight, name="actor_loss")     \
            .add_loss(Loss(entropy_loss, model["probs"]),                  self.EntropyWeight, name="entropy_loss") \
            .add_loss(Loss(invalid_actions_loss, model["probs"]),          self.InvalidActionWeight, name="invalid_action_loss")   \
            .compile(optimizer=self.Optimizer)
        return model

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

    def policy(self, probs, training, valid_actions=None):
        if False and not training:
            # make it a bit more greedy
            probs = probs*probs
            probs = probs/np.sum(probs)

        if valid_actions is not None:
            probs = self.valid_probs(probs, valid_actions)

        action = np.random.choice(self.NActions, p=probs)
        return action

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
        
