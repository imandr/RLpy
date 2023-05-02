import random, math
import numpy as np
from gradnet import Model, Input, Loss
from gradnet.layers import Dense, LSTM, Concatenate, Flatten
from gradnet.optimizers import get_optimizer
from gradnet.activations import get_activation
from gradnet.losses import get_loss
from rlpy.gradnet.AC import Brain
from gym import spaces

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
        controls = data["actions"]
        #print("actor_loss:")
        #print("            controls:", controls)
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

class BrainMixed(Brain):
    
    def __init__(self, observation_space, nactions, ncontrols, **args):
        self.NActions = nactions
        self.NControls = ncontrols
        Brain.__init__(self, observation_space, nactions, **args)
        #print("BrainContinuous(): NControls:", self.NControls)
            
    def create_model(self, input_shape, hidden):
        model = self.default_model(input_shape, self.NActions, self.NControls, hidden)
        model.add_loss(Loss(critic_loss, model["value"]),                   self.CriticWeight, name="critic_loss")

        if self.NActions:
            print('create_model: model["probs"]:', model["action_probs"], '   model["value"]:', model["value"])
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

class BrainDiscrete(BrainMixed):
    
    def __init__(self, observation_space, nactions, **args):
        BrainMixed.__init__(self, observation_space, nactions, 0, **args)

class BrainContinuous(BrainMixed):
    
    def __init__(self, observation_space, ncontrols, **args):
        BrainMixed.__init__(self, observation_space, 0, ncontrols, **args)
