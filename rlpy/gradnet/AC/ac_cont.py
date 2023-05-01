import random, math
import numpy as np
from gradnet import Model, Input, Loss
from gradnet.layers import Dense, LSTM, Concatenate, Flatten
from gradnet.optimizers import get_optimizer
from gradnet.activations import get_activation
from gradnet.losses import get_loss
from rlpy.gradnet.AC import Brain
from gym import spaces

def entropy_loss(_, sigmas, data):
        # means, sigmas: [mb, ncontrols]
        sigmas = np.clip(sigmas, 1e-3, None)
        ncontrols = sigmas.shape[-1]
        sprod = np.prod(sigmas, axis=-1)**(2/ncontrols)
        entropy = ncontrols/2 * np.log(2*math.pi*math.e*sprod)
        grads = 1/sigmas
        return entropy, [grads]

def actor_loss(_, means, sigmas, values, data):        
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
    
def critic_loss(_, values, data):
    returns = data["returns"]
    d = values - returns[:,None]
    losses = d*d
    grads = 2*d
    return losses, grads
    
class BrainContinuous(Brain):
    
    def __init__(self, observation_space, ncontrols, **args):
        Brain.__init__(self, observation_space, ncontrols, **args)
        self.NControls = ncontrols
        #print("BrainContinuous(): NControls:", self.NControls)
            
    def create_model(self, input_shapes, num_controls, hidden):
        model = self.default_model(input_shapes, num_controls, hidden)
        model   \
            .add_loss(Loss(critic_loss, model["value"]),                   self.CriticWeight, name="critic_loss")  \
            .add_loss(Loss(actor_loss, model["means"], model["sigmas"], model["value"]),    
                                                                            self.ActorWeight, name="actor_loss")     \
            .add_loss(Loss(entropy_loss, model["sigmas"]),                  self.EntropyWeight, name="entropy_loss")    \
            .add_loss(Loss("zero", model["means"]),                         self.InvalidActionWeight, name="invalid_action_loss")   \
            .compile(optimizer=self.Optimizer)
        return model
    
    def default_model(self, input_shapes, num_controls, hidden):
        if isinstance(input_shapes, tuple):
            input_shapes = [input_shapes]
        inputs = [Input(input_shape, name=f"input_{i}") for i, input_shape in enumerate(input_shapes)]
        if len(inputs) == 1:
            inp = inputs[0]
        else:
            flattened = [Flatten()(inp) if len(inp.Shape)>1 else inp for inp in inputs]
            inp = Concatenate()(*flattened)
        common1 = Dense(hidden, activation="relu", name="common1")(inp)
        common = Dense(max(hidden//2, num_controls*10), activation="relu", name="common")(common1)

        critic1 = Dense(max(hidden//5, 10), name="critic1", activation="relu")(common)
        value = Dense(1, name="critic")(critic1)

        action1 = Dense(max(hidden//5, num_controls*10), activation="relu", name="action1")(common)
        sigmas = Dense(num_controls, name="sigmas", activation="softplus")(action1)
        means = Dense(num_controls, name="means", activation="linear")(action1)
        
        probs = Concatenate()(means, sigmas)
        
        model = Model([inp], [probs, value])
        
        model["value"] = value
        model["means"] = means
        model["sigmas"] = sigmas
        model["probs"] = probs
        
        return model

    def policy(self, controls, training, valid_actions=None):
        # controls is tuple (means, sigmas)
        
        means, sigmas = controls
        
        if not training:
            # make it a bit more greedy
            sigmas = sigmas/2

        actions = np.random.normal(means, sigmas)
        #print("BrainContinuous.policy(): means:", means, "   sigmas:", sigmas, "   actions:", actions)
        return actions
        
    def evaluate_state(self, state):
        if not isinstance(state, list):
            state = [state]
        state = [s[None,...] for s in state]        # add minibatch dimension
        #print("evaluate_single: state shapes:", [s.shape for s in state])
        probs, values = self.Model.compute(state)
        #print("BrainContinuous.evaluate_state: probs, values:", type(probs), probs, type(values), values)
        means, sigmas = probs[:,:self.NControls], probs[:,self.NControls:]
        #print("BrainContinuous.evaluate_state: means, sigmas:", means, sigmas)
        return (means[0,:], sigmas[0,:]), values[0,0]
        
    def evaluate_states(self, states):
        #print("Brain.evaluate_many: states:", type(states), len(states))
        probs, values = self.Model.compute([states])
        means, sigmas = probs[:,:self.NControls], probs[:,self.NControls:]
        return (means, sigmas), values[:,0]
        
