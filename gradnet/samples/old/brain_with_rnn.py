from gradnet import Input, Model
from gardnet.layers Dense, LSTM, Concatenate

class EntropyLoss(Loss):
    
    def compute(self, data):
        probs = self.Inputs[0].Y                # [mb, t, actions]
        p = np.clip(probs, 1e-5, None)
        values = -np.sum(p*np.log(p), axis=-1)
        grads = (np.log(p)+1.0)
        self.Grads = [grads]
        self.Values = values        
        return self.value

class ActorLoss(Loss):
    
    def compute(self, data):
        mb, t = data.shape
        mb_t = mb*t
        returns = data["returns"]           # [mb, t]
        actions = data["actions"]           # [mb, t]
        probs, values = self.Inputs[0].Y, self.Inputs[1].Y          # values: [mb, t, 1]
        
        n_actions =probs.shape[-1]
        
        probs = probs.reshape((-1, n_actions))
        values = values.reshape((-1,))
        returns = returns.reshape((-1,))
        actions = actions.reshape((-1,))
        
        action_mask = np.eye(n_actions)[actions]
        action_probs = np.sum(action_mask*probs, axis=-1)
        advantages = returns - values
        #print("ActorLoss: rewards:", rewards.shape, "   values:", values.shape, "   probs:", probs.shape, "   advantages:", advantages.shape,
        #    "   action_probs:", action_probs.shape)
        loss_grads = -advantages/np.clip(action_probs, 1.e-5, None)  
        grads = action_mask * loss_grads[:,None]
        
        self.Grads = [grads.reshape((mb, t, n_actions)), None]
        self.Values = -advantages*np.log(np.clip(action_probs, 1.e-5, None)).reshape((mb, t))          # not really loss values

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
        mb, t, n_actions = probs.shape
        probs = probs.reshape((-1, n_actions))
        valid_mask = data.get("valids")
        if valid_mask is not None:
            valid_mask = valid_mask.reshape((-1, n_actions))
            self.Values = np.mean(probs*probs*(1-valid_mask), axis=-1).reshape((mb, t))
            self.Grads = [(2*(1-valid_mask)*probs).reshape((mb, t, n_actions))]
        else:
            self.Values = np.zeros((mb, t))
            self.Grads = [None]
        return self.value

class BrainWithRNN(Brain):
    
    def __init__(self, input_shape=None, num_actions=None, model=None, optimizer=None, hidden=128, gamma=0.99, 
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
            model = self.default_model(input_shape, num_actions, hidden, critic_weight, actor_weight, entropy_weight, invalid_action_weight)

        model.add_loss(ActorLoss(model["probs"], model["value"]), actor_weight, name="actor_loss")
        model.add_loss(get_loss("mse")(model["value"]), critic_weight, name="critic_loss")
        model.add_loss(EntropyLoss(model["probs"]), entropy_weight, name="entropy_loss")
        model.add_loss(InvalidActionLoss(model["probs"]), invalid_action_weight, name="invalid_action_loss")
        model.compile(optimizer=self.Optimizer)

        self.Model = model
        

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

        obs = Input((n_inputs))                   
        rnn = LSTM(hidden, return_sequences=True)(obs)
        dense = Dense(hidden, activation="relu")(obs)
        concatenated = Concatenate()(rnn, dense)
        probs = Dense(n_actions, activation="softmax")(concatenated)
        values = Dense(1)(concatenated)

        model = Model(obs, [probs, values])

        model["value"] = values
        model["probs"] = probs
    
        return model
        
    def evaluate(self, state):
        if isinstance(state, (list, tuple)):
            assert len(state) == 1
            state = state[0]
        probs, value = self.Model.call(state[None,:])
        probs = probs[0,:]
        value = value[0,0]
        return probs, value
    
