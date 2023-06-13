import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import random

# Actions

class SimpleBlackJackEnv(gym.Env):
    
    CardValues = [2,3,4,5,6,7,8,9,10,10,10,10,11]
    NCards = len(CardValues)
    CardVectors = np.array(list(np.eye(NCards)) + [np.zeros((NCards,))])        # CardVectors[-1] = all zeros
    NSuits = 4
    Ace = NCards-1
    NObservation = NCards*2


    STAY=0
    HIT=1
    DOUBLE=2

    NActions = 2
    
    def __init__(self):
        self.action_space = spaces.Discrete(self.NActions)
        low = np.zeros((self.NObservation))
        high = np.ones((self.NObservation,))*30
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        
    def deal(self, hand_value):
        card_index = self.Deck.pop()
        card_value = self.CardValues[card_index]
        
        if (card_value == 11) and (card_value + hand_value > 21):
            card_value = 1      # Ace can be converted to value = 1
        return card_index, float(hand_value + card_value)
        
    def reset(self):
        self.Deck = list(range(self.NCards))*self.NSuits
        random.shuffle(self.Deck)
        self.Double = 0
        self.Reward = 0
        self.Done = False
        self.DealerStays = False
        self.PlayerStays = False
        self.DealerCard, self.DealerValue = self.deal(0)
        self.PlayerCard, self.PlayerValue = self.deal(0)
        return self.observation(), {"valid_actions":np.array([0., 1.])}     # player can not stay after first card
        
    def observation(self):
        obs = np.zeros((self.NObservation,))
        if self.PlayerCard is not None:
            obs[:self.NCards] = self.CardVectors[self.PlayerCard]
        if self.DealerCard is not None:
            obs[self.NCards:self.NCards*2] = self.CardVectors[self.DealerCard]
        #obs[self.NCards*2] = self.Double
        #obs[self.NCards*2] = self.PlayerValue/21
        #obs[self.NCards*2+1] = self.DealerValue/21
        return obs
        
    def step(self, action):
        self.Action = action

        #
        # Dealer action
        #
        reward = 0.0
        done = False
        self.DealerCard = self.PlayerCard = None
        
        if action == self.DOUBLE:
            self.Double = 1
        else:
            if not self.PlayerStays:
                #
                # Player action
                #
                if action == 0:
                    self.PlayerStays = True
                else:
                    self.PlayerCard, self.PlayerValue = self.deal(self.PlayerValue)
                    if self.PlayerValue > 21:
                        reward = -1.0
                        done = True

            if not done:
                if self.DealerValue >= 17:
                # dealer stays
                    self.DealerStays = True

                if not done and not self.DealerStays:
                    self.DealerCard, self.DealerValue = self.deal(self.DealerValue)
                    if self.DealerValue > 21:
                        done = True
                        reward = 1.0
        
            if not done:
                if self.DealerStays and self.PlayerStays:
                    done = True
                    if self.PlayerValue == self.DealerValue:
                        reward = 0.0
                    elif self.PlayerValue > self.DealerValue:
                        reward = 1.0
                    else:
                        reward = -1.0
                    
            # check if anyone has a blackjack
            if not done:
                player_blackjack = self.PlayerValue == 21
                dealer_blackjack = self.DealerValue == 21
                if player_blackjack or dealer_blackjack:
                    done = True
                    if player_blackjack:
                        reward = 1.0
                    else:
                        reward = -1.0
        
        self.Reward = reward
        self.Done = done
        valid = [1.0, float(not self.PlayerStays)]
        return self.observation(), reward*(1+self.Double), done, {"valid_actions":np.array(valid)}
        
    def render(self):
        pass
        
if __name__ == "__main__":
    
    env = SimpleBlackJackEnv()
    
    state, meta = env.reset()
    print(env.PlayerCard, env.PlayerValue, env.DealerCard, env.DealerValue, meta)
    valid = meta["valid_actions"]
    #env.render()
    done = False
    while not done:
        while True:
            a = 0 if random.random() < 0.5 else 1
            if valid[a]:
                break
        state, reward, done, meta = env.step(a)
        print(a, env.PlayerCard, env.PlayerValue, env.DealerCard, env.DealerValue, reward, done, meta)
        valid = meta["valid_actions"]
        
        
        
    
    
