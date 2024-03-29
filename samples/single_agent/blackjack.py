import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import random

# Actions

class SimpleBlackJackEnv(gym.Env):
    
    CardValues = [2,3,4,5,6,7,8,9,10,10,10,10,11]
    NSuits = 4
    NCards = len(CardValues)
    DeckSize = NSuits * NCards
    CardVectors = np.array(list(np.eye(DeckSize)) + [np.zeros((DeckSize,))])        # CardVectors[-1] = all zeros
    Ace = NCards-1
    NObservation = DeckSize

    STAY=0
    HIT=1
    DOUBLE=2        # not used yet

    NActions = 2
    
    def __init__(self):
        self.action_space = spaces.Discrete(self.NActions)
        low = -np.ones((self.NObservation,))*30
        high = np.ones((self.NObservation,))*30
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        
    def deal(self, hand_value):
        card_index = self.Deck.pop()
        card_value = self.CardValues[card_index % len(self.CardValues)]
        
        if (card_value == 11) and (card_value + hand_value > 21):
            card_value = 1      # Ace can be converted to value = 1
        return card_index, float(hand_value + card_value)
        
    def reset(self):
        self.Deck = list(range(self.NCards*self.NSuits))
        self.DeckPlayed = np.zeros((self.DeckSize,), dtype=np.float)
        random.shuffle(self.Deck)
        self.Double = 0
        self.Reward = 0
        self.Done = False
        self.PlayerStays = False
        self.DealerCard, self.DealerValue = self.deal(0)
        self.PlayerCard, self.PlayerValue = self.deal(0)
        self.DeckPlayed[self.DealerCard] = -1.0
        self.DeckPlayed[self.PlayerCard] = 1.0
        #print("Env.reset: player:", self.PlayerCard, "  dealer:", self.DealerCard)
        return self.observation(), {"values":np.array([self.PlayerValue, self.DealerValue])}
        
    def dealer_stays(self):
        return self.DealerValue >= 17
        
    def observation(self):
        return self.DeckPlayed.copy()
        
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
            self.PlayerStays = self.PlayerStays or action == self.STAY
            if not self.PlayerStays:
                self.PlayerCard, self.PlayerValue = self.deal(self.PlayerValue)
                if self.PlayerValue > 21:
                    reward = -10.0
                    done = True

            if not done:

                if not done and not self.dealer_stays():
                    self.DealerCard, self.DealerValue = self.deal(self.DealerValue)
                    if self.DealerValue > 21:
                        done = True
                        reward = 10.0
        
            if not done:
                if self.dealer_stays() and self.PlayerStays:
                    done = True
                    if self.PlayerValue == self.DealerValue:
                        reward = 0.0
                    elif self.PlayerValue > self.DealerValue:
                        reward = 10.0
                    else:
                        reward = -10.0
                    
            # check if anyone has a blackjack
            if not done:
                player_blackjack = self.PlayerValue == 21
                dealer_blackjack = self.DealerValue == 21
                if player_blackjack or dealer_blackjack:
                    done = True
                    if player_blackjack:
                        reward = 10.0
                    else:
                        reward = -10.0
        if self.DealerCard is not None: self.DeckPlayed[self.DealerCard] = -1.0
        if self.PlayerCard is not None: self.DeckPlayed[self.PlayerCard] = 1.0
        self.Reward = reward
        self.Done = done
        return self.observation(), reward*(1+self.Double), done, {
                "values":np.array([self.PlayerValue, self.DealerValue]),
                "cards":[self.PlayerCard, self.DealerCard]
            }
        
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
        
        
        
    
    
