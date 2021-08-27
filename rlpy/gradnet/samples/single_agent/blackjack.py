import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import random


class SimpleBlackJackEnv(gym.Env):
    
    Values = [2.,3.,4.,5.,6.,7.,8.,9.,10.,10.,10.,10.,11.]
    NCards = len(Values)
    NSuits = 4
    
    def __init__(self):
        self.action_space = spaces.Discrete(2)
        low = np.zeros((6,))
        high = np.ones((6,))*30
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        
    def deal(self, value):
        card = float(self.Deck.pop()[0])
        if card + value > 21 and card == 11:
            card = 1
        return card, value + card
        
    def reset(self):
        self.Deck = [(v, s) for s in range(self.NSuits) for v in self.Values]
        repeat = True
        while repeat:
            # make sure noone gets 21 right away
            random.shuffle(self.Deck)

            card1,self.PlayerValue = self.deal(0)
            card2,self.PlayerValue = self.deal(self.PlayerValue)
            if self.PlayerValue >= 21:  continue
            self.PlayerHand = (card1, card2)
            
            card1,self.DealerValue = self.deal(0)
            card2,self.DealerValue = self.deal(self.DealerValue)
            if self.DealerValue >= 21:  continue
            self.DealerHand = (card1, card2)

            repeat = False

        self.Reward = 0
        self.Done = False
        self.DealerStays = False
        self.PlayerStays = False
        return self.observation()
        
    def observation(self):
        obs = np.empty((6,))
        dealer_hand = self.DealerHand or (0,0)
        obs[0] = dealer_hand[0]
        obs[1] = dealer_hand[1]
        obs[2] = self.DealerValue
        obs[3] = self.PlayerHand[0]
        obs[4] = self.PlayerHand[1]
        obs[5] = self.PlayerValue
        return obs/21.0
        
    def step(self, action):
        self.Action = action
        
        #
        # Player action
        #
        
        self.DealerHand = None
        if self.PlayerStays and action != 0:
            return self.observation(), -10.0, True, {}
        
        if action == 0:
            # stay
            done = self.DealerStays
            self.PlayerHand = (0, 0)
            self.PlayerStays = True
        else:
            # hit
            card, self.PlayerValue = self.deal(self.PlayerValue)
            self.PlayerHand = (card, 0)
            done = self.PlayerValue >= 21

        if not done:
            #
            # dealer action
            #
        
            dealer_hit = self.DealerValue < 17
            if dealer_hit:
                card, self.DealerValue = self.deal(self.DealerValue)
                self.DealerHand = (card, 0)
                if self.DealerValue >= 21:
                    done = True
            else:
                self.DealerStays = True
                self.DealerHand = (0, 0)
                done = self.PlayerStays

            done = done or (action == 0 and self.DealerStays)


        reward = 0.0
        if done:
            if self.PlayerValue > 21:
                reward = -1.0
            elif self.DealerValue > 21:
                reward = 1.0
            elif self.PlayerValue == self.DealerValue:
                reward = 0.0
            elif self.PlayerValue < self.DealerValue:
                reward = -1.0
            else:
                reward = 1.0
        
        self.Reward = reward
        self.Done = done
        return self.observation(), reward, done, {}
        
    def render(self):
        c1, c2 = self.PlayerHand
        print("Player:", c1 or "", c2 or "", "->", self.PlayerValue)
        if self.DealerHand is not None:
            c1, c2 = self.DealerHand
            print("Dealer:", c1 or "", c2 or "", "->", self.DealerValue)
        if self.Done:
            print("--- done. reward:", self.Reward)
        
if __name__ == "__main__":
    
    env = SimpleBlackJackEnv()
    
    state = env.reset()
    env.render()
    done = False
    while not done:
        a = 0 if random.random() < 0.5 else 1
        state, reward, done, meta = env.step(a)
        env.render()
        
        
        
    
    
