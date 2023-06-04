import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import random


class Game21Env(gym.Env):
    
    CardValues = list(range(1,7))
    NSuits = 4
    NCards = len(CardValues)
    CardVectors = np.eye(NCards)
    NObservation = NCards
    
    def __init__(self):
        self.action_space = spaces.Discrete(2)
        low = np.zeros((self.NObservation,))
        high = np.ones((self.NObservation,))
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        
    def deal(self, hand_value):
        card_index = self.Deck.pop()
        card_value = self.CardValues[card_index]       
        self.PlayerCards.append(card_value) 
        return card_index, float(hand_value + card_value)

    def reset(self):
        self.PlayerCards = []
        self.Deck = list(range(self.NCards))*self.NSuits
        random.shuffle(self.Deck)
        self.Reward = 0
        self.Done = False
        self.PlayerCard, self.PlayerValue = self.deal(0)
        self.Action = None
        return self.observation()

    def observation(self):
        obs = np.zeros((self.NObservation,))
        obs[:self.NCards] = self.CardVectors[self.PlayerCard]
        return obs

    def step(self, action):
        self.Action = action
        reward = 0.0
        done = False
        if action == 0:
            reward = self.PlayerValue
            done = True
        else:
            self.PlayerCard, self.PlayerValue = self.deal(self.PlayerValue)
            if self.PlayerValue > 21:
                reward = -1.0
                done = True

        self.Reward = reward
        self.Done = done
        return self.observation(), reward, done, {}
        
    def render(self):
        if self.Action is None:
            print("initial hand:     ", self.PlayerCards, "=", self.PlayerValue)
        else:
            print("action :", self.Action, "hand ->", self.PlayerCards, "=", self.PlayerValue, " ---done---" if self.Done else "")
        
