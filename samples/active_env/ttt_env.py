#
# Tic Tac Toe
#

import numpy as np
from rlpy import ActiveEnvironment
from gym import spaces

WinMasks = [
    [
        [1,0,0],
        [1,0,0],
        [1,0,0],
    ],
    [
        [0,1,0],
        [0,1,0],
        [0,1,0],
    ],
    [
        [0,0,1],
        [0,0,1],
        [0,0,1],
    ],
    
    [
        [1,1,1],
        [0,0,0],
        [0,0,0],
    ],
    [
        [0,0,0],
        [1,1,1],
        [0,0,0],
    ],
    [
        [0,0,0],
        [0,0,0],
        [1,1,1],
    ],

    [
        [1,0,0],
        [0,1,0],
        [0,0,1],
    ],
    [
        [0,0,1],
        [0,1,0],
        [1,0,0],
    ]
]

WinMasks = np.array(WinMasks)

class TicTacToeEnv(ActiveEnvironment):
    
    NActions = 9
    ObservationShape = (9,)
    NState = 9
    
    Symbols = "XO"
    
    def __init__(self):
        ActiveEnvironment.__init__(self, name="Tic-Tac-Toe", 
                action_space=spaces.Discrete(self.NActions), 
                observation_space=spaces.Box(-np.ones((self.NState,)), np.ones((self.NState,)), dtype=np.float32))
        self.Board = np.zeros((3,3))
        self.Agents = []
        
    def reset(self, agents, training=True):
        self.Training = training
        self.Board[...] = 0.0
        self.BoardHistory = []
        self.Agents = agents
        self.Side = 0
        for a in agents:    
            #print(a)
            a.reset(training=training)
        assert len(agents) == 2
        self.FirstMove = True
        self.Done = False
        self.Win = None
        
    def observation(self, iagent):
        color = iagent*2 - 1     # +1 or -1
        return (self.Board.reshape((-1,)) * color).copy()

    def turn(self):
        win = False
        draw = False
        side = self.Side
        other_side = 1-side
        color = side*2 - 1     # +1 or -1
        obs = self.observation(side)
        available = np.array([1,1,0,0,1,0,0,0,0], dtype=np.float32) if self.FirstMove else np.asarray(obs == 0, dtype=np.float32)        # mask
        
        if not np.any(available):
            # draw
            self.Agents[side].done(self.observation(side))
            self.Agents[other_side].done(self.observation(other_side))
            self.Done = True
            return True
        
        action = self.Agents[side].action(obs, available)

        x = action//3
        y = action%3
        assert self.Board[x,y] == 0.0
        self.Board[x,y] = color
        self.BoardHistory.append(self.Board.copy())
    
        for win_mask in WinMasks:
            masked = self.Board*color*win_mask
            if np.sum(masked) == np.sum(win_mask):
                self.Agents[other_side].done(self.observation(other_side), -1.0)
                self.Agents[side].done(self.observation(side), 1.0)
                self.Done = True
                self.Win = side
                return True
            
        self.Side = other_side
        self.FirstMove = False
        return False
            
    def show_history(self, history=None):
        if history is None: history = self.BoardHistory
        sep = "+---"*len(history) + "+"
        lines = [sep]
        for irow in (0,1,2):
            line = "|"
            for b in history:
                row = "".join(".ox"[int(c)] for c in b[irow])
                line += row + "|"
            lines.append(line)
        outcome = "draw"
        if self.Win is not None:
            outcome = self.Symbols[self.Win] + " won"
        lines.append(sep+" "+outcome)
        return "\n".join(lines)
        
    def render(self):
        if self.Done:
            print(self.show_history())
        
        
if __name__ == "__main__":
    
    import random
    
    def show_board(board):
        sep = "+---"*3 + "+"
        out = [sep]
        for row in board.reshape((3,3)):
            line = "| "
            for x in row:
                line += " OX"[int(x)] + " | "
            out.append(line)
            out.append(sep)
        return "\n".join(out)
    
    class Agent(object):
        
        def __init__(self, side):
            self.Side = side
            self.Sign = "XO"[side]
            self.Color = side*2-1
            
        def reset(self):
            pass
        
        def action(self, reward, observation, available_actions):
            print(f"{self.Sign}: action:", reward, observation, available_actions)
            choices = [i for i, x in enumerate(available_actions) if x]
            i = random.choice(choices)
            return i
        
        def reward(self, r):
            #print(f"{self.Sign}: reward: {r}")
            pass
            
        def done(self, r, last_observation):
            if r > 0:
                print(f"===== {self.Sign} won")
            elif r < 0:
                print(f"===== {self.Sign} lost")
            else:
                print("===== draw")
            
    class Callback(object):
        
        def end_turn(self, agents, data):
            print(show_board(data["board"]))
            
        def end_episode(self, agents, data):
            print("--- game over ---")
            print(env.show_history(data["board_history"]))
        
    x_agent = Agent(0)
    y_agent = Agent(1)
    
    env = TicTacToeEnv()
    env.run([x_agent, y_agent], [Callback])
    
    
    
    