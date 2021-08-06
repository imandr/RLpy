#
# Tic Tac Toe
#

import numpy as np
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

WinMasks = np.array(WinMasks).reshape((-1,9))

class SingleAgentTicTacToeEnv(object):
    
    NActions = 9
    ObservationShape = (9,)
    
    def __init__(self):
        self.Board = np.zeros((9,))
        self.action_space = spaces.Discrete(self.NActions)
        high = np.ones((self.NActions,))
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        
    def reset(self):
        self.Done = False
        self.Board[...] = 0.0
        self.BoardHistory = []
        self.Side = 1
        self.FirstMove = True
        return self.observation(self.Side), {"valid_actions":np.array([1,1,0,0,1,0,0,0,0], dtype=np.float32)}
        
    def observation(self, side):
        return self.Board * side
        
    def step(self, action):
        win = False
        draw = False
        side = self.Side
        other_side = -side
        color = side
        
        reward = 0.0
        done = False

        if self.Board[action] != 0:
            # invalid move
            reward = -1.0
            done = True
        else:
            self.Board[action] = side
            self.BoardHistory.append(self.Board.reshape((3,3)).copy())
    
            for win_mask in WinMasks:
                masked = self.Board*color*win_mask
                if np.sum(masked) == 3:
                    reward = 1.0
                    done = True
                    break
                    
            if np.all(self.Board != 0):
                done = True     # draw
        self.Side = other_side
        self.Done = done
        self.Reward = reward
        return self.observation(self.Side), reward, done, {"valid_actions":np.asarray(self.Board==0, dtype=np.float32)}
            
    def render(self):
        if self.Done:
            last_move = -self.Side
            history = self.BoardHistory
            sep = "+---"*len(history) + "+"
            lines = [sep]
            for irow in (0,1,2):
                line = "|"
                for b in history:
                    row = "".join(".xo"[int(c)] for c in b[irow])
                    line += row + "|"
                lines.append(line)
            outcome = "draw"
            if self.Reward:
                outcome = "%s won" % (".xo"[int(last_move)])
            lines.append(sep + " " + outcome)
            print("\n".join(lines))
        
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
    
    
    
    