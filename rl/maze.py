import numpy as np
import os, subprocess, time, signal
import sys
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding

action2index = {}
index2action = []
cnt = 0
for i in range(10):
    for j in range(10):
        for k in range(10):
            for l in range(10):
                if i+j+k+l == 10:
                    action2index[ (i,j,k,l) ] = cnt
                    index2action.append( [i,j,k,l] )
                    cnt += 1

assert len(action2index) == len(index2action)

label_to_action = {'up': 0, 'down': 1, 'left': 2, 'right': 3}
action_to_label = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}

class MazeEnv(gym.Env, utils.EzPickle):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.is_simulate = True
        self.X, self.Y = 8, 8
        # self.observation_space = spaces.Box(0,2,shape=(8,8))
        # self.action_space = spaces.Discrete(4)
        # self.reward_range = (-1e9, -1)
        self.maze = np.zeros((8,8), dtype=int)

        self.round = 0
        self.sequences = np.arange(1, 63)
        np.random.shuffle(self.sequences)
        assert len(self.sequences) == 62
        # self._print_maze()

    def _print_maze(self):
        print(self.maze)

    def _step(self, action):
        prob = index2action[action]

        reward = self._run_step(prob)*-1.
        # print("reward", reward)
        # self._print_maze()
        ob = self.getState()
        episode_over = self.round == 62
        return ob, reward, episode_over, {}


    def _run_step(self, prob):
        prob = np.array(list(prob))/10
        dx = [1,0,-1,0]
        dy = [0,1,0,-1]
        # update state
        obstacle = (self.sequences[self.round]//8, self.sequences[self.round]%8)
        self.maze[ obstacle[0] ][ obstacle[1] ] = 1
        self.round += 1


        ''' Run by real simulation '''
        if self.is_simulate:
            res = []
            ITER = 100
            for _ in range(ITER):
                pos = (0,0)
                rounds = 0
                while pos != (7,7):
                    move = np.random.choice(np.arange(4), p=prob)
                    nx, ny = pos[0]+dx[move], pos[1]+dy[move]
                    if not self._out_of_bound(nx, ny) and (nx,ny)!=obstacle:
                        pos = (nx, ny)
                    rounds += 1
                    if rounds >= 10000: break
                res.append(rounds)
            return np.array(res).mean()
        else:
            ''' Calculate with linalg '''
            sz = self.X
            A = np.eye(sz*sz)
            b = np.ones(sz*sz)
            for i in range(sz):
                for j in range(sz):

                    idx = i*sz + j
                    if idx == sz*sz-1:
                        b[-1] = 0
                        continue

                    for k in range(4):
                        ni, nj = i+dx[k],j+dy[k]
                        if self._out_of_bound(ni, nj) or (ni,nj) == obstacle:
                            A[idx][idx] -= prob[k]
                        else:
                            A[idx][ni*sz+nj] -= prob[k]

            try:
                x = np.linalg.solve(np.mat(A), np.mat(b).T)
                ret = x[0][0][0][0]
                if ret<=0: return 1e9
                return min(1e9, ret)
            except np.linalg.linalg.LinAlgError as e:
                return 1e9


    def _out_of_bound(self, x, y):
        return x < 0 or y < 0 or x >= self.X or y >= self.Y

    def getState(self):
        return self.maze

    def _get_reward(self):
        """ Reward is given for scoring a goal. """
        return -1

    def _reset(self):
        self.sequences = np.arange(1, 63)
        np.random.shuffle(self.sequences)
        assert len(self.sequences) == 62
        self.round = 0
        self.maze = np.zeros((8,8), dtype=int)
        # self._print_maze()
        return self.maze

    def _render(self, mode='human', close=False):
        pass

if __name__=="__main__":
    env = MazeEnv()
    print(env._run_step((2, 3, 3, 2)))
