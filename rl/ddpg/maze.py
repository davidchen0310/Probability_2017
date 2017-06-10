import numpy as np
import os, subprocess, time, signal
import sys
import copy
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding

dx = [1,0,-1,0]
dy = [0,1,0,-1]

class MazeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.is_simulate = False
        self.X, self.Y = 8, 8
        self.maze = np.zeros((8,8), dtype=int)

        self.round = 0
        self.sequences = np.arange(1, 63)
        np.random.shuffle(self.sequences)
        assert len(self.sequences) == 62

    def _print_maze(self):
        print(self.maze)

    def _step(self, prob):

        # update state
        obstacle = (self.sequences[self.round]//8, self.sequences[self.round]%8)
        assert self.maze[ obstacle[0] ][ obstacle[1] ] == 0
        self.maze[ obstacle[0] ][ obstacle[1] ] = 1
        self.round += 1

        #reward =  1e9 - self._run_step(obstacle, prob)
        reward = -self._run_step(obstacle, prob)

        done = self.round == 62
        return self.getState(), reward, done, {}


    # given the obstacle position and chosen probability
    # return the expected steps needed for this round
    def _run_step(self, obstacle, prob):

        if min(prob) < 0 or max(prob) > 1:
            return 1e9

        ''' Run by real simulation '''
        if self.is_simulate:
            res = []
            ITER = 100
            for _ in range(ITER):
                pos = (0,0)
                steps = 0
                while pos != (7,7):
                    move = np.random.choice(np.arange(4), p=prob)
                    nx, ny = pos[0]+dx[move], pos[1]+dy[move]
                    if not self._out_of_bound(nx, ny) and (nx,ny)!=obstacle:
                        pos = (nx, ny)
                    steps += 1
                    if steps >= 10000: break
                res.append(steps)
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
                ret = x[0,0]
                if ret<=0: return 1e9
                return min(1e9, ret)
            except np.linalg.linalg.LinAlgError as e:
                return 1e9


    def _out_of_bound(self, x, y):
        return x < 0 or y < 0 or x >= self.X or y >= self.Y

    def getState(self):
        return copy.deepcopy(np.reshape(self.maze, (8,8)))


    def _reset(self):
        self.sequences = np.arange(1, 63)
        np.random.shuffle(self.sequences)
        assert len(self.sequences) == 62
        self.round = 0
        self.maze = np.zeros((8,8), dtype=int)
        return self.getState()

    def _render(self, mode='human', close=False):
        return None

if __name__=="__main__":
    env = MazeEnv()
    print(env._run_step((2,1),(0.45, 0.45, 0.05, 0.05)))
