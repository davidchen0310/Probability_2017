# -*- coding: utf-8 -*-
import os
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense, Dropout, Activation, Flatten

from maze import MazeEnv

import tensorflow as tf

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
print("action_space", len(index2action))

def sum_sample(n=4, total=10):
    """Return a randomly chosen list of n positive integers summing to total.
    Each such list is equally likely to occur."""
    while True:
        dividers = sorted(random.sample(range(1, total), n - 1))
        ret = [a - b for a, b in zip(dividers + [total], [0] + dividers)]
        if max(ret) < total:
            return ret



class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.1
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        input_dim = self.state_size + (1,)
        model = Sequential()

        model.add(Conv2D(64, (2,2), input_shape=input_dim))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(282, activation='linear'))

        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        print(model.summary())
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            ret = random.randrange(self.action_size)
            return ret
        state = np.reshape(state, (1,8,8,1))
        act_values = self.model.predict(state)
        return np.argmax(act_values[0]) # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:

            state = np.reshape(state, (1,8,8,1))
            next_state = np.reshape(next_state, (1,8,8,1))

            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def evaluate(self, env):

        EPISODES = 1
        rew = []
        for e in range(EPISODES):
            state = env.reset()

            reward_sum = 0
            for _ in range(62):
                # print("Evaluating, round", _)

                #print(np.reshape(state, (8,8)))
                state = np.reshape(state, (1,) + self.state_size+(1,))
                act_values = self.model.predict(state)
                action =  np.argmax(act_values[0]) # returns action

                next_state, reward, done, _ = env.step(action)
                state = next_state

                reward_sum += reward

            print('reward_sum', reward_sum)
            rew.append(reward_sum)

        return np.array(rew).mean()

