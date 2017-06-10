import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.layers.core import Reshape
from keras.callbacks import Callback

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy, GreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor

from maze import MazeEnv
ENV_NAME = 'myMaze'


class History(Callback):
    def on_train_begin(self,logs={}):
        self.performance = []

    def on_epoch_end(self,epoch,logs={}):
        #dqn.test(env, nb_episodes=2, visualize=False)
        pass

class Processor(Processor):
    def process_observation(self, ob):
        ob = np.reshape(ob, (8,8))
        return ob

# np.random.seed(123)

# Get the environment and extract the number of actions.
env = MazeEnv()# gym.make(ENV_NAME)
# env.seed(123)
# nb_actions = env.action_space.n
nb_actions = 282

# Next, we build a very simple model.
model = Sequential()
model.add(Reshape((8,8,1), input_shape=(1,8,8,1)))
model.add(Conv2D(64, (2,2)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(nb_actions, activation='linear'))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=1)
#policy = BoltzmannQPolicy()
policy = GreedyQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100, target_model_update=1e-2, policy=policy)

dqn.compile(Adam(lr=1e-4), metrics=['mae'])

#dqn.load_weights('dqn_{}_weights.h5f'.format(ENV_NAME))
#dqn.test(env, nb_episodes=10, visualize=True)

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.

#history = dqn.fit(env, nb_steps=500000000000000, visualize=False, verbose=2, callbacks=[History()])
history = dqn.fit(env, nb_steps=500000000000000, visualize=False, verbose=2 )

import matplotlib.pyplot as plt
print(history.history)
# summarize history for loss
tmp = history.history['episode_reward']
np.save('history', np.array(tmp))

plt.plot(history.history['episode_reward'])
plt.ylabel('episode_reward')
plt.xlabel('epoch')
plt.savefig('learning_curve')

# After training is done, we save the final weights.
dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=10, visualize=True)
