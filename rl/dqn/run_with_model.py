import numpy as np
from maze import MazeEnv
from dqn import DQNAgent

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

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


if __name__ == "__main__":
    env = MazeEnv()
    state_size = (8,8)
    print("state_size", state_size)
    action_size = 282
    print("action_size", action_size)
    agent = DQNAgent(state_size, action_size)
    agent.load("model.h5")

    action_values = agent.model.predict( np.zeros((1,8,8,1)) )[0]
    for idx in range(282):
        print(index2action[idx])
        print('Q values', action_values[idx])

    exit(0)

    done = False
    batch_size = 32

    EPISODES = 1000

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, state_size+(1,))

        for _ in range(62):
            # print("round", _)
            # env.render()
            # action = agent.act(state)
            action = np.argmax( agent.model.predict( np.reshape(state, (1,8,8,1))) )
            print(action)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, state_size+(1,) )
            agent.remember(state, action, reward, next_state, done)
            state = next_state

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        eva = agent.evaluate(env)
        print("evaluate", eva)
        with open('tmp', 'a') as f:
            print(eva, file=f)
        if e % 10 == 0:
             agent.save("./model.h5")



