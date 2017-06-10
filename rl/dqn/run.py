import copy
import numpy as np
from maze import MazeEnv
from dqn import DQNAgent

if __name__ == "__main__":
    env = MazeEnv()
    state_size = (8,8)
    print("state_size", state_size)
    action_size = 282
    print("action_size", action_size)
    agent = DQNAgent(state_size, action_size)
    #agent.load("model.h5")
    done = False
    batch_size = 32

    EPISODES = 1000

    episode_reward = []

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, state_size+(1,))

        reward_sum = 0
        for _ in range(62):
            # print("round", _)
            # env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward_sum += reward
            next_state = np.reshape(next_state, state_size+(1,) )
            #print(np.reshape(state, (8,8)))
            agent.remember(state, action, reward, next_state, done)
            state = next_state

        print(reward_sum)
        episode_reward.append(reward_sum)

        if len(agent.memory) > batch_size:
            print(agent.epsilon)
            agent.replay(batch_size)
            eva = agent.evaluate(env)
            print("evaluate", eva)
            agent.save("./model.h5")

    from matplotlib import pyplot as plt

    plt.plot(episode_reward)

    plt.ylabel('episode_reward')
    plt.xlabel('epoch')
    plt.savefig('learning_curve')

