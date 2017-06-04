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
    # agent.load("./save/cartpole-master.h5")
    done = False
    batch_size = 32

    EPISODES = 1000

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, state_size+(1,))

        for _ in range(62):
            # print("round", _)
            # env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, state_size+(1,) )
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                break

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        eva = agent.evaluate(env)
        print("evaluate", eva)
        with open('tmp', 'a') as f:
            print(eva, file=f)
        if e % 10 == 0:
             agent.save("./model.h5")



