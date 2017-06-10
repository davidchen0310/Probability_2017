import numpy as np
from maze import MazeEnv
import random


def sum_sample(n=4, total=1000):
    """Return a randomly chosen list of n positive integers summing to total.
    Each such list is equally likely to occur."""
    while True:
        dividers = sorted(random.sample(range(1, total), n - 1))
        ret = [a - b for a, b in zip(dividers + [total], [0] + dividers)]
        if max(ret) < total:
            return np.array(list(ret))/total*0.2

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

if __name__ == "__main__":
    env = MazeEnv()
    state_size = (8,8)
    print("state_size", state_size)
    action_size = 282
    print("action_size", action_size)
    done = False
    EPISODES = 1000
    strategy = 2

    res = []
    for e in range(EPISODES):
        state = env.reset() #state = np.reshape(state, state_size+(1,))

        reward_sum = 0
        for _ in range(62):
            # print("round", _)
            # env.render()

            state = np.reshape(state, (8,8))
            expected_reward = []

            trials = 100
            test_prob = []
            for action in range(trials):
                prob = np.array(list())/10
                if strategy == 1:
                    prob = sum_sample()
                    prob[0] += 0.4
                    prob[1] += 0.4

                    test_prob.append(prob)
                    tmp = []
                    for idx in range(1, 63):
                        i,j = idx//8, idx%8
                        if state[i][j] == 0:
                            tmp.append( env._run_step((i,j), prob) )
                    expected_reward.append( np.array(tmp).mean() )

                    chosen_action = test_prob[np.argmin(expected_reward)]
                    print("chosen_action", chosen_action)

                elif strategy == 2:
                    chosen_action = [0.4, 0.4, 0.1, 0.1]

            # Here, chosen_action is the probability
            next_state, reward, done, _ = env.step(chosen_action)
            reward_sum += reward
            state = next_state

        res.append(reward_sum)
        print('episode', e)
        print("current mean reward", np.array(res).mean())
    print(np.array(res).mean())
