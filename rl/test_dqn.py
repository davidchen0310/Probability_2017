from dqn import DQNAgent
from maze import MazeEnv

state_size = (8,8)
action_size  = 282

agent = DQNAgent(state_size, action_size)
agent.load('model.h5')

env = MazeEnv()
print( agent.evaluate(env) )
