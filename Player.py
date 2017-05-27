import numpy as np
from Grid import Grid
import copy

class Player:
	def __init__(self, size):
		self.position = Grid(0, 0)
		self.num_moves = 0
		self.history = np.zeros((size, size))
		self.maze_size = size


	# return a numpy array of shape (4,)
	# up, down, left, right
	def get_strategy(self):
		# strategy = np.full((4,), 0.25)
		# strategy = np.array([0.5, 0, 0, 0.5])
		strategy = self.get_B_strategy()
		return strategy


	# the naive one
	def get_A_strategy(self):
		return np.array([0.25, 0.25, 0.25, 0.25])


	# 1. if up and right are both safe, return [0.5, 0, 0, 0.5]
	# 2. if one of up and right is safe, set it to 1
	# 3. if at the right most line, return [0.5, 0, 0.5, 0]
	def get_B_strategy(self):
		
		x, y = self.position.x, self.position.y
		size = self.maze_size

		# 1.
		if x + 1 < size and y + 1 < size:
			if self.history[x + 1][y] == 1 and self.history[x][y + 1] == 1:
				return np.array([0.5, 0, 0, 0.5])

		# 2. right is save
		if x + 1 < size:
			if self.history[x + 1][y] == 1:
				return np.array([0, 0, 0, 1])

		# 2. up is save
		if y + 1 < size:
			if self.history[x][y + 1] == 1:
				return np.array([1, 0, 0, 0])

		# 3. right most
		if x == size - 1:
			return np.array([0.5, 0, 0.5, 0])

		# 3. up most
		if y == size - 1:
			return np.array([0, 0.5, 0, 0.5])

		return np.array([0.5, 0, 0, 0.5])


	def get_next_move(self, strategy):
		
		assert strategy.shape == (4,)
		choice = np.random.choice(4, p=strategy)
		
		next_move = copy.deepcopy(self.position)

		if choice == 0:  # up
			next_move.y += 1
		if choice == 1:  # down
			next_move.y -= 1
		if choice == 2:  # left
			next_move.x -= 1
		if choice == 3:  # right
			next_move.x += 1

		self.num_moves += 1

		return next_move


	def change_position(self, position):
		self.position = position


	def update_history(self, obstacle_position):
		self.history[obstacle_position.x][obstacle_position.y] = 1