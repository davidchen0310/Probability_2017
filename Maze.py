import numpy as np
from Grid import Grid
from utils import grid2index, index2grid


class Maze:
	def __init__(self, size):
		self.obstacle_sign = "*"
		self.player_sign = "."
		self.obstacle_not_appeared_sign = "?"  # the obstacle has not appeared at this grid
		self.obstacle_appeared_sign = "X"  # the obstacle has appeared at this gird

		self.cur_round = 0
		self.size = size
		self.end = Grid(size - 1, size - 1)

		self.obstacle_occurences = np.arange(size * size)
		
		start_index = np.argwhere(self.obstacle_occurences == size * (size - 1))
		self.obstacle_occurences = np.delete(self.obstacle_occurences, start_index)
		end_index = np.argwhere(self.obstacle_occurences == size - 1)
		self.obstacle_occurences = np.delete(self.obstacle_occurences, end_index)
		
		np.random.shuffle(self.obstacle_occurences)


	def get_obstacle_position(self):
		index =  self.obstacle_occurences[self.cur_round]
		return index2grid(index, self.size)


	def has_obstacle(self, grid):
		return (grid == self.get_obstacle_position())


	def out_of_maze(self, grid):
		if grid.x < 0 or grid.x >= self.size or grid.y < 0 or grid.y >= self.size:
			return True
		else:
			return False


	def feasible(self, grid):
		# print(grid)
		if self.out_of_maze(grid) or self.has_obstacle(grid):
			return False
		else:
			return True


	def nextRound(self):
		self.cur_round += 1


	def reach_end(self, player):
		return (self.end == player.position)


	def print(self, player):
		chars = np.empty((self.size * self.size), dtype=str)
		
		chars[:] = self.obstacle_not_appeared_sign
		chars[self.obstacle_occurences[:self.cur_round]] = self.obstacle_appeared_sign

		grid = self.get_obstacle_position()
		chars[grid2index(grid, self.size)] = self.obstacle_sign
		
		grid = player.position
		chars[grid2index(grid, self.size)] = self.player_sign

		for i in range(self.size):
			for j in range(self.size):
				print(chars[i * self.size + j], end='')
			print()