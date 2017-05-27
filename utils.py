import math
import copy
from Grid import Grid

# transform a integer to a grid based on given maze size
def index2grid(index, size):
	x = index % size
	y = size - math.floor(index / size) - 1
	grid = Grid(x, y)
	return grid


def grid2index(grid, size):
	index = (size - grid.y - 1) * size + grid.x
	# print(grid, " ", index)
	return index


def stuck(maze, position, strategy):
	for i, s in enumerate(strategy):
		if s > 0:
			temp = copy.deepcopy(position)
			if i == 0:  # up
				temp.y += 1
			if i == 1:  # down
				temp.y -= 1
			if i == 2:  # left
				temp.x -= 1
			if i == 3:  # right
				temp.x += 1
			if maze.feasible(temp):
				return False
	return True