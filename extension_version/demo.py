import numpy as np
from Maze import Maze
from Player import Player
from utils import stuck


maze_size = 8
num_rounds = 62
num_trials = 10000


def main():
	num_moves = np.empty(num_trials)
	for k in range(num_trials):
		maze = Maze(maze_size)
		player = Player(maze_size)
		for i in range(num_rounds):
			strategy = player.get_strategy()
			if stuck(maze, player.position, strategy):
				print("stuck at", player.position, ",obstacle at", maze.get_obstacle_position())
				print("strategy:", strategy)
				print("total moves:", player.num_moves)
				exit()
			while True:
				next_move = player.get_next_move(strategy)
				# print(next_move)
				if maze.feasible(next_move):
					player.change_position(next_move)
					player.update_history(maze.get_obstacle_position())
					maze.nextRound()
					break
			# print(player.num_moves)
			# maze.print(player)
			if maze.reach_end(player):
				break
		num_moves[k] = player.num_moves
	print(num_moves.mean())

if __name__ == "__main__":
	main()