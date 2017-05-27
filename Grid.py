class Grid:
	def __init__(self, x, y):
		self.x = x
		self.y = y


	def __eq__(self, other):
		if isinstance(other, self.__class__):
			return self.__dict__ == other.__dict__
		else:
			return False


	def __ne__(self, other):
		return not self.__eq__(other)


	def __str__(self):
		return ("(" + str(self.x) + ", " + str(self.y) + ")")