class Gaga:

	def __init__(self, x, y, z = False): self.set(x, y, z)
	def notegal(self): return self.x != self.y
	def rate(self): return self.x / self.y
	def set(self, x, y, z = False):

		self.x = x
		self.y = y
		if z != False: self.z = z 
		
