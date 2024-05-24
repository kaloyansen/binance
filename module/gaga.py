class gaga:

	def __init__(self, x, y, z = False): self.set(x, y, z)
	def notegal(self): return self.x != self.y
	def rate(self): return self.x / self.y
	def get(self): return y
	def append(self, xval, yval, zval = False):

		self.x.append(xval)
		self.y.append(yval)
		if self.z != False: self.z.append(zval)

	def clear(self):

		self.x.clear()
		self.y.clear()
		if self.z != False: self.z.clear()

	def set(self, x, y, z = False):

		self.x = x
		self.y = y
		self.z = z 
		
