from module.Gaga import Gaga

class Progress:

	def __init__(self,
				 progress_bar_size = 55,
				 marker_past       = '=',
				 marker_future     = ':',
				 marker_now        = '>'):

		self.set(progress_bar_size, marker_past, marker_future, marker_now)
				 
	def set(self, progress_bar_size, marker_past, marker_future, marker_now):

		self.progress_bar_size = progress_bar_size
		self.marker_past = marker_past
		self.marker_future = marker_future
		self.marker_now = marker_now

	def go(self, progress, total, label = ''):

		scale = self.progress_bar_size / 100	
		partition = progress / total
		percent = partition * 100
		factor = int(percent * scale)
		bar_size = Gaga(factor, self.progress_bar_size - factor)
		bar =\
		self.marker_past * (bar_size.x - 1) +\
		self.marker_now +\
		self.marker_future * bar_size.y

		print(f'\r[{bar}] {percent:.0f}% {label:10s}', end = '')

