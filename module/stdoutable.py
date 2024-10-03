import curses
import time

class stdoutable:

	def __init__(self, stdscr, data):

		self.column_size = 15
		self.stdscr = stdscr
		self.data = data
		self.h, self.w = stdscr.getmaxyx()
		self.start_x = 0 #self.w // 2 - len(self.data[0]) * self.column_size // 2
		self.start_y = 0 #self.h // 2 - len(self.data) // 2
		curses.curs_set(0)
		stdscr.nodelay(True)
		stdscr.timeout(1000)
		curses.initscr()
		curses.start_color()
		curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_GREEN)
		curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
		curses.init_pair(2, curses.COLOR_WHITE, curses.COLOR_RED)
		# curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)
		curses.init_pair(3, curses.COLOR_WHITE, curses.COLOR_BLACK)
		curses.init_pair(4, curses.COLOR_BLACK, curses.COLOR_BLACK)

	def draw_table(self):

		self.stdscr.clear()
		for i, row in enumerate(self.data):

			for j, val in enumerate(row):

				c = curses.color_pair(3)
				if isinstance(val, str):

					if '+' in val: c = curses.color_pair(1)
					if '-' in val: c = curses.color_pair(2)
					if val == '0': c = curses.color_pair(4)

				self.stdscr.addstr(self.start_y + i,
								   self.start_x + j * self.column_size,
								   str(val),
								   c)
		self.stdscr.refresh()

	def update_cell(self, row, col, value):

		if 0 <= row < len(self.data) and 0 <= col < len(self.data[0]):

			self.data[row][col] = value

	def randomize(self, x, y):

		#for i in range(1, len(self.data)):

		self.data[x][y] = time.time()

	def run(self, x):

		self.draw_table()
		while True:

			key = self.stdscr.getch()
			self.data[2][2] = x
			# self.randomize(2, 2)
			self.draw_table()
			if key == ord('q'): break

