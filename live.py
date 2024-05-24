#!/usr/bin/env python

import os
import time
import asyncio
import curses

from module.stdoutable import stdoutable
from module.streamprice import streamprice

def printable(stdscr): asyncio.ensure_future(task_printable(stdscr))	
async def task_printable(stdscr):

	global live_data, dead_time

	table = stdoutable(stdscr, live_data)
	while True:

		table.draw_table()
		await asyncio.sleep(dead_time)

def read_coin_list(filename):

	with open(filename, 'r') as file:

		lines = file.readlines()
	clist = [line.strip() for line in lines]
	return clist


if __name__ == '__main__':

	dead_time = 0.1

	coin_list = read_coin_list('coin.list')
	live_data = [['crypto', 'price/usdt', 'change/%']]
	for coin in coin_list:

		live_data.append([coin, 'loading...', '-'])

	binance = streamprice(live_data, coin_list)
	binance.account(os.environ.get('BINANCE_KEY'), os.environ.get('BINANCE_SECRET'))

	loop = asyncio.get_event_loop()

	try:

		minitask = asyncio.ensure_future(binance.task_miniticker_socket())
		curses.wrapper(printable)

		loop.run_forever()
	except KeyboardInterrupt:

		pass
	finally:

		print("closing Loop")
		minitask.cancel()
		loop.close()
