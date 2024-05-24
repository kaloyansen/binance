#!/usr/bin/env python

import sys
from module.live import live

def main(coiname):

	if coiname == 'help':

		print('\nusage: {} <coin>\n'.format(sys.argv[0]))
		return 0

	live(coiname + 'usdt')


if __name__ == "__main__":

	coin = sys.argv[1] if len(sys.argv) > 1 else 'btc'
	main(coin)
