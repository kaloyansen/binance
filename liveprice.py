#!/usr/bin/env python
import sys
import websocket
import json

class Live:

	def __init__(self, crypto = 'fetusdt', trace = False):

		self.crypto = crypto
		websocket.enableTrace(trace)
		ws = websocket.WebSocketApp("wss://stream.binance.com:9443/ws",
									on_message = self.on_message,
									on_error = self.on_error,
									on_close = self.on_close)
		ws.on_open = self.on_open
		ws.run_forever()


	def on_error(self, ws, error): print(error)
	def on_close(self, ws): print("socket closed")
	def on_message(self, ws, message):

		data = json.loads(message)
		current_price = float(data['c'])
		print(f'\r{self.crypto} {current_price:.10f}', end = '')

	def on_open(self, ws):

		ws.send(json.dumps({"method": "SUBSCRIBE",
							"params": [f"{self.crypto}@ticker"], "id": 1}))


def main(coiname):

	if coiname == 'help':

		print('\nusage: {} <coin>\n'.format(sys.argv[0]))
		return 0

	Live(coiname + 'usdt')


if __name__ == "__main__":

	coin = sys.argv[1] if len(sys.argv) > 1 else 'btc'
	main(coin)
