import websocket
import json

class web:

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


