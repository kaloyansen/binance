from binance import AsyncClient, BinanceSocketManager

class streamprice:

	def __init__(self, data_table, crypto_list):

		self.crypto = crypto_list
		self.data_table = data_table


	def account(self, api_key, api_sec):

		self.api_key = api_key
		self.api_sec = api_sec
		

	async def task_miniticker_socket(self):
	
		client = await AsyncClient.create(self.api_key, self.api_sec)
		bm = BinanceSocketManager(client)
		mt = bm.miniticker_socket()
		async with mt as miniticker_socket:

			while True:

				res = await miniticker_socket.recv()				
				for asset in res: self.update_data(asset)


	def update_data(self, raw_data):
		
		close_index = 1
		change_index = 2
		offset = 1 # leave space for a raw with column titles

		dindex = self.get_data_index(raw_data)
		if dindex < 0: return

		close = float(raw_data['c'])		
		change = 100 * self.get_change(raw_data)
		
		self.data_table[dindex + offset][close_index] = '{:14.7f}'.format(close)
		self.data_table[dindex + offset][change_index] = '{:+8.2f}%'.format(change)


	def get_data_index(self, raw_data):

		scoin = raw_data['s']
		search = scoin[:-4]
		try: return self.crypto.index(search.lower())
		except ValueError: return -1


	def get_change(self, dik):

		if not dik: return 0

		ppx = float(dik['o'])
		ppy = float(dik['c'])
		return 2 * (ppy - ppx) / (ppx + ppy)


