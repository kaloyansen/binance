from binance import AsyncClient, BinanceSocketManager

class streamprice:

	def __init__(self, data_table, crypto_list):

		self.crypto = crypto_list
		self.data_table = data_table
		self.quit = False



	def close(self): self.quit = True
	def login(self, api_key, api_sec):

		self.api_key = api_key
		self.api_sec = api_sec
		

	async def connexion(self):

		client = await AsyncClient.create(self.api_key, self.api_sec)
		return client

	async def task_account(self):

		client = await self.connexion()

		#while True:

		account_info = await client.get_account()
		bal = account_info['balances']

		for asset in bal:

			aname = asset['asset']
			afree = float(asset['free'])				
			message = f'{aname} {afree}' if afree > 0 else False
			if message: print(message)
			if self.quit: break
			
		ac = await client.close_connection()		
		
		
	async def task_miniticker_socket(self):
	
		client = await self.connexion()
		bm = BinanceSocketManager(client)
		mt = bm.miniticker_socket()
		async with mt as miniticker_socket:

			while True:

				res = await miniticker_socket.recv()				
				for asset in res: self.update_data(asset)
				if self.quit: break
		ac = await client.close_connection()


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


