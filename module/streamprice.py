from binance import AsyncClient, BinanceSocketManager
import asyncio, time

class streamprice:

	def __init__(self, data_table, crypto_list):

		self.crypto = crypto_list
		self.data_table = data_table
		self.quit = False
		self.offset = 1

	def close(self): self.quit = True
	def login(self, api_key, api_sec):

		self.api_key = api_key
		self.api_sec = api_sec
		

	async def connexion(self):

		client = await AsyncClient.create(self.api_key, self.api_sec)
		return client

	async def task_account(self):

		client = await self.connexion()

		account_info = await client.get_account()
		bal = account_info['balances']
		while True:
			
			if self.quit: break
			for asset in bal:

				aname = asset['asset']
				afree = float(asset['free'])				
				if afree > 0: self.capital(aname.lower(), afree)

			await asyncio.sleep(1)
							
		await client.close_connection()
		
		
	def capital(self, name, free):

		index = self.get_table_index(name)
		if not index < 0:

			index += 1
			raw = self.data_table[index]
			price = raw[1]
			raw[3] = '{:7.2f}'.format(float(price) * free)

		count = self.offset
		total = 0
		while count < len(self.crypto) + self.offset:

			total += float(self.data_table[count][3])
			count += 1
		self.data_table[count][2] = 'total'
		self.data_table[count][3] = '{:7.2f}'.format(total)

		


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
		dindex = self.get_raw_data_index(raw_data)
		if dindex < 0: return

		close = float(raw_data['c'])		
		change = 100 * self.get_change(raw_data)
		
		self.data_table[dindex + self.offset][close_index] = '{:14.7f}'.format(close)
		self.data_table[dindex + self.offset][change_index] = '{:+8.2f}'.format(change)


	def get_table_index(self, crypto):

		try: return self.crypto.index(crypto.lower())
		except ValueError: return -1


	def get_raw_data_index(self, raw_data):

		scoin = raw_data['s']
		search = scoin[:-4]
		return self.get_table_index(search)


	def get_change(self, dik):

		if not dik: return 0

		ppx = float(dik['o'])
		ppy = float(dik['c'])
		return 2 * (ppy - ppx) / (ppx + ppy)


