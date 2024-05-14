#!/home/kalo/venv/bin/python

import os
import sys
import time
import datetime

from binance.client import Client
from binance import ThreadedWebsocketManager
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.optimize import curve_fit

def to_float(x):
	f = float(x)
	return f
	#return int(f)

def date_to_unix(time_str, time_format = '%Y-%m-%d %H:%M:%S'):

    try:
        dt_obj = datetime.datetime.strptime(time_str, time_format)
        unix_time = int(dt_obj.timestamp())
        return unix_time
    except ValueError:
        print("Error: Invalid time string or format.")
        return None

def gaussian(x, mu, sigma):
	return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - mu)**2 / (2 * sigma**2))

def unix_to_date(rawtime):
	bintime = str(rawtime)
	unixtime = int(bintime[:10])

	timestamp = datetime.datetime.utcfromtimestamp(unixtime)
	human_date = timestamp.strftime('%Y-%m-%d %H:%M:%S')
	return human_date

def cli(coin_by_default = 'btc', ticksize_by_default = '1h'):
	print('cli: valid intervals - 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M')
	coiname = coin_by_default
	ticksize = ticksize_by_default

	clarg = sys.argv
	clsize = len(clarg)
	if clsize > 1:
		coiname = clarg[1]
	if clsize > 2:
		ticksize = clarg[2]

	coiname += 'usdt'
	coiname = coiname.upper()
	print('cli:', coiname, ticksize)
	return coiname, ticksize

def month_ago(delta = 1):

	now = time.time()
	return now - delta * 30 * 24 * 60 * 60

def calculate_number_of_bins(number_of_events, number_of_bins_min = 10, number_of_bins_max = 22, events_per_bin = 500):

	number_of_bins = int(number_of_events / events_per_bin)
	if number_of_bins < number_of_bins_min: number_of_bins = number_of_bins_min
	if number_of_bins > number_of_bins_max: number_of_bins = number_of_bins_max
	return number_of_bins


def main():

	client = Client(os.environ.get('BINANCE_KEY'), os.environ.get('BINANCE_SECRET'))
	client.API_URL = 'https://testnet.binance.vision/api'
	client.API_URL = 'https://api.binance.com/api'
	#print(client.get_account())
	#print(client.get_asset_balance(asset='ATA'))

	coin, ticksize = cli('btc', '1h')

	halving24 = 1714510800
	forever = client._get_earliest_valid_timestamp(coin, ticksize)

	coinstart = forever
	coinstart = halving24
	coinstart = month_ago(1)
	print(coin, coinstart, unix_to_date(coinstart))

	coindata = client.get_historical_klines(coin, ticksize, str(coinstart), limit = 1000)

	def data_frame(coindata):

		df = pd.DataFrame(coindata, columns = ['date',
											   'open',
											   'high',
											   'low',
											   'close',
											   'volume',
											   'closetime',
											   'quoteassetvolume',
											   'trades',
											   'takerbuybaseassetvolume',
											   'takerbyquoteassetvolume',
											   'ignore'])
		df['date'] = df['date'].apply(to_float)
		df['open'] = df['open'].apply(to_float)
		df.set_index('date', inplace = True)
		return df

	df = data_frame(coindata)
	t0 = df.index[0]
	t1 = df.index[-1]
	df['weight'] = df.index - t0

	print(df.info())

	dfrows, dfcolumns = df.shape

	nbins = 16 #calculate_number_of_bins(dfrows, 10, 22, 500)

	print('frame size: ', dfrows, 'histogram size: ', nbins)

	whist, wbin = np.histogram(df['open'], weights = df['weight'], bins = nbins, density = True)
	plt.hist(df['open'], bins = nbins, alpha = 0.5, label = "uniform", color = 'yellow', edgecolor = 'black', density = True)
	plt.hist(df['open'], bins = nbins, alpha = 0.5, label = "weighted", color = 'grey', edgecolor = 'white', density = True, weights = df['weight'])
	plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

	mean = df['open'].mean()
	std = df['open'].std()
	cur = df['open'].iloc[-1]

	plt.axvline(x = cur, color = 'cyan', linestyle = ':', linewidth = 3, label = 'now')
	xlabel = "<{:7.2f}> +/- {:7.2f}          code kaloyansen.github.io".format(mean, std)
	ylabel = "since {0}".format(unix_to_date(t0))
	title = "{0} {1} {2}".format(cur, coin, unix_to_date(t1))
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title, color = 'cyan')

	popt, pcov = curve_fit(gaussian, wbin[:-1], whist)

	mu, std = norm.fit(df['open'])
	x = np.linspace(df['open'].min(), df['open'].max(), 100)
	p = norm.pdf(x, mu, std)

	plt.plot(x, gaussian(x, *popt), 'k+', label = 'weighted fit')
	plt.plot(x, p, 'k-', linewidth = 2, label = 'uniform fit')
	plt.legend()
	plt.show()




if __name__ == "__main__":
	main()
