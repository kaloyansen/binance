#!/usr/bin/env python

import os
import sys
import time
import datetime

from binance.client import Client
# from binance import ThreadedWebsocketManager
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#from scipy.stats import norm
from scipy.optimize import curve_fit
from colorama import Fore, Style, init

coin_list_full = ['BTCUSDT',
				  'AKROUSDT', 'ATAUSDT',
				  'COSUSDT', 'CITYUSDT',
				  'DOGEUSDT',
				  'ELFUSDT', 'ETHUSDT',
				  'FETUSDT', 'FILUSDT', 'FIOUSDT',
				  'GTCUSDT',
				  'IDUSDT', 'INJUSDT',
				  'JASMYUSDT',
				  'LOKAUSDT', 'LOOMUSDT',
				  'PEPEUSDT',
				  'RADUSDT', # 'REZUSDT',
				  'SOLUSDT',
				  'UNFIUSDT',
				  'VICUSDT',
				  ]
coin_list_test = ['BTCUSDT', 'ETHUSDT', 'FIOUSDT', 'JASMYUSDT']

run_mode = 'test'
# run_mode = 'default'

coin_list = coin_list_test if run_mode == 'test' else coin_list_full

binance_client = Client()
data_delta = '1h'
data_history = '1M'
data_t0 = 1714510800 # halving 2024
events_min = 1e3
events_max = 1e5
events_default = 4e3
polynom_degree = 6


class baby:

	def __init__(self, x, y): self.set(x, y)

	def set(self, x, y):

		self.x = x
		self.y = y
		
	def notegal(self): return self.x != self.y
	def rate(self): return self.x / self.y


def interval_to_seconds(interval):
    """Convert a Binance interval string to milliseconds

    :param interval: Binance interval string 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w
    :type interval: str

    :return:
         None if unit not one of m, h, d or w
         None if string not in correct format
         int value of interval in milliseconds
    """
    seconds = None
    seconds_per_unit = {
        "m": 60,
        "h": 60 * 60,
        "d": 24 * 60 * 60,
        "w": 7 * 24 * 60 * 60,
        "M": 30 * 24 * 60 * 60,
        "Y": 365 * 24 * 60 * 60
    }

    unit = interval[-1]
    if unit in seconds_per_unit:
        try:
            seconds = int(interval[:-1]) * seconds_per_unit[unit]
        except ValueError:
            pass
    return float(seconds)


def cli(clarg, age_by_default, delta_by_default):

	valid_intervals = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']
	print('cli: valid interval - m/inute, h/our, d/ay, w/eek, M/onth, Y/ear')

	global data_delta, data_t0, data_history, events_max, events_default
	data_delta = delta_by_default
	data_history = age_by_default

	clsize = len(clarg)
	if clsize > 1:
		data_history = clarg[1]
		historinsec = interval_to_seconds(data_history)

	if clsize == 2:
		number_of_events = events_max
		inter = 0
		while number_of_events > events_default:
			delta = valid_intervals[inter]
			number_of_events = historinsec / interval_to_seconds(delta)
			data_delta = delta
			inter += 1

	if clsize > 2: data_delta = clarg[2]
	if 'test' in clarg:
		data_history = '1d'
		data_delta = '1m'

	historinsec = interval_to_seconds(data_history)
	data_t0 = time.time() - historinsec;
	if not data_delta in valid_intervals:
		print('cli:', valid_intervals)
		exit(1)

		

	number_of_events = historinsec / interval_to_seconds(data_delta)
	if number_of_events < events_min: die('not enough events {}'.format(number_of_events))
	if number_of_events > events_max: die('too much events {}'.format(number_of_events))
	#from_the_very_beginning = binance_client._get_earliest_valid_timestamp(coin, data_delta)
	#data_t0 = from_the_very_beginning


def die(message):

	print(message)
	exit(1)

def gaussian(x, mu, sigma):

	gauss = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
	return gauss

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

def unix_to_date(rawtime):
	bintime = str(rawtime)
	unixtime = int(bintime[:10])

	timestamp = datetime.datetime.utcfromtimestamp(unixtime)
	human_date = timestamp.strftime('%d/%m/%Y') # %H:%M:%S')
	return human_date

def month_ago(delta = 1):

	now = time.time()
	return now - delta * 30 * 24 * 60 * 60

def calculate_number_of_bins(number_of_events, number_of_bins_min = 10, number_of_bins_max = 22, events_per_bin = 500):

	number_of_bins = int(number_of_events / events_per_bin)
	if number_of_bins < number_of_bins_min: number_of_bins = number_of_bins_min
	if number_of_bins > number_of_bins_max: number_of_bins = number_of_bins_max
	return number_of_bins

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
	df.to_csv('coindata.csv')
	return df


def update_price(info):
	latest = {}
	latest['last'] = info['c']
	latest['bid'] = info['b']
	latest['ask'] = info['a']
	history.append(latest)
	print(history[-1])

def plotext(plot, var, val, x, y, va = 'bottom', ha = 'right', scale = 100):

	val = val * scale
	color = 'green' if val > 0 else 'red'
	plot.text(x, y,
			  '{} {:+5.4}%'.format(var, val),
			  transform = plot.transAxes,
			  ha = ha,
			  va = va,
			  color = color)

def correlation(asset1, asset2 , plot, tickcolor = 'black'):

	coindatax = binance_client.get_historical_klines(asset1, data_delta, str(data_t0), limit = 1000)
	coindatay = binance_client.get_historical_klines(asset2, data_delta, str(data_t0), limit = 1000)

	df = baby(data_frame(coindatax), data_frame(coindatay))
	prix = baby(df.x['open'], df.y['open'])
	t0 = df.x.index[0]
	t1 = df.x.index[-1]
	#df.x['weight'] = df.x.index - t0
	#df.y['weight'] = df.y.index - t0

	# print(df.info())

	dfrows, dfcolumns = df.x.shape

	nbins = calculate_number_of_bins(dfrows, 10, 33, 222)

	step = baby(int(len(prix.x) / 4), int(len(prix.y) / 4))

	x = []
	x.append(prix.x.iloc[0 * step.x])
	x.append(prix.x.iloc[1 * step.x])
	x.append(prix.x.iloc[2 * step.x])
	x.append(prix.x.iloc[3 * step.x])
	x.append(prix.x.iloc[-1])
	xmin = prix.x.min()
	xmax = prix.x.max()
	
	y = []
	y.append(prix.y.iloc[0 * step.y])
	y.append(prix.y.iloc[1 * step.y])
	y.append(prix.y.iloc[2 * step.y])
	y.append(prix.y.iloc[3 * step.y])
	y.append(prix.y.iloc[-1])
	ymin = prix.y.min()
	ymax = prix.y.max()

	dx = []
	dx.append(x[1] - x[0])
	dx.append(x[2] - x[1])
	dx.append(x[3] - x[2])
	dx.append(x[4] - x[3])
	dx.append(x[-1] - x[0])

	dy = []
	dy.append(y[1] - y[0])
	dy.append(y[2] - y[1])
	dy.append(y[3] - y[2])
	dy.append(y[4] - y[3])
	dy.append(y[-1] - y[0])


	bins = baby(np.linspace(xmin, xmax, nbins), np.linspace(ymin, ymax, nbins))

	lendf = baby(len(df.x), len(df.y))
	while lendf.notegal():

		print('\n{} {} != {} dropping 0'.format(asset1, lendf.x, lendf.y))
		if lendf.x < lendf.y: df.y = df.y.drop(df.y.index[0])
		else: df.x = df.x.drop(df.x.index[0])

		lendf.set(len(df.x), len(df.y))
		print('new data {} {}'.format(lendf.x, lendf.y))



	
	prix.set(df.x['open'], df.y['open'])
	hist2d, xedges, yedges = np.histogram2d(prix.x, prix.y, bins = [bins.x, bins.y])
	hist_flat = hist2d.flatten()
	bin_centers = baby((xedges[:-1] + xedges[1:]) / 2, (yedges[:-1] + yedges[1:]) / 2)
	bin_centers_flat = baby(np.tile(bin_centers.x, len(bin_centers.y)), np.repeat(bin_centers.y, len(bin_centers.x)))

	scatter = plot.scatter(bin_centers_flat.x, bin_centers_flat.y, s = hist_flat, c = hist_flat, cmap = 'Blues', marker = 'p', alpha = 0.4, edgecolor = 'yellow', label = 'correlation')

	
	#plot.plot([xmin, xmax], [ymin, ymax], 'w:', label = '_dyagunal')
	#plot.plot(df1['open'].iloc[-1], df2['open'].iloc[-1], 'ro', label = 'last')


	#plot.hist2d(df1['open'], df2['open'], bins = nbins, alpha = 0.4, label = "uniform", color = 'magenta', edgecolor = 'white', density = True, cmap = 'plasma')
	# im = plot.imshow(hist2d.T, origin = 'lower', aspect = 'auto', extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]])

	change = baby(dx[-1] / x[-1], dy[-1] / y[-1])
	rate = change.rate()
	status = 0

	if change.y < 0: status = 'fall' if change.x < 0 else 'antirise'
	else: status = 'antifall' if change.x < 0 else 'rise'
	vectorcolor = 'red'	if change.x < 0 else 'green'

	#plot.quiver(x[0], y[0], dx[-1], dy[-1], angles = 'xy', scale_units = 'xy', scale = 1, linewidth = 5, linestyle = '-', color = vectorcolor, alpha = 0.2, label = 'change')
	q = 0
	while q < 4:
		plot.quiver(x[q], y[q], dx[q], dy[q], angles = 'xy', scale_units = 'xy', scale = 1, linewidth = 5, linestyle = '-', color = vectorcolor, alpha = 0.2, label = 'change')
		q += 1

	q = 0
	while q < abs(rate):
		
		q += 1
		status = 'super-' + status if abs(rate) > q else status 

	correlation_matrix = np.corrcoef(prix.x, prix.y)
	correlation_matrix_0_1 = correlation_matrix[0, 1]

	jvhdskvs = str(x[-1]).rstrip('0').rstrip('.')
	title = '{:s} = {:s}\n{:s}'.format(asset1, jvhdskvs, status)
	plot.set_title(title, color = tickcolor, x = 0.06, y = 0.9, ha = 'left', va = 'top')
	plot.set_yticklabels([])

	plotext(plot, '', change.x, 0.96, 0.96, 'top')
	plotext(plot, 'corr', correlation_matrix_0_1, 0.96, 0.15)
	plotext(plot, 'rate', rate, 0.96, 0.05)

	return t0, t1


def histogram(coin, plot, tickcolor = 'black'):

	# print(coin, data_t0, unix_to_date(data_t0))

	coindata = binance_client.get_historical_klines(coin, data_delta, str(data_t0), limit = 1000)

	df = data_frame(coindata)
	t0 = df.index[0]
	t1 = df.index[-1]
	df['weight'] = df.index - t0
	price = df['open']
	weight = df['weight']

	# print(df.info())

	dfrows, dfcolumns = df.shape

	nbins = calculate_number_of_bins(dfrows, 10, 33, 222)

	# print('frame size: ', dfrows, 'histogram size: ', nbins)

	whist, wbin = np.histogram(price, bins = nbins, density = True, weights = weight)
	uhist, ubin = np.histogram(price, bins = nbins, density = True)
	plot.hist(price, bins = nbins, alpha = 0.4, label = "uniform", color = 'magenta', edgecolor = 'white', density = True)
	plot.hist(price, bins = nbins, alpha = 0.4, label = "weighted", color = 'cyan', edgecolor = 'black', density = True, weights = weight)
#	plot.ticklabel_format(axis = 'y', style='sci', scilimits = (0, 0))

	fromto = baby(price.iloc[0], price.iloc[-1])

	plot.axvline(x = fromto.x, color = tickcolor, linestyle = ':', linewidth = 3, label = unix_to_date(t0))
	plot.axvline(x = fromto.y, color = tickcolor, linestyle = '-', linewidth = 3, label = unix_to_date(t1))

	ucoeff = poly_fit(plot, ubin, uhist, 'uni poly', 'm-')
	wcoeff = poly_fit(plot, wbin, whist, 'wei poly', 'c-')
	# umean, ustdev = gauss_fit(plot, price, ubin, uhist, 'uni gauss', 'y-')
	# wmean, wstdev = gauss_fit(plot, price, wbin, whist, 'wei gauss', 'k-')
	# print ('histogram:', ucoeff, wcoeff)


	title = "{:s} {:7.3f}".format(coin, fromto.y)
	plot.set_title(title, color = tickcolor, loc = 'center', y = 0.66)
		
	return t0, t1 


def gauss_fit(plot, column, bins, hist, label, color):

	x = np.linspace(column.min(), column.max(), 33)

	popt, pcov = curve_fit(gaussian, bins[:-1], hist)
	mean = popt[0]
	stdev = popt[1]

	plot.plot(x, gaussian(x, *popt), color, label = label)
	return mean, stdev 

def poly_fit(plot, bins, hist, label = 'poly', color = 'k'):

	global polynom_degree
	centers = (bins[:-1] + bins[1:]) / 2
	coeff = np.polyfit(centers, hist, deg = polynom_degree)
	p = np.poly1d(coeff)

	plot.plot(centers, p(centers), color, label = '{} {}'.format(label, polynom_degree))
	return coeff


def check_vector_direction(dx, dy):

    if dx > 0:
        horizontal_direction = "right"
    elif dx < 0:
        horizontal_direction = "left"
    else:
        horizontal_direction = "none"

    if dy > 0:
        vertical_direction = "up"
    elif dy < 0:
        vertical_direction = "down"
    else:
        vertical_direction = "none"

    return horizontal_direction, vertical_direction



def progress_bar(progress, total):

	partition = progress / total
	percent = partition * 100
	bar = '#' * int(percent / 2) + '-' * (50 - int(percent / 2))
	print(f'\r[{bar}] {percent:.2f}%', end = '')


def update_figure(ax, ynet, mode = 'corr'):
	global coinlist
	index = baby(0, 0)
		
	for coin in coin_list:

		progress_bar(ynet * index.x + index.y, len(coin_list))
		# print('net: {0}, {1}'.format(index.x, index.y))
		if mode == 'corr':
			correlation(coin, 'BTCUSDT', ax[index.x, index.y], 'orange')
		else:
			histogram(coin, ax[index.x, index.y], 'orange')
			if coin == 'BTCUSDT': ax[index.x, index.y].legend()
			
		if index.y == ynet - 1:

			index.x += 1
			index.y = 0
		else:

			index.y += 1

	progress_bar(ynet * index.x + index.y, len(coin_list))
	print()

	

def main():

	global binance_client, coin_list, run_mode
	binance_client = Client(os.environ.get('BINANCE_KEY'), os.environ.get('BINANCE_SECRET'))
	binance_client.API_URL = 'https://testnet.binance.vision/api'
	binance_client.API_URL = 'https://api.binance.vision'
	binance_client.API_URL = 'https://api.binance.com/api'
	#print(binance_client.get_account())
	#print(binance_client.get_asset_balance(asset='ATA'))

	cli(sys.argv, '1M', '1h')
	events = int(interval_to_seconds(data_history) / interval_to_seconds(data_delta))
	print('main:', data_t0, unix_to_date(data_t0), data_history, data_delta, events)

	coin_list_size = len(coin_list)
	# coin_list_size += 1
	
	fig_size = baby(5, 8)
	aspect_ratio = fig_size.x / fig_size.y

	net = baby(0, 0)
	while net.x * net.y < coin_list_size:
		if (net.x < net.y): net.x += 1
		else: net.y += 1

	print('mode: {0}, net: {1} * {2} > {3}'.format(run_mode, net.x, net.y, coin_list_size))
	

	#plt.ion()
	fig, axs = plt.subplots(net.x, net.y, figsize = (fig_size.x, fig_size.y))
	plt.subplots_adjust(wspace = 0, hspace = 0.44, left = 0, right = 1)
	figg, axss = plt.subplots(net.x, net.y, figsize = (fig_size.x, fig_size.y))
	plt.subplots_adjust(wspace = 0, hspace = 0.44, left = 0, right = 1)
	fig.suptitle('history = {0}, time = {1}, {2} events'.format(data_history, data_delta, events))
	figg.suptitle('history = {0}, time = {1}, {2} events'.format(data_history, data_delta, events))
	fig.text(0.8, 0.03, 'kaloyansen.github.io', ha = 'center', va = 'center')
	figg.text(0.8, 0.03, 'kaloyansen.github.io', ha = 'center', va = 'center')

	
	#init() # colorama

	print('coin list:', coin_list)

	continuer = True
	while continuer:
		update_figure(axs, net.y, 'corr')
		update_figure(axss, net.y, 'hist')
		#plt.draw()
		#plt.show()
		#plt.pause(0.1)
		# time.sleep(10)
		continuer = False

	#plt.ioff()
	plt.show()


if __name__ == "__main__":
	main()
