#!/home/kalo/venv/bin/python

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

coin_list = ['BTCUSDT',
			 'ATAUSDT',
			 'COSUSDT',
			 'ELFUSDT', 'ETHUSDT',
			 'FETUSDT', 'FILUSDT', 'FIOUSDT',
			 'GTCUSDT',
			 'IDUSDT', 'INJUSDT',
			 'JASMYUSDT',
			 'LOKAUSDT', 'LOOMUSDT',
			 'PEPEUSDT',
			 'RADUSDT', 'REZUSDT',
			 'SOLUSDT',
]
coin_list_test = ['BTCUSDT', 'ETHUSDT', 'FIOUSDT', 'JASMYUSDT']
# coin_list = coin_list_test

binance_client = Client()
data_delta = '1h'
data_history = '1M'
data_t0 = 1714510800 # halving 2024
events_min = 1e2
events_max = 1e5
polynom_degree = 8

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
    return seconds


def cli(clarg, age_by_default, delta_by_default):

	valid_intervals = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']
	print('cli: valid interval - m/inute, h/our, d/ay, w/eek, M/onth, Y/ear')

	global data_delta, data_t0, data_history
	data_delta = delta_by_default
	data_history = age_by_default

	clsize = len(clarg)
	if clsize > 1: data_history = clarg[1]
	if clsize > 2: data_delta = clarg[2]

	data_t0 = time.time() - interval_to_seconds(data_history)

	if not data_delta in valid_intervals:
#		print('cli:', data_t0, unix_to_date(data_t0), data_delta)
#	else:
		print('cli:', data_delta, 'is not a valid interval, valid intervals:', valid_intervals)
		exit(1)

		

	number_of_events = int(interval_to_seconds(data_history) / interval_to_seconds(data_delta))
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

def correlation(asset1, asset2 , plot):

	tickcolor = 'black'
	print(asset1, asset2, data_t0, unix_to_date(data_t0))

	coindatax = binance_client.get_historical_klines(asset1, data_delta, str(data_t0), limit = 1000)
	coindatay = binance_client.get_historical_klines(asset2, data_delta, str(data_t0), limit = 1000)

	dfx = data_frame(coindatax)
	dfy = data_frame(coindatay)
	t0 = dfx.index[0]
	t1 = dfx.index[-1]
	#dfx['weight'] = dfx.index - t0
	#dfy['weight'] = dfy.index - t0

	# print(df.info())

	dfrows, dfcolumns = dfx.shape

	nbins = calculate_number_of_bins(dfrows, 10, 33, 222)

	x0   = dfx['open'].iloc[0]
	x    = dfx['open'].iloc[-1]
	xmin = dfx['open'].min()
	xmax = dfx['open'].max()
	
	y0   = dfy['open'].iloc[0]
	y    = dfy['open'].iloc[-1]
	ymin = dfy['open'].min()
	ymax = dfy['open'].max()

	dx = x - x0
	dy = y - y0

	x_bins = np.linspace(xmin, xmax, nbins)
	y_bins = np.linspace(ymin, ymax, nbins)

	lendfx = len(dfx['open'])
	lendfy = len(dfy['open'])
	while lendfx < lendfy:

		print('{} != {} dropping index 0'.format(lendfx, lendfy))
		dfy = dfy.drop(dfy.index[0])
		lendfx = len(dfx['open'])
		lendfy = len(dfy['open'])
	
	corr, xedges, yedges = np.histogram2d(dfx['open'], dfy['open'], bins = [x_bins, y_bins])
	hist_flat = corr.flatten()
	x_bin_centers = (xedges[:-1] + xedges[1:]) / 2
	y_bin_centers = (yedges[:-1] + yedges[1:]) / 2
	x_bin_centers_flat = np.tile(x_bin_centers, len(y_bin_centers))
	y_bin_centers_flat = np.repeat(y_bin_centers, len(x_bin_centers))

	scatter = plot.scatter(x_bin_centers_flat, y_bin_centers_flat, s = hist_flat, c = hist_flat, cmap = 'viridis', alpha = 0.4, edgecolor = 'yellow', label = 'correlation')

	plot.quiver(x0, y0, dx, dy, angles = 'xy', scale_units = 'xy', scale = 1, linewidth = 3, color = 'b', alpha = 0.2, label = 'change')
	
	plot.plot([xmin, xmax], [ymin, ymax], 'w:', label = '_dyagunal')
	#plot.plot(df1['open'].iloc[-1], df2['open'].iloc[-1], 'ro', label = 'last')


	#plot.hist2d(df1['open'], df2['open'], bins = nbins, alpha = 0.4, label = "uniform", color = 'magenta', edgecolor = 'white', density = True, cmap = 'plasma')
	# im = plot.imshow(corr.T, origin = 'lower', aspect = 'auto', extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]])

	xchange = dx / x
	ychange = dy / y
	rate = xchange / ychange
	status = 0

	if ychange < 0: # bc down
		status = 'fall' if xchange < 0 else 'antirise'
	else: #bc up
		status = 'antifall' if xchange < 0 else 'rise'

	if rate > 1 or rate < -1: status = '{} {}'.format('super-', status)

	correlation_matrix = np.corrcoef(dfx['open'], dfy['open'])
	correlation_matrix_0_1 = correlation_matrix[0, 1]


	#print('status:', status)
	#rate = (dx / dy) * (y / x) - 1

	def colored_float(variable, value):

		color = 'green' if value > 0 else 'red'
		formatted_float = '{}: <span style="color:{}">{:+5.2f}</span>'.format(variable, color, value)
		# formatted_float = '{}: {}{:+5.2f}{}'.format(variable, color, value, Style.RESET_ALL)

		return formatted_float

	
	title = '{:s}: {:s}\nstatus: {:s}'.format(asset1,
											  str(x).rstrip('0').rstrip('.'),
											  status)
	plot.set_title(title, color = tickcolor, loc = 'left', x = 0.03, y = 0.66)
	#plot.axes.get_yaxis().set_visible(False)
	plot.set_yticklabels([])

	def plotext(plot, var, val, x, y, ha = 'right', va = 'bottom'):
		color = 'green' if val > 0 else 'red'
		plot.text(x, y, '{}: {:+5.4}'.format(var, val), transform = plot.transAxes, ha = ha, va = va, color = color)

	plotext(plot, 'rate', rate, 0.96, 0.05)
	plotext(plot, 'correlation', correlation_matrix_0_1, 0.96, 0.15)
	plotext(plot, 'change', xchange, 0.96, 0.25)

	return t0, t1


def histogram(coin, plot):

	tickcolor = 'black'
	print(coin, data_t0, unix_to_date(data_t0))

	coindata = binance_client.get_historical_klines(coin, data_delta, str(data_t0), limit = 1000)

	df = data_frame(coindata)
	t0 = df.index[0]
	t1 = df.index[-1]
	df['weight'] = df.index - t0

	# print(df.info())

	dfrows, dfcolumns = df.shape

	nbins = calculate_number_of_bins(dfrows, 10, 33, 222)

	print('frame size: ', dfrows, 'histogram size: ', nbins)

	whist, wbin = np.histogram(df['open'], bins = nbins, density = True, weights = df['weight'])
	uhist, ubin = np.histogram(df['open'], bins = nbins, density = True)
	plot.hist(df['open'], bins = nbins, alpha = 0.4, label = "uniform", color = 'magenta', edgecolor = 'white', density = True)
	plot.hist(df['open'], bins = nbins, alpha = 0.4, label = "weighted", color = 'cyan', edgecolor = 'black', density = True, weights = df['weight'])
#	plot.ticklabel_format(axis = 'y', style='sci', scilimits = (0, 0))

	first = df['open'].iloc[0]
	last = df['open'].iloc[-1]

	plot.axvline(x = first, color = tickcolor, linestyle = ':', linewidth = 3, label = unix_to_date(t0))
	plot.axvline(x = last, color = tickcolor, linestyle = '-', linewidth = 3, label = unix_to_date(t1))

	ucoeff = poly_fit(plot, ubin, uhist, 'uni poly', 'm-')
	wcoeff = poly_fit(plot, wbin, whist, 'wei poly', 'c-')
	# umean, ustdev = gauss_fit(plot, df['open'], ubin, uhist, 'uni gauss', 'y-')
	# wmean, wstdev = gauss_fit(plot, df['open'], wbin, whist, 'wei gauss', 'k-')
	print ('histogram:', ucoeff, wcoeff)


	title = "{:s} {:7.3f}".format(coin, last)
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


def update_figure(plot, ynet):
	global coinlist
	xindex = yindex = 0
		
	for coin in coin_list:
		#t0, t1 = histogram(coin, plot[xindex, yindex])
		#print('net: {0}, {1}'.format(xindex, yindex))
		t0, t1 = correlation(coin, 'BTCUSDT', plot[xindex, yindex])
		# if coin == 'BTCUSDT': plot[xindex, yindex].legend()
		if yindex == ynet - 1:
			xindex += 1
			yindex = 0
		else:
			yindex += 1


	plt.subplots_adjust(wspace = 0, hspace = 0.44, left = 0, right = 1)
	

def main():

	global binance_client
	binance_client = Client(os.environ.get('BINANCE_KEY'), os.environ.get('BINANCE_SECRET'))
	binance_client.API_URL = 'https://testnet.binance.vision/api'
	binance_client.API_URL = 'https://api.binance.vision'
	binance_client.API_URL = 'https://api.binance.com/api'
	#print(binance_client.get_account())
	#print(binance_client.get_asset_balance(asset='ATA'))

	cli(sys.argv, '1M', '1h')
	events = int(interval_to_seconds(data_history) / interval_to_seconds(data_delta))
	print('main:', data_t0, unix_to_date(data_t0), data_history, data_delta, events)

	global coin_list
	coin_list_size = len(coin_list)
	# coin_list_size += 1
	
	fig_width = 5
	fig_height = 8
	aspect_ratio = fig_width / fig_height

	netx = nety = 0
	while netx * nety < coin_list_size:
		if (netx < nety): netx += 1
		else: nety += 1

	print('net: {0} * {1} > {2}'.format(netx, nety, coin_list_size))
	

	plt.ion()
	fig, axs = plt.subplots(netx, nety, figsize = (fig_width, fig_height))
	fig.suptitle('history = {0}, time = {1}, {2} events'.format(data_history, data_delta, events))
	fig.text(0.8, 0.03, 'kaloyansen.github.io', ha = 'center', va = 'center')

	
	t0 = t1 = 0
	init() # colorama

	print('coin list:', coin_list)

	continuer = True
	while continuer:
		update_figure(axs, nety)
		plt.draw()
		#plt.show()
		plt.pause(0.1)
		# time.sleep(10)
		continuer = False

	plt.ioff()
	plt.show()


if __name__ == "__main__":
	main()
