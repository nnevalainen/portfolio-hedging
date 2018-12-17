import numpy as np
import math
import scipy.stats

def calculate_implied_volatility_bs(maturity, strike, spot, q, price, threshold = 0.01, initial = 0.5):
	'''
	Calculates the implied volatility of the option using the Black-Scholes formula and the
	Newton method.
	'''


	# Initialize
	sigma = initial
	diff = 1

	while diff > threshold:
		# Calculate Black-Scholes spot
		d1 = (np.log(spot/strike) + (q**2 + sigma**2/2) * maturity)/(sigma*math.sqrt(maturity))
		d2 = d1 - sigma * math.sqrt(maturity)
		bs = spot * scipy.stats.norm.cdf(d1) - strike * math.exp(-q * maturity) * scipy.stats.norm.cdf(d2)

		# First derivaive of BS w.r.t volatility
		bs_der = spot * math.sqrt(maturity) * math.exp(-d1**2 / 2) / (math.sqrt(2 * math.pi))

		# Calculate the volatility difference
		diff = (price - bs) / bs_der

		# updaede
		sigma += diff

	return sigma

def delta(maturity, strike, spot, q, sigma):
	'''
	Calculates the option delta (price-sensitivity)
	'''

	# d1 from the Black-Scholes equation
	d1 = (np.log(spot/strike) + (q**2 + sigma**2 / 2) * maturity)/(sigma*math.sqrt(maturity))

	return scipy.stats.norm.cdf(d1)

def vega(maturity, strike, spot, q, sigma):
	'''
	Calculates the option vega (volatility-sensitivity)
	'''

	# d1 from the Black-Scholes equation
	d1 = (np.log(spot/strike) + (q**2 + sigma**2/2) * maturity) / (sigma * math.sqrt(maturity))

	# Calculate vega value
	vega = spot * math.exp(-d1**2 / 2)*math.sqrt(maturity) / (math.sqrt(2 * math.pi))

	# Return as fraction
	return vega/100

def delta_hedge(maturity, strike, spot, q, sigma):
	'''
	Returns the amount of stock to buy (short) for any given option position to
	make the position delta-neutral
	'''

	# Calculate delta. This is the number to short the underlying
	delta_bs = delta(maturity, strike, spot, q, sigma)

	return -delta_bs

def vega_hedge(maturity_1, maturity_2, strike, spot, q, sigma):
	'''
	Returns the multipliers on which to make the portfolio vega-neutral. We should buy
	alpha amount of the underlying and eta amount of replicating option
	with a longer maturity than the original option, maturity_2 > maturity_1
	'''


	# Calculate hedged option and replication option deltas
	delta_bs = delta(maturity_1, strike, spot, q, sigma)
	delta_rep = delta(maturity_2, strike, spot, q, sigma)

	# Calculate hedged option and replication option vegas
	vega_bs = vega(maturity_1, strike, spot, q, sigma)
	vega_rep = vega(maturity_2, strike, spot, q, sigma)

	# Alpha is the amount of stock to hold - eta is the amount of replication stock to hold
	alpha = -delta_bs + vega_bs / vega_rep * delta_rep
	eta = - vega_bs / vega_rep

	return alpha, eta