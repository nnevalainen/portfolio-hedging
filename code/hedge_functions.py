def delta_hedge(maturity, strike, price, q, sigma):

	# d1 from the Black-Scholes equation
	d1 = (np.log(price/strike) + (q**2 + sigma**2/2) * maturity)/(sigma*math.sqrt(maturity))

	return scipy.stats.norm.cdf(d1)

def calculate_implied_volatility_bs(maturity, strike, price, q, Cobs, threshold = 0.01, initial = 0.5):

	sigmas = initial
	diff = 1

	while diff > threshold:
		# Calculate Black-Scholes price
		d1 = (np.log(price/strike) + (q**2 + sigma**2/2) * maturity)/(sigma*math.sqrt(maturity))
		d2 = d1 - sigma*math.sqrt(maturity)
		bs = price * scipy.stats.norm.cdf(d1) - strike*math.exp(-q*maturity)*scipy.stats.norm.cdf(d2)

		# First derivaive of BS w.r.t volatility
		bs_der = price*math.sqrt(maturity)*math.exp(-d1**2/2)/(math.sqrt(2*math.pi))

		# Calculate the volatility difference
		diff = (Cobs-bs)/bs_der

		# updaede
		sigma += diff

	return sigma