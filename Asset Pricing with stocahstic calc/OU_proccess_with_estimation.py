import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
from scipy.optimize import minimize
import yfinance as yf

'''
Implementing the Ornstein-Uhlenbeck SDE (mean reversion).
We define the log likelihood function as well as a function that minimizes and produces the MLE for 
mu, alpha, sigma.
'''

def loglikelihood(c, Xt_partial, time_partial):  # c is our parameters
    mu, alpha, sigma = c
    n = len(time_partial)
    dt = time_partial[1] - time_partial[0]

    L = -n / 2 * np.log(sigma**2 / (2 * alpha)) \
        - 1/2 * np.sum(np.log(1 - np.exp(-2 * alpha * dt))) \
        - alpha / sigma**2 * np.sum((Xt_partial[1:n] - mu - (Xt_partial[0:n-1] - mu) * np.exp(
            -alpha * dt)) ** 2 / (1 - np.exp(-2 * alpha * dt)))

    return -L

def estimate_parameters(x0, Xt_partial, time_partial):
    result = minimize(
        loglikelihood,
        x0=x0,
        args=(Xt_partial, time_partial),
        method="L-BFGS-B",
        bounds=((None, None), (0.05, None), (0.05, None))
    )
    
    mu, alpha, sigma = result.x
    return mu, alpha, sigma

# Load in AAPL data and retrieve open prices

data = yf.download('AAPL', start='2024-01-01', end='2024-08-01')

open_prices = data["Open"].dropna()

# Estimate parameters on the first 30 days of data
start_index = 60
start_data = open_prices[:start_index].values

dt = 1

mu, alpha, sigma = estimate_parameters([170, 3, 0.1], start_data, np.arange(0, len(start_data), dt))

Xt = open_prices[start_index:].values

time = np.arange(0, len(Xt), dt)

N = len(Xt)

sigma_o = 20

print(mu, alpha, sigma, sigma_o, N)

# Simulation of the Ornstein-Uhlenbeck process
simulated_Xt = np.zeros(N)
simulated_Xt[0] = Xt[0]

for t in range(1, N):
    dW = np.random.normal(0, np.sqrt(dt))
    simulated_Xt[t] = simulated_Xt[t-1] + alpha * (mu - simulated_Xt[t-1]) * dt + sigma * dW

plt.figure(figsize=(12, 6))
plt.plot(data.index[start_index:], simulated_Xt, label='Simulated')
plt.plot(data.index, open_prices, label='Actual')
plt.xlabel('Date')
plt.ylabel('Price in Dollars')
plt.title('Ornstein-Uhlenbeck Process Simulation vs Actual Prices (AAPl)')
plt.legend()
plt.show()

print(len(open_prices))