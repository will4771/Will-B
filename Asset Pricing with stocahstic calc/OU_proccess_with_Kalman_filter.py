import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
from scipy.optimize import minimize
import yfinance as yf

# A.1 Simulation of Ornstein-Uhlenbeck Process using EM scheme


# Model parameters
alpha = 3
mu = 0.5
sigma = 0.5
Z0 = 2

# Simulation parameters
T = 1
N = 1000
dt = T / N
time = np.linspace(0, T, N)
Zt = np.zeros(N)
Zt[0] = Z0

# EM steps
for t in range(1, N):
    dW = np.random.normal(0, np.sqrt(dt))
    Zt[t] = Zt[t-1] + alpha * (mu - Zt[t-1]) * dt + sigma * dW


plt.plot(time, Zt)
plt.title('Simulated OU Process using Euler-Maruyama', size=20)
plt.xlabel('Time', size=18)
plt.ylabel(r'$Z_t$', size=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.text(0.85, 0.9, r'$\mu$ = ' + str(mu) + "\n" + r'$\alpha$ = ' + str(alpha) + "\n" + r'$\sigma$ = ' + str(sigma) + "\n" + r'$Z_0$ = ' + str(Z0),
         transform=plt.gca().transAxes, fontsize=18, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
plt.savefig("ou_process.svg", bbox_inches='tight')
plt.show()

# A.2 Simulation of Ornstein-Uhlenbeck Process from solution

# Model parameters
alpha = 3
mu = 0.5
sigma = 0.5
Z0 = 2

# Simulation parameters
T = 1
N = 1000
dt = T / N
time = np.linspace(0, T, N)
Zt = np.zeros(N)
Zt[0] = Z0

# Simulate directly from solution
for t in range(1, N):
    Zt[t] = mu + (Zt[t-1] - mu) * np.exp(-alpha * dt) + np.sqrt(sigma**2 / (2 * alpha) * (1 - np.exp(-2 * alpha * dt))) * np.random.normal(0, 1)


plt.plot(time, Zt)
plt.title("Simulated OU Process directly from solution", size=20)
plt.xlabel("Time", size=18)
plt.ylabel(r'$Z_t$', size=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.text(0.85, 0.9, r'$\mu$ = ' + str(mu) + "\n" + r'$\alpha$ = ' + str(alpha) + "\n" + r'$\sigma$ = ' + str(sigma) + "\n" + r'$Z_0$ = ' + str(Z0),
         transform=plt.gca().transAxes, fontsize=18, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

plt.show()

# A.3 OU parameter estimation using minimum negative log-likelihood
def loglikelihood(c, Xt_partial, time_partial):
    mu, alpha, sigma = c
    n = len(time_partial)
    dt = time[1] - time[0]
    L = -n / 2 * np.log(sigma**2 / (2 * alpha)) \
        - 1 / 2 * np.sum(np.log(1 - np.exp(-2 * alpha * dt))) \
        - alpha / sigma**2 * np.sum((Xt_partial[1:n] - mu - (Xt_partial[0:n-1] - mu) * np.exp(-alpha * dt))**2 / (1 - np.exp(-2 * alpha * dt)))
    return -L

def estimate_parameters(x0, Xt_partial, time_partial):
    result = minimize(loglikelihood, x0=x0, args=(Xt_partial, time_partial), method="L-BFGS-B", bounds=((None, None), (0.05, None), (0.05, None)))
    return result.x

# A.4 Simulation of OU process with noisy observations


sigma_o = 0.1
eps = ss.norm.rvs(loc=0, scale=sigma_o, size=N)
eps[0] = 0
Xt = Zt + eps


plt.plot(time, Xt, "-", alpha=0.5, label="Observations of Process")
plt.plot(time, Zt, "-b", label="True Process")
plt.title('Simulated OU process with noisy observations', size=20)
plt.xlabel('Time', size=18)
plt.ylabel(r'$Z_t$', size=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=18)
plt.text(0.05, 0.45, r'$\mu$ = ' + str(mu) + "\n" + r'$\alpha$ = ' + str(alpha) + "\n" + r'$\sigma$ = ' + str(sigma) + "\n" + r'$Z_0$ = ' + str(Z0) + "\n" + r'$\sigma_o$ = ' + str(sigma_o),
         transform=plt.gca().transAxes, fontsize=18, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

plt.show()

# A.5 OU Kalman Filter
def kalman_filter(Xt, time, mu, alpha, sigma, sigma_o):
    dt = time[1] - time[0]
    A = mu * (1 - np.exp(-alpha * dt))
    B = np.exp(-alpha * dt)
    F = np.array([[1, 0], [A, B]])
    sigma_p = np.sqrt(sigma**2 / (2 * alpha) * (1 - np.exp(-2 * alpha * dt)))
    P = np.eye(2) * sigma_p**2
    H = np.eye(2)
    Q = np.eye(2) * sigma_p**2
    R = np.eye(2) * sigma_o**2
    Z = np.zeros([len(Xt), 2])
    Xt = np.column_stack((np.ones_like(Xt), Xt))
    Z[0] = Xt[0]
    for i in range(len(time) - 1):
        Z[i + 1] = F @ Z[i]
        P = F @ P @ F.T + Q
        Y = Xt[i + 1] - H @ Z[i + 1]
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        Z[i + 1] = Z[i] + K @ Y
        P = (np.eye(2) - K @ H) @ P
    return Z

# A.6 Kalman filter with OU process on Apple's stock returns, fixed parameters
qqq = yf.Ticker("AAPL")
data = qqq.history(period="1y")
open_prices = data['Open'].dropna()
close_prices = data['Close'].dropna()
open_returns = open_prices.pct_change().dropna().values

# Estimate parameters on the first 30 days of data
start_index = 30
start_data = open_prices[0:start_index]
dt = 1
mu, alpha, sigma = estimate_parameters([170, 3, 0.1], start_data, np.arange(0, len(start_data), dt))
print(mu, alpha, sigma, sigma_o)

Yt = open_prices[start_index:]
time = np.arange(0, len(Yt), dt)
sigma_o = 20
Z = kalman_filter(Yt, time, mu, alpha, sigma, sigma_o)


plt.plot(np.array(data.index[1:start_index + 1]), np.array(open_prices)[1:start_index + 1], label=r'Parameter Estimation', color='green')
plt.plot(np.array(data.index[start_index:]), np.array(open_prices)[start_index:], "-", label='Stock Price', color='blue')
plt.plot(np.array(data.index[start_index:]), Z[:, 1], "--", label=r'Kalman Filter, $\sigma_o$ = ' + str(sigma_o), color="red")
plt.title('AAPL 1 Year Daily Stock Price with Kalman Filter', size=25)
plt.xlabel('Date', size=20)
plt.ylabel('Price [ $USD ]', size=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=18, loc="lower right")
plt.grid(True)

plt.show()
