import numpy as np
import matplotlib . pyplot as plt
import scipy . stats as ss
from scipy . optimize import minimize
import scipy
import yfinance as yf






def main():


    # Model Parameters

    mu = 0.5 # mean
    alpha = 3 # spead of the mean reversion
    sigma = 0.5 # S.D
    X0 = 2 ## intial price

    # simulation paramters
    
    T = 1 # end time
    N = 1000 # number of steps
    dt = T/N 
    time = np.linspace(0, T, N)
    Xt = np.zeros(N)

    Xt[0] = X0

    for t in range(1, N):
        dW = np.random.normal(0, np.sqrt(dt))
        Xt[t] = Xt[t-1] + alpha * (mu - Xt[t-1]) * dt + sigma * dW

    plt.plot(time,Xt)
    plt.show()


main()
