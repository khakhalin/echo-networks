import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

def lorenz(n, t=None, start=None, alpha=0.01, sigma=10, beta=8/3, rho=28):
    """Lorenz system, with manual resampling"""
    if not start: start = (1,0,1)
    if not t: t = n # Default sampling of 1
    (x,y,z) = start
    full_n = round(t/alpha*1.1) # With some excess just in case
    rate = t/n                  # Resampling rate
    if rate < alpha:
        #raise Exception('Reampling rate is too high, hot supported.')
        alpha = rate
    history = np.zeros((n,3))
    (t, j) = (0, 0)         # Starting time and sampling counter
    for i in range(full_n):
        x,y,z = (x + alpha*sigma*(y - x),
                 y + alpha*(x*(rho-z) - y),
                 z + alpha*(x*y - beta*z))
        t += alpha
        if t >= rate:
            history[j,:] = (x,y,z)
            t += -rate
            j += 1
        if j==n:
            break
    return history