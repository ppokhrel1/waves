import numpy as np
#import pylab as pl
from numpy import fft
import sys



from scipy import signal


#data, trend repeats, number of harmonics
def fourierExtrapolation(x, n_predict, n_harm):
    n = len(x)
    #n_harm = int(sys.argv[3])                     # number of harmonics in model
    t = [a for a in range(0, n) ]
    #print(len(x))
    print(len(t))
    print(len(x))
    p = np.polyfit(t, x, 1)         # find linear trend in x
    #x_notrend = x - p[0] * t        # detrended x
    x_notrend = []
    for a in range( len(x) ):
        x_notrend.append( x[a] - p[0] * t[a] )
    x_freqdom = fft.fft(x_notrend)  # detrended x in frequency domain
    f = fft.fftfreq(n)              # frequencies
    indexes = [a for a in range(n) ]
    # sort indexes by frequency, lower -> higher
    indexes.sort(key = lambda i: np.absolute(f[i]))
    t = np.arange(0, n + n_predict)
    restored_sig = np.zeros(t.size)
    for i in indexes[:1 + n_harm * 2]:
        ampli = np.absolute(x_freqdom[i]) / n   # amplitude
        phase = np.angle(x_freqdom[i])          # phase
        restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
    return restored_sig + p[0] * t

from sklearn.linear_model import LinearRegression

def flatten_data(x, y):
    lr = LinearRegression().fit(x, y)
    slope = lr.coef_[0]
    y_int = lr.intercept_
    flattened = []
    for i in range(len(x)):
        expected_val = slope * x[i] + y_int
        expected_val = lr.predict([ x[i] ] )[0]
        flattened.append(y[i] - expected_val)
    return flattened


def difference(X, y,steps):
    x_val = X#[:len(X_temp) - steps]
    y_val = y#[a[0]  for a in y[steps: ]
    if steps == 0:
        return x_val, y_val, [0 for a in y_val]
    ret_val = [  ]
    trend = [  ]
    #print(y[:5])
    #ret_val = [a-b for a, b in zip(y, y[steps:])]
    ret_val = [y[a]-y[a-steps] for a in range(steps, len(y)) ]
    #ret_val.extend([y[a+steps]-y[a] for a in range(0, len(y) - steps)  ] )
    y_val = ret_val#[steps:]
    #print(len(X))
    #print(len(ret_val))
    #print(len(y))
    trend.extend([ y[a-steps] for a in range(steps, len(y))])
    return x_val[:len(x_val)-steps], ret_val, trend
from statistics import *

def subtract_mean(y):
    #mean = (float) sum(y)/len(y)
    mean_ = mean(y)
    return [a-mean_ for a in y]



