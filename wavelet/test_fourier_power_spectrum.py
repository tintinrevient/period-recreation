import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import welch, hanning

x = np.linspace(0, 10, 100001)
dt = x[1] - x[0]
fs = 1 / dt

a1 = 1
f1 = 500

a2 = 10
f2 = 2000

y = a1 * np.sin(2*np.pi*f1*x) + a2 * np.sin(2*np.pi*f2*x)

datos = y

nblock = 1024
overlap = 128
win = hanning(nblock, True)

f, Pxxf = welch(datos, fs, window=win, noverlap=overlap, nfft=nblock, return_onesided=True, detrend=False)

plt.semilogy(f, Pxxf, '-o')

plt.grid()
plt.show()