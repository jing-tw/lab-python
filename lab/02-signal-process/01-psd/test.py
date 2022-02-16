# psd: https://matplotlib.org/stable/gallery/lines_bars_and_markers/psd_demo.html
# fft: https://matplotlib.org/stable/gallery/lines_bars_and_markers/spectrum_demo.html?highlight=fft
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec

sinefreq = 100 #Hz

# Fixing random state for reproducibility
np.random.seed(19680801)

dt = 0.002
t = np.arange(0, 10, dt)
nse = np.random.randn(len(t))
r = np.exp(-t / 0.05)

cnse = np.convolve(nse, r) * dt
cnse = cnse[:len(t)]
s = 0.1 * np.sin(2 * np.pi * t * sinefreq) + cnse

fig, (ax0, ax1, ax2) = plt.subplots(3, 1)
ax0.set_title('original')
ax0.plot(t, s)
ax1.set_title('PSD')
ax1.psd(s, 512, 1 / dt)
ax2.set_title('Log. Magnitude Spectrum')
ax2.magnitude_spectrum(s, 1/dt)
plt.show()