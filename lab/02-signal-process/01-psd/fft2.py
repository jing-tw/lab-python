import matplotlib.pyplot as plt
import numpy as np
from numpy import where
from numpy.fft import fft


N = 1600         # Define the total number of data points 
fs = 800.0        # Define the sampling rate
dt = 1.0 / fs      # Define the sampling interval (sec)
T = N*dt         # Define the total duration of data (sec)
t = np.linspace(0.0, N*dt, N) # start=0, stop = N*T, total sample = N
s = np.sin(50.0 * 2.0*np.pi*t) + 0.5*np.sin(80.0 * 2.0*np.pi*t)

# Calcuating the Spectrum
sf = fft(s) # sf = fft(s-s.mean())  # calcuate the fft

# Calcuating the Power Spectrum
# page: https://mark-kramer.github.io/Case-Studies-Python/03.html
# Sxx = 2 * dt ** 2 / T * (sf * sf.conj())  # Compute spectrum

# slides: https://docs.google.com/presentation/d/1POYr80Tsy-0XmC98zUVn95Qo-u_0kVr01pY2PJq7F_Q/edit#slide=id.g115180020e7_0_17
# Sxx = (sf * sf.conj()) * 1/2 * T   

Sxx = (sf * sf.conj()) * 1.0/2 * 0.5*fs/T

# matlab
# Sxx = (1/(fs*N)) * abs(sf * sf.conj());  # matlab: https://www.mathworks.com/help/signal/ug/power-spectral-density-estimates-using-fft.html
# Sxx[2:len(Sxx)-1] = 2*Sxx[2:len(Sxx)-1] # matlab
# Sxx = 10 * np.log10(Sxx) # matlab

# Show the signal
fig, (ax0, ax1, ax2) = plt.subplots(3, 1)
ax0.set_title('Signal ' +  'T = ' + str(T) + ' (sec)')
ax0.set_xlabel('Time [s]')
ax0.set_ylabel('Voltage [$\mu V$]')
ax0.plot(t, s)

# Determine frequency resolution (df = fNQ / (N//2) ???)
fNQ = fs/ 2.0                       # Determine Nyquist frequency
faxis = np.linspace(0.0, fNQ, N//2) # Construct frequency axis

# Show the spectrum
ax1.set_title('FFT')
ax1.set_xlabel('Frequency [Hz]')
ax1.set_ylabel('Magnitude')
ax1.set_xlim(0,100)
ax1.plot(faxis, np.abs(sf[:N//2]))   # Ignore negative frequencies
# ax1.plot(faxis, 2.0/N * np.abs(sf[:N//2])) // amplitude normalization

# Show the power spectrum
ax2.set_title('PSD')
ax2.set_xlabel('Frequency [Hz]')
ax2.set_ylabel('Power [$\mu V^2$/Hz]')
ax2.set_xlim(0,100)
ax2.plot(faxis, np.abs(Sxx[:N//2]))  # Ignore negative frequencies
# end of test
fig.tight_layout()
plt.show()