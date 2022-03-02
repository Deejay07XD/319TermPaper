from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

# read array

file = r'C:\Users\Adity\Downloads\X_test.npy'
a = np.load(file, mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')

# plot raw data after taking average of 6 elements of a row for 155 rows

t = np.linspace(0, 1, 155, False)
sig = np.average(a[0],axis=1) # in order to filter a different data set n change a[0] to a[n]
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(t, sig)
ax1.set_title('before filter')
ax1.axis([0, 1,-1,1])

# apply filter (sampling frequency is upto 2hz, we're interested in up to 0.4hz where power is concentrated, hence value of wc = 0.5/2 = 0.25)

sos = signal.butter(10, 0.25, 'lp', output='sos')
filtered = signal.sosfilt(sos, sig)
ax2.plot(t, filtered)
ax2.set_title('After 2hz hp filter with sampling frequency =1000hz')
ax2.axis([0, 1, -1, 1])
ax2.set_xlabel('Time [seconds]')
plt.tight_layout()
plt.show()

# this code displays graphs for visualization

# remember to change the paths of input 
