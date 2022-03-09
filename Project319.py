from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

# read array

file = r'C:\Users\Adity\Downloads\X_train.npy'
a = np.load(file, mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')

a_transposed = np.transpose(a[0])

#now lets prepare something for O/P

out = np.empty(shape = [6,155])
Final_out = np.empty(shape = [15600,155,6])

#process the data
for y in range(15600):
    for x in range(6):
        # plot raw data after taking average of 6 elements of a row for 155 rows

        t = np.linspace(0, 1, 155, False)
        sig = a_transposed[x] # in order to filter a different data set n change a[0] to a[n]
        #fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        #ax1.plot(t, sig)
        #ax1.set_title('before filter')
        #ax1.axis([0, 1,-1.5,3])

        # apply filter (sampling frequency is upto 2hz, we're interested in up to 0.4hz where power is concentrated, hence value of wc = 0.5/2 = 0.25)

        sos = signal.butter(10, 0.25, 'lp', output='sos')
        filtered = signal.sosfilt(sos, sig)
        #ax2.plot(t, filtered)
        #ax2.set_title('After 2hz hp filter with sampling frequency =1000hz')
        #ax2.axis([0, 1, -1.5, 3])
        #ax2.set_xlabel('Time [seconds]')
        #plt.tight_layout()
        #plt.show()

        out[x] = filtered
        #print(sig.shape)
        #print(out.shape)
        #print(out[x].shape)

        #print(filtered)
        #print(out[x])
        # this code displays graphs for visualization

        # remember to change the paths of input 

    op = np.transpose(out)
    #print(op)
    Final_out[y] = op
    #print(Final_out[0])
    
np.save("X_Train_Processed.npy",Final_out)
