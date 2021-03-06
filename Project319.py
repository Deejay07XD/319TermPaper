from scipy import signal, fft
import matplotlib.pyplot as plt
import numpy as np

# read array

file = r'C:\Users\Adity\Downloads\X_test.npy'
a = np.load(file, mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')

a_transposed = np.transpose(a[0])

#now lets prepare something for O/P

out = np.empty(shape = [6,155])
Final_out = np.empty(shape = [5850,155,6])
Ftr = np.empty(shape = [5850,9,6])

#process the data
for y in range(5850):
    for x in range(6):
        # plot raw data after taking average of 6 elements of a row for 155 rows

        t = np.linspace(0, 1, 155, False)
        sig = a_transposed[x] # in order to filter a different data set n change a[0] to a[n]
        #fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        #ax1.plot(t, sig)
        #ax1.set_title('before filter')
        #ax1.axis([0, 1,-1.5,3])

        # apply filter (sampling frequency is upto 2hz, we're interested in up to 0.4hz where power is concentrated, hence value of wc = 0.5/2 = 0.25)

        sos = signal.butter(10, 0.12903225806, 'lp', output='sos')
        filtered = signal.sosfilt(sos, sig)
        transformed = fft.fft(filtered)
        transformed[72:] = 0
        #ax2.plot(t, filtered)
        #ax2.set_title('After 2hz hp filter with sampling frequency =1000hz')
        #ax2.axis([0, 1, -1.5, 3])
        #ax2.set_xlabel('Time [seconds]')
        #plt.tight_layout()
        #plt.show()

        out[x] = transformed

        #   Feature Extraction

        Ftr[y,0,x] = sum(np.square(transformed))
        Ftr[y,2,x] = sum(sig)
        Ftr[y,3,x] = np.mean(sig)
        Ftr[y,6,x] = sum(abs(sig))
        Ftr[y,7,x] = np.mean(abs(sig))
        
               
        #print(sig.shape)
        #print(out.shape)
        #print(out[x].shape)

        #print(filtered)
        #print(out[x])
        # this code displays graphs for visualization

        # remember to change the paths of input 

    # More Feature extraction

    Ftr[y,4,0:3] =(Ftr[y,3,0] + Ftr[y,3,1] + Ftr[y,3,2])/3
    Ftr[y,4,3:6] = (Ftr[y,3,3] + Ftr[y,3,4] + Ftr[y,3,5])/3

    Ftr[y,5,0] = Ftr[y,3,1] - Ftr[y,3,0]
    Ftr[y,5,1] = Ftr[y,3,2] - Ftr[y,3,0]
    Ftr[y,5,2] = Ftr[y,3,2] - Ftr[y,3,1]
    Ftr[y,5,3] = Ftr[y,3,4] - Ftr[y,3,3]
    Ftr[y,5,4] = Ftr[y,3,5] - Ftr[y,3,3]
    Ftr[y,5,5] = Ftr[y,3,5] - Ftr[y,3,4] 

    Ftr[y,8,0:3] = (Ftr[y,6,0] + Ftr[y,6,1] + Ftr[y,6,2])
    Ftr[y,8,3:6] = (Ftr[y,6,3] + Ftr[y,6,4] + Ftr[y,6,5])

    
    op = np.transpose(out)
    #print(op)
    Final_out[y] = op
    #print(Final_out[0])
    
np.save("X_Test_Processed.npy",Final_out)
np.save("Feature.npy",Ftr)

