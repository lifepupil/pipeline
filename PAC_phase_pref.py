# -*- coding: utf-8 -*-
"""
Created on Tue May 20 13:28:42 2025

@author: lifep
"""
import os
import numpy as np
import pandas as pd
from tensorpac import PreferredPhase
from tensorpac.signals import pac_signals_wavelet
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8')

epoch_dur = 30 # HOW MANY SECONDS IN EACH EPOCH
read_dir = "D:\\COGA_eec\\"
write_dir = "D:\\COGA_eec\\"
which_pacdat = 'pacdat_MASTER.pkl'
pacdat = pd.read_pickle(read_dir + which_pacdat)

# sf = 1024.
# n_epochs = 100
# n_times = 2000
# pp = np.pi / 2
# data, time = pac_signals_wavelet(f_pha=6, f_amp=100, n_epochs=n_epochs, sf=sf,
#                                  noise=1, n_times=n_times, pp=pp)

data = np.array([np.zeros(15000)])

chpac = pacdat[pacdat.channel=='FZ']
for i in range(0,len(chpac)):
    sample_rate = int(chpac.iloc[i].eeg_file_name.split('_')[-1])
    thisFileName = chpac.iloc[i].eeg_file_name
#        thisFileName = 'TP8_eec_4_f1_10006013_32_cnt_500'
    
    thisPathFileName = read_dir + 'FZ\\' + thisFileName + '.csv'
#        thisPathFileName = read_dir + sub_dir + '/' + thisFileName + '.csv'
    # thisPathFileName = read_dir + thisFileName + '.csv'

    if os.path.exists(thisPathFileName):
        this_rec = np.loadtxt(thisPathFileName, delimiter=',', skiprows=1)
        print('Working on ' + thisFileName + ', ' + str(i+1) + ' of ' + str(len(chpac)) + ' files' )
    else:
        continue
    
    time_intervals = list(range(0,len(this_rec),sample_rate*epoch_dur))
    
    for t in range(0,len(time_intervals)-1): 
        start = time_intervals[t]
        end = time_intervals[t+1]
        segment = this_rec[start:end]
        data = np.append(data, [segment], axis=0)

    data = data[1:]
    time = np.arange(0,30, 1/(sample_rate*30))
       
    p = PreferredPhase(f_pha=[5, 7], f_amp=(30, 50, 1, 0.1))
    
    # Extract the phase and the amplitude :
    pha = p.filter(sample_rate, data, ftype='phase', n_jobs=1)
    amp = p.filter(sample_rate, data, ftype='amplitude', n_jobs=1)
    
    # Now, compute the PP :
    ampbin, pp, vecbin = p.fit(pha, amp, n_bins=72)
    
    
    # Reshape the PP to be (n_epochs, n_amp) and the amplitude to be
    # (nbins, n_amp, n_epochs). Finally, we take the mean across trials
    pp = np.squeeze(pp).T
    ampbin = np.squeeze(ampbin).mean(-1)
    
    plt.figure(figsize=(20, 35))
    
    # Plot the prefered phase
    plt.subplot(221)
    plt.pcolormesh(p.yvec, np.arange(100), np.rad2deg(pp), cmap='RdBu_r')
    cb = plt.colorbar()
    plt.clim(vmin=-180., vmax=180.)
    plt.axis('tight')
    plt.xlabel('Amplitude frequencies (Hz)')
    plt.ylabel('Epochs')
    plt.title("Single trial PP according to amplitudes.\n100hz amplitudes"
              " are phase locked to 90° (pi/2)")
    cb.set_label('PP (in degrees)')
    
    # Then, we show  the histogram corresponding to an 100hz amplitude :
    idx100 = np.abs(p.yvec - 100.).argmin()
    plt.subplot(222)
    h = plt.hist(pp[:, idx100], color='#ab4642')
    plt.xlim((-np.pi, np.pi))
    plt.xlabel('PP')
    plt.title('PP across trials for the 100hz amplitude')
    plt.xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    plt.gca().set_xticklabels([r"$-\pi$", r"$-\frac{\pi}{2}$", "$0$",
                              r"$\frac{\pi}{2}$", r"$\pi$"])
    
    p.polar(ampbin.T, vecbin, p.yvec, cmap='RdBu_r', interp=.1, subplot=212,
            cblabel='Amplitude bins')
    
    p.show()