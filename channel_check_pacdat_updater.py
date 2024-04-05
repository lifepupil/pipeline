# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 16:19:05 2024

@author: lifep
"""

import os 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

epoch_dur = 30 # how many seconds in each epoch
# TIME POINTS IN A ROW THAT MISS AMPLITUDE CUTOFFS
slip_f_cutoff = 0 # FOR FLATNESS (intervals of below 5uV diff)
slip_n_cutoff = 0 # FOR NOISINESS (intervals continuously above 100uV diff)

read_dir = "D:\\COGA_eec\\"
write_dir = "D:\\COGA_eec\\"
sub_dir = 'cleaned_data'

#read_dir = "/ddn/crichard/eeg_csv/"
#write_dir = "/ddn/crichard/pipeline/processed/"
# read_dir = os.environ['TMPDIR'] + '/input/'
# write_dir = os.environ['TMPDIR'] + '/results/'
which_pacdat = 'pacdat_MASTER.pkl'


pacdat = pd.read_pickle(read_dir + which_pacdat)
# pacdat.insert(11,'sample_rate', np.zeros(len(pacdat)))
# pacdat.insert(2,'max_noise', np.zeros(len(pacdat)))
# pacdat.insert(2,'max_flat', np.zeros(len(pacdat)))

ri = 493610
for r in range(ri,len(pacdat)):
    
    
    if pacdat.iloc[r].channel=='FZ':
        sample_rate = int(pacdat.iloc[r].eeg_file_name.split('_')[-1])
        pacdat.at[r,'sample_rate'] = sample_rate
        thisFileName = pacdat.iloc[r].eeg_file_name
        
        thisPathFileName = read_dir + sub_dir + '\\' + thisFileName + '.csv'        
        if os.path.exists(thisPathFileName):
            eeg_signal = np.loadtxt(thisPathFileName, delimiter=',', skiprows=1)
            print('Working on ' + thisFileName + ', ' + str(r+1) + ' of ' + str(len(pacdat)) + ' files' )
        else:
            continue
        
        seg_a = np.append(eeg_signal,0)            
        seg_b = np.insert(eeg_signal,0,0)
        uv_diff = abs(seg_a - seg_b)

        flat_interval = 0 
        flat_intervals = np.array([0])
        
        noise_interval = 0 
        noise_intervals = np.array([0])
        
        slip_f = 0
        slip_n = 0
        
        for t in range(0,len(uv_diff)-1):
            this_t = uv_diff[t]
            
            if this_t<=0.000005:
                flat_interval += 1 
                if slip_f>0:
                    slip_f -= 1
            elif flat_interval>0:
                if slip_f==slip_f_cutoff:
                    flat_intervals = np.append(flat_intervals,flat_interval)
                    flat_interval = 0
                else:
                    slip_f += 1
                    
            if this_t>0.000100:
                noise_interval += 1 
                if slip_n>0:
                    slip_n -= 1
            elif noise_interval>0:
                if slip_n==slip_n_cutoff:
                    noise_intervals = np.append(noise_intervals,noise_interval)
                    noise_interval = 0
                else:
                    slip_n += 1  
                    
        flat_intervals = flat_intervals/sample_rate
        noise_intervals = noise_intervals/sample_rate
        pacdat.at[r,'max_flat'] = max(flat_intervals)
        pacdat.at[r,'max_noise'] = max(noise_intervals)
pacdat.to_pickle(read_dir + which_pacdat)

# fz = pacdat[(pacdat.channel=='FZ') & (pacdat.max_flat>0)]
# fz = pacdat[(pacdat.channel=='FZ') & (pacdat.max_noise>0)]
# fz = pacdat[(pacdat.channel=='FZ') & (pacdat.max_noise>0) & (pacdat.max_flat>0)]
# fz = pacdat[(pacdat.channel=='FZ')]
# fz[['max_flat']].plot.hist(bins=10,xlabel='seconds', title='Duration of maximum flat interval\n(by EEG channel)',logy=True)
# fz[['max_noise']].plot.hist(bins=10,xlabel='seconds', title='Duration of maximum noise interval\n(by EEG channel)',logy=True)