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
slip_f_cutoff = 1 # FOR FLATNESS (intervals of below 5uV diff)
slip_n_cutoff = 1 # FOR NOISINESS (intervals continuously above 100uV diff)
flat_threshold = 0.000005 # STANDARD 5 uV
noise_threshold = 0.000100 # STANDARD 100 uV

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
# pacdat.insert(4,'max_noise_slip'+str(slip_n_cutoff), np.zeros(len(pacdat))) 
pacdat.insert(4,'perc_noise_slip'+str(slip_n_cutoff), np.ones(len(pacdat))*99999) 
pacdat.insert(4,'max_flat_slip'+str(slip_n_cutoff), np.ones(len(pacdat))*99999)
pacdat.insert(4,'perc_flat_slip'+str(slip_n_cutoff), np.ones(len(pacdat))*99999)
# pacdat.insert(2,'max_flat', np.zeros(len(pacdat)))

ftot = str(len(pacdat[pacdat.channel=='FZ']))
fcnt = 0

# ri = 493610
ri = 0
for r in range(ri,len(pacdat)):
    # print('Working on ' + pacdat.iloc[r].eeg_file_name)
    
    if pacdat.iloc[r].channel=='FZ':
        sample_rate = int(pacdat.iloc[r].eeg_file_name.split('_')[-1])
        pacdat.at[r,'sample_rate'] = sample_rate
        thisFileName = pacdat.iloc[r].eeg_file_name
        
        thisPathFileName = read_dir + sub_dir + '\\' + thisFileName + '.csv'        
        if os.path.exists(thisPathFileName):
            fcnt += 1 
            eeg_signal = np.loadtxt(thisPathFileName, delimiter=',', skiprows=1)
            print('Working on ' + thisFileName + ', ' + str(fcnt) + ' of ' + ftot + ' files (' + str(r+1) + ' of ' + str(len(pacdat)) + ')')

        else:
            continue
        
        # CALCULATE ABSOLUTE CHANGE IN VOLTAGE BETWEEN ADJACENT SAMPLES IN EEG SIGNAL
        seg_a = np.append(eeg_signal,0)            
        seg_b = np.insert(eeg_signal,0,0)
        uv_diff = abs(seg_a - seg_b)

        # CURRENT LENGTH OF LOW AMPLITUDE (FLAT) INTERVAL (CONSECUTIVE SAMPLES MEETING FLAT CRITERION)
        flat_interval = 0 
        # ARRAY CONTAINING LENGTHS OF ALL FLAT AMPLITUDE INTERVALS IN SIGNAL
        flat_intervals = np.array([0])
        
        # CURRENT LENGTH OF HIGH AMPLITUDE INTERVAL (CONSECUTIVE SAMPLES MEETING NOISE CRITERION)
        noise_interval = 0 
        # ARRAY CONTAINING LENGTHS OF ALL HIGH AMPLITUDE INTERVALS IN SIGNAL
        noise_intervals = np.array([0])
        
        # KEEPS TRACK OF HOW MANY SAMPLES IN A ROW HAVE NOT BEEN FLAT
        slip_f = 0
        # KEEPS TRACK OF HOW MANY SAMPLES IN A ROW HAVE NOT BEEN NOISY
        slip_n = 0
        
        for t in range(0,len(uv_diff)-1):
            this_t = uv_diff[t]
            
            if this_t<= flat_threshold:
                flat_interval += 1 
                if slip_f>0:
                    slip_f -= 1
            elif flat_interval>0:
                if slip_f == slip_f_cutoff:
                    flat_intervals = np.append(flat_intervals,flat_interval)
                    flat_interval = 0
                else:
                    slip_f += 1
                    
            if this_t>= noise_threshold:
                noise_interval += 1 
                if slip_n>0:
                    slip_n -= 1
            elif noise_interval>0:
                if slip_n == slip_n_cutoff:
                    noise_intervals = np.append(noise_intervals,noise_interval)
                    noise_interval = 0
                else:
                    slip_n += 1  
                    
        flat_intervals = flat_intervals/sample_rate
        pacdat.at[r,'max_flat_slip'+str(slip_f_cutoff)] = max(flat_intervals)
        pacdat.at[r,'perc_flat_slip'+str(slip_f_cutoff)] = (sum(noise_intervals)/len(uv_diff))*100

        noise_intervals = noise_intervals/sample_rate
        # pacdat.at[r,'max_noise_slip'+str(slip_n_cutoff)] = max(noise_intervals)
        pacdat.at[r,'perc_noise_slip'+str(slip_n_cutoff)] = max(noise_intervals)
        
pacdat.to_pickle(read_dir + which_pacdat)

# fz = pacdat[(pacdat.channel=='FZ') & (pacdat.max_flat>0)]
# fz = pacdat[(pacdat.channel=='FZ') & (pacdat.max_noise>0)]
# fz = pacdat[(pacdat.channel=='FZ') & (pacdat.max_noise>0) & (pacdat.max_flat>0)]
fz = pacdat[(pacdat.channel=='FZ')]
# fz[['max_flat']].plot.hist(bins=10,xlabel='seconds', title='Duration of maximum flat interval\n(by EEG channel)',logy=True)
# fz[['max_noise']].plot.hist(bins=10,xlabel='seconds', title='Duration of maximum noise interval\n(by EEG channel)',logy=True)
fz[['max_noise_slip1']].plot.hist(bins=10,xlabel='seconds', title='Duration of maximum noise interval with slip1\n(by EEG channel from eec)',logy=True)