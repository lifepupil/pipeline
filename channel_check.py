# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 16:19:05 2024

@author: lifep
"""

import os 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# from tensorpac import Pac #, EventRelatedPac, PreferredPhase
# from tensorpac.utils import PeakLockedTF, PSD, ITC, BinAmplitude
# from tensorpac.signals import pac_signals_wavelet
# import mne
# import pickle5 as pickle

epoch_dur = 30 # how many seconds in each epoch
pac_method = 5 # Phase-Locking Value=5, modulation index=2
surrogate_method = 2 # METHOD FOR COMPUTING SURROGATES - Swap amplitude time blocks
norm_method = 4 # normalization method for correction - z-scores
# FOR ALL POSSIBLE SETTINGS, SEE:
#  https://etiennecmb.github.io/tensorpac/generated/tensorpac.Pac.html#tensorpac.Pac

read_dir = "D:\\COGA_eec\\"
write_dir = "D:\\COGA_eec\\"
sub_dir = 'cleaned_data'

#read_dir = "/ddn/crichard/eeg_csv/"
#write_dir = "/ddn/crichard/pipeline/processed/"
# read_dir = os.environ['TMPDIR'] + '/input/'
# write_dir = os.environ['TMPDIR'] + '/results/'
which_pacdat = 'pacdat_cutoffs_flat_25_excessnoise_25.pkl'
vmin = -3
vmax = 7

f_pha = [0, 13]       # frequency range phase for the coupling
f_amp = [4, 50]      # frequency range amplitude for the coupling

#print('TEST TEST')

# 10-20 CHANNEL LIST 
chanList_10_20 = [
        'FZ',
        'CZ',
        'PZ',
        'OZ',
        'C3',
        'C4',
        'F3',
        'F4',
        'F7',
        'F8',
        'O1',
        'O2',
        'P3',
        'P4',
        'T3',
        'T4',
        'T5',
        'T6',
        'FP1',
        'FP2']
chanList_10_20 = ['FZ']



mn = []
mx = []

# pacdat = pd.read_csv(read_dir + which_pacdat)
pacdat = pd.read_pickle(read_dir + which_pacdat)

for c in range(0,len(chanList_10_20)):
#for c in range(0,1):
    
    chpac = pacdat[pacdat.channel==chanList_10_20[c]]
#    chpac = pacdat[pacdat.channel=='TP8']
    
    for i in range(0,len(chpac)):
#    for i in range(0,1):
        sample_rate = int(chpac.iloc[i].eeg_file_name.split('_')[-1])
        thisFileName = chpac.iloc[i].eeg_file_name
#        thisFileName = 'TP8_eec_4_f1_10006013_32_cnt_500'
        
        thisPathFileName = read_dir + sub_dir + '\\' + thisFileName + '.csv'        
#        print(thisPathFileName)
        if os.path.exists(thisPathFileName):
            eeg_signal = np.loadtxt(thisPathFileName, delimiter=',', skiprows=1)
            print('Working on ' + thisFileName + ', ' + str(i+1) + ' of ' + str(len(chpac)) + ' files' )
        else:
            continue


#        if chpac.iloc[i].AUD_this_visit:
            # dx_folder = 'alcoholic\\'
#            dx_folder = 'alcoholic/'
#        else:
            # dx_folder = 'nonalcoholic\\'
#            dx_folder = 'nonalcoholic/'

        
        seg_a = np.append(eeg_signal,0)            
        seg_b = np.insert(eeg_signal,0,0)
        uv_diff = abs(seg_a - seg_b)
        
        # perc_high_amp = len(uv_diff[uv_diff>0.000100])/len(uv_diff)
        # perc_flat_amp = len(uv_diff[uv_diff>0.000005])/len(uv_diff)
        # print('channel -- percent high amplitude = ' + str(perc_high_amp) + ', percent flat amplitude = ' + str(perc_flat_amp))
        flat_interval = 0 
        flat_intervals = np.array(0)
        slip = 0
        slip_cutoff = 0
        
        # plt.plot(eeg_signal)
        # plt.plot(uv_diff)
        
        for t in range(0,len(uv_diff)-1):
            this_t = uv_diff[t]
            if this_t<=0.000005:
                flat_interval += 1 
                if slip>0:
                    slip -= 1
                    
            elif flat_interval>0:
                if slip==slip_cutoff:
                    flat_intervals = np.append(flat_intervals,flat_interval)
                    flat_interval = 0
                else:
                    slip += 1 
        flat_intervals = flat_intervals/sample_rate
        # # plt.plot(flat_intervals[t-5:t+5])
        # # fig, (ax1, ax2) = plt.subplots(2)
        # # ax1.plot(np.arange(0,len(uv_diff),1),uv_diff)
        # # ax2.plot(np.arange(0,len(flat_intervals),1),flat_intervals)
        # plt.plot(flat_intervals)
        max_flat = np.array(0)
        
        time_intervals = list(range(0,len(eeg_signal),sample_rate*epoch_dur))

        for ti in range(0,len(time_intervals)-1): 
            start = time_intervals[ti]
            end = time_intervals[ti+1]
            eeg_segment = eeg_signal[start:end]

            seg_a = np.append(eeg_segment,0)            
            seg_b = np.insert(eeg_segment,0,0)
            uv_diff = abs(seg_a - seg_b)
            
            # perc_high_amp = len(uv_diff[uv_diff>0.000100])/len(uv_diff)
            # perc_flat_amp = len(uv_diff[uv_diff>0.000005])/len(uv_diff)
            # print('segment -- percent high amplitude = ' + str(perc_high_amp) + ', percent flat amplitude = ' + str(perc_flat_amp))

            flat_interval = 0 
            flat_intervals = np.array(0)
            slip = 0
            
            for t in range(0,len(uv_diff)-1):
                this_t = uv_diff[t]
                if this_t<=0.000010:
                # if this_t>0.000100:
                    flat_interval += 1 
                    if slip>0:
                        slip -= 1
                        
                elif flat_interval>0:
                    if slip==slip_cutoff:
                        flat_intervals = np.append(flat_intervals,flat_interval)
                        flat_interval = 0
                    else:
                        slip += 1 
            # flat_intervals = flat_intervals/sample_rate
            # plt.plot(flat_intervals)
            if np.asarray(flat_intervals).size==1:
                flat_intervals = [0]
            max_flat = np.append(max_flat,max(flat_intervals))


        print(str(i))
        max_flat = max_flat/sample_rate
        plt.plot(max_flat)
        plt.plot(eeg_signal)
        ii = 0
        plt.plot(eeg_signal[time_intervals[ii]:time_intervals[ii+1]])
# with open('min_pac.pkl', 'wb') as f:
#     pickle.dump(mn, f)
# with open('max_pac.pkl', 'wb') as f:
#     pickle.dump(mx, f)
