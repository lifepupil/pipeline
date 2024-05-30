# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 16:19:05 2024

@author: lifep
"""

import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# TIME POINTS IN A ROW THAT MISS AMPLITUDE CUTOFFS
slip_f_cutoff = 0 # FOR FLATNESS (intervals of below 5uV diff)
slip_n_cutoff = 0 # FOR NOISINESS (intervals continuously above 100uV diff)
flat_threshold = 1e-06 # STANDARD 5 uV
noise_threshold = 0.000100 # STANDARD 100 uV
sub_dir = 'cleaned_data' # cleaned_data FZ
min_dur_flat = 0.0 # MINIMUM DURATION THAT FLAT INTERVAL MUST BE IN SECONDS
min_dur_noise = 0.0 # MINIMUM DURATION THAT NOISE INTERVAL MUST BE IN SECONDS
which_pacdat = 'pacdat_MASTER.pkl'


# csv_dir = "D:\\COGA_eec\\"
# pac_dir = "D:\\COGA_eec\\"
csv_dir = "/ddn/crichard/eeg_csv/"
pac_dir = "/ddn/crichard/eeg_csv/pacdat/"
# csv_dir = os.environ['TMPDIR'] + '/data/'
# pac_dir = os.environ['TMPDIR'] + '/input/'


pacdat = pd.read_pickle(pac_dir + which_pacdat)


if not(any([c=='max_flat_slip'+str(slip_f_cutoff) for c in pacdat.columns])):
    # pacdat.insert(4,'max_flat_slip'+str(slip_f_cutoff), np.ones(len(pacdat))*99999)
    pacdat['max_flat_slip'+str(slip_f_cutoff)] = np.ones(len(pacdat))*99999
    
if not(any([c=='perc_flat_slip'+str(slip_f_cutoff) for c in pacdat.columns])):    
    # pacdat.insert(4,'perc_flat_slip'+str(slip_f_cutoff), np.ones(len(pacdat))*99999)
    pacdat['perc_flat_slip'+str(slip_f_cutoff)] = np.ones(len(pacdat))*99999

if not(any([c=='N_flat_slip'+str(slip_f_cutoff) for c in pacdat.columns])):
    # pacdat.insert(4,'N_flat_slip'+str(slip_f_cutoff), np.ones(len(pacdat))*99999)
    pacdat['N_flat_slip'+str(slip_f_cutoff)] = np.ones(len(pacdat))*99999
    
if not(any([c=='avg_flat_slip'+str(slip_f_cutoff) for c in pacdat.columns])):
    # pacdat.insert(4,'avg_flat_slip'+str(slip_f_cutoff), np.ones(len(pacdat))*99999)
    pacdat['avg_flat_slip'+str(slip_f_cutoff)] = np.ones(len(pacdat))*99999
    
    
if not(any([c=='max_noise_slip'+str(slip_n_cutoff) for c in pacdat.columns])):
    # pacdat.insert(4,'max_noise_slip'+str(slip_n_cutoff), np.ones(len(pacdat))*99999)  
    pacdat['max_noise_slip'+str(slip_n_cutoff)] = np.ones(len(pacdat))*99999
    
if not(any([c=='perc_noise_slip'+str(slip_n_cutoff) for c in pacdat.columns])):
    # pacdat.insert(4,'perc_noise_slip'+str(slip_n_cutoff), np.ones(len(pacdat))*99999) 
    pacdat['perc_noise_slip'+str(slip_n_cutoff)] = np.ones(len(pacdat))*99999
    
if not(any([c=='avg_noise_slip'+str(slip_n_cutoff) for c in pacdat.columns])):
    # pacdat.insert(4,'avg_noise_slip'+str(slip_n_cutoff), np.ones(len(pacdat))*99999) 
    pacdat['avg_noise_slip'+str(slip_n_cutoff)] = np.ones(len(pacdat))*99999
    
if not(any([c=='N_noise_slip'+str(slip_n_cutoff) for c in pacdat.columns])):
    # pacdat.insert(4,'N_noise_slip'+str(slip_n_cutoff), np.ones(len(pacdat))*99999) 
    pacdat['N_noise_slip'+str(slip_n_cutoff)] = np.ones(len(pacdat))*99999
    
    
# pacdat.insert(2,'max_flat', np.zeros(len(pacdat)))
# pacdat.insert(11,'sample_rate', np.zeros(len(pacdat)))
# pacdat.insert(2,'max_noise', np.zeros(len(pacdat)))
# git commit -am "added more data plots, commented out, and slip = 1"

        
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
        
        thisPathFileName = csv_dir + sub_dir + '\\' + thisFileName + '.csv'        
        # thisPathFileName = csv_dir + sub_dir + '/' + thisFileName + '.csv'        
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
        uv_diff = np.ones(len(uv_diff))*(2/1000000)
        # len_good = len(uv_diff[((uv_diff>=0.000005) & (uv_diff<=0.000100))])
        # if len_good == len(uv_diff):
        #     # INSERT CODE TO ENTER APPROPRIATE VALUES INTO pacdat 
        #     # BEFORE SKIPPING THIS FILE BECAUSE NO POINT IN PROCESSING 
        #     # IF ABOVE IS TRUE
        #     print('WHAT THE WHAT??!! THERE\' S ACTUALLY A PERFECT SIGNAL??n\ ')
        #     pacdat.at[r,'perc_flat_slip'+str(slip_f_cutoff)] = 0            
        #     pacdat.at[r,'max_flat_slip'+str(slip_f_cutoff)] = 0
            
        #     continue
        
        
        # CURRENT LENGTH OF LOW AMPLITUDE (FLAT) INTERVAL (CONSECUTIVE SAMPLES MEETING FLAT CRITERION)
        flat_interval = 0 
        # ARRAY CONTAINING LENGTHS OF ALL FLAT AMPLITUDE INTERVALS IN SIGNAL
        flat_intervals = np.array([0])
        # KEEPS TRACK OF HOW MANY SAMPLES IN A ROW HAVE NOT BEEN FLAT
        slip_f = 0
        
        
        # CURRENT LENGTH OF HIGH AMPLITUDE INTERVAL (CONSECUTIVE SAMPLES MEETING NOISE CRITERION)
        noise_interval = 0 
        # ARRAY CONTAINING LENGTHS OF ALL HIGH AMPLITUDE INTERVALS IN SIGNAL
        noise_intervals = np.array([0])
        # KEEPS TRACK OF HOW MANY SAMPLES IN A ROW HAVE NOT BEEN NOISY
        slip_n = 0
        
        
        for t in range(0,len(uv_diff)-1):
            this_t = uv_diff[t]
            
            if this_t <= flat_threshold:
                flat_interval += 1 
                if slip_f>0:
                    slip_f -= 1
            # FLAT INTERVAL MUST BE GREATER THAN 4 SAMPLES IN DURATION
            elif flat_interval > sample_rate*min_dur_flat:
                if slip_f == slip_f_cutoff:
                    flat_intervals = np.append(flat_intervals,flat_interval)
                    flat_interval = 0
                else:
                    slip_f += 1
                    
            if this_t >= noise_threshold:
                noise_interval += 1 
                if slip_n>0:
                    slip_n -= 1
            # NOISY INTERVAL MUST BE GREATER THAN 4 SAMPLES IN DURATION
            elif noise_interval > sample_rate*min_dur_noise:
                if slip_n == slip_n_cutoff:
                    noise_intervals = np.append(noise_intervals,noise_interval)
                    noise_interval = 0
                else:
                    slip_n += 1  

        if len(flat_intervals)>1:                    
            flat_intervals = flat_intervals[1:] # WE WANT TO GET RID OF THE LEADING ZERO
        perc_flat = 100*(flat_intervals.sum()/len(uv_diff))
        pacdat.at[r,'perc_flat_slip'+str(slip_f_cutoff)] = perc_flat
        pacdat.at[r,'N_flat_slip'+str(slip_f_cutoff)] = len(flat_intervals)
        
        flat_intervals = flat_intervals/sample_rate
        pacdat.at[r,'max_flat_slip'+str(slip_f_cutoff)] = max(flat_intervals)
        pacdat.at[r,'avg_flat_slip'+str(slip_f_cutoff)] = np.mean(flat_intervals)
        pacdat.at[r,'N_flat_slip'+str(slip_f_cutoff)] = len(flat_intervals)
        
        
        if len(noise_intervals)>1:
            noise_intervals = noise_intervals[1:] # WE WANT TO GET RID OF THE LEADING ZERO
        perc_noise = 100*(noise_intervals.sum()/len(uv_diff))
        pacdat.at[r,'perc_noise_slip'+str(slip_n_cutoff)] = perc_noise
        pacdat.at[r,'N_noise_slip'+str(slip_n_cutoff)] = len(noise_intervals)
        
        noise_intervals = noise_intervals/sample_rate
        pacdat.at[r,'max_noise_slip'+str(slip_n_cutoff)] = max(noise_intervals)
        pacdat.at[r,'avg_noise_slip'+str(slip_n_cutoff)] = np.mean(noise_intervals)
        pacdat.at[r,'N_noise_slip'+str(slip_n_cutoff)] = len(noise_intervals)
        
pacdat.to_pickle(pac_dir + which_pacdat)

# fz = pacdat[(pacdat.channel=='FZ') & (pacdat.max_flat>0)]
# fz = pacdat[(pacdat.channel=='FZ') & (pacdat.max_noise>0)]
# fz = pacdat[(pacdat.channel=='FZ') & (pacdat.max_noise>0) & (pacdat.max_flat>0)]
# fz = pacdat[(pacdat.channel=='FZ')]
# fz = pd_filtered[(pd_filtered.channel=='FZ')]
# logy = False
# logy = True
# fz[['max_flat']].plot.hist(bins=50,xlabel='seconds', title='Duration of maximum flat interval\n(by EEG channel)',logy=logy)
# fz[['max_noise']].plot.hist(bins=10,xlabel='seconds', title='Duration of maximum noise interval\n(by EEG channel)',logy=logy)
# fz[['max_flat_slip0']].plot.hist(bins=50,xlabel='seconds', title='Max duration of flat interval with slip0\n(by EEG channel from eec)',logy=logy)
# fz[['max_flat_slip1']].plot.hist(bins=50,xlabel='seconds', title='Max duration of flat interval with slip1\n(by EEG channel from eec)',logy=logy)
# fz[['avg_flat_slip0']].plot.hist(bins=50,xlabel='seconds', title='Avg duration of flat interval with slip0\n(by EEG channel from eec)',logy=logy)
# fz[['perc_flat_slip0']].plot.hist(bins=50,xlabel='percentage', title='Percent flat interval with slip0\n(by EEG channel from eec)',logy=logy)
# fz[['perc_flat_slip1']].plot.hist(bins=50,xlabel='percentage', title='Percent flat interval with slip1\n(by EEG channel from eec)',logy=logy)
# fz[['max_noise_slip0']].plot.hist(bins=50,xlabel='seconds', title='Max duration of noise interval with slip0\n(by EEG channel from eec)',logy=logy)
# fz[['max_noise_slip1']].plot.hist(bins=50,xlabel='seconds', title='Max duration of noise interval with slip1\n(by EEG channel from eec)',logy=logy)
# fz[['avg_noise_slip0']].plot.hist(bins=50,xlabel='seconds', title='Avg duration of noise interval with slip0\n(by EEG channel from eec)',logy=logy)
# fz[['avg_noise_slip1']].plot.hist(bins=50,xlabel='seconds', title='Avg duration of noise interval with slip1\n(by EEG channel from eec)',logy=logy)
# fz[['perc_noise_slip0']].plot.hist(bins=50,xlabel='percentage', title='Percent noise interval with slip0\n(by EEG channel from eec)',logy=logy)
# fz[['perc_noise_slip1']].plot.hist(bins=50,xlabel='percentage', title='Percent noise interval with slip1\n(by EEG channel from eec)',logy=logy)


# ss = list(set(fz.site))
# for i in range(0,len(ss)): print(ss[i]+' '+ str(len(fz[(fz.max_flat_slip0==0) & (fz.site==ss[i])])))

