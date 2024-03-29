# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 13:53:28 2024

@author: lifep
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 13:32:42 2023

@author: CRichard
"""
import os 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tensorpac import Pac 
import mne

epoch_dur = 30 # how many seconds in each epoch
pac_method = 5 # Phase-Locking Value=5, modulation index=2
surrogate_method = 2 # METHOD FOR COMPUTING SURROGATES - Swap amplitude time blocks
norm_method = 4 # normalization method for correction - z-scores
# FOR ALL POSSIBLE SETTINGS, SEE:
#  https://etiennecmb.github.io/tensorpac/generated/tensorpac.Pac.html#tensorpac.Pac

whichDx = 'AUD_this_visit' # ALAB_this_visit ALD_this_visit

read_dir = "D:\\COGA_eec\\"
write_dir = "D:\\COGA_eec\\"
# read_dir = "/ddn/crichard/data/"
#write_dir = "/ddn/crichard/pipeline/processed/"
#read_dir = os.environ['TMPDIR'] + '/input/'
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
chanList_10_20 = ['T8']



mn = []
mx = []

# pacdat = pd.read_csv(read_dir + which_pacdat)
pacdat = pd.read_pickle(read_dir + which_pacdat)

for c in range(0,len(chanList_10_20)):    
    chpac = pacdat[pacdat.channel==chanList_10_20[c]]    
    
    
    for i in range(0,len(chpac)):
        sample_rate = int(chpac.iloc[i].eeg_file_name.split('_')[-1])
        thisFileName = chpac.iloc[i].eeg_file_name
        thisPathFileName = read_dir + 'cleaned_data\\' + thisFileName + '.csv'
        if chpac.iloc[i].AUD_this_visit:
            dx_folder = 'alcoholic\\'
        else:
            dx_folder = 'nonalcoholic\\'

        if os.path.exists(thisPathFileName):
            data = np.loadtxt(thisPathFileName, delimiter=',', skiprows=1)
            # print('Working on ' + thisFileName + ', ' + str(i+1) + ' of ' + str(len(chpac)) + ' files' )
        else:
            continue
        
        time_intervals = list(range(0,len(data),sample_rate*epoch_dur))
        
        for t in range(0,len(time_intervals)-1): 
            start = time_intervals[t]
            end = time_intervals[t+1]
            segment = data[start:end]
            
    
            p = Pac(idpac=(pac_method, surrogate_method, norm_method), 
                    f_pha=(f_pha[0], f_pha[1], 1, 0.1), 
                    f_amp=(f_amp[0], f_amp[1], 1, 0.1),
                    dcomplex='wavelet', width=7, verbose=None)
            
            # Now, extract all of the phases and amplitudes
            phases = p.filter(sample_rate, segment, ftype='phase')
            amplitudes = p.filter(sample_rate, segment, ftype='amplitude')
            xpac = p.fit(phases, amplitudes, n_perm=200, p=0.05, mcp='fdr')
            x = xpac.mean(-1)         
            
            img = sns.heatmap(np.flip(x,0), cmap='Reds',vmin=vmin, vmax=vmax, xticklabels=False,yticklabels=False, cbar=False)
            fig = plt.Axes.get_figure(img)
            
            # FINALLY WE SAVE IT AS A JPG -    THIS WILL BE IMPORTANT FOR RESIZING 
            # THIS IMAGE FOR RESNET-50 USING PIL PACKAGE 
            fig.savefig(write_dir + 'pac_figures_segmented\\' + dx_folder + thisFileName + '_t' + str(t) + '.jpg', bbox_inches='tight')
            plt.close(fig)

            
            mn.append(x.min())
            mx.append(x.max()) 
            
    print(str(min(mn)) + ' to ' + str(max(mx)) + '\n')


# for c in range(0,len(chanList_10_20)):    
#     chpac = pacdat[pacdat.channel==chanList_10_20[c]]    
#     for i in range(0,len(chpac)):
#         sample_rate = int(chpac.iloc[i].eeg_file_name.split('_')[-1])
#         thisFileName = chpac.iloc[i].eeg_file_name
#         thisPathFileName = read_dir + 'cleaned_data/' + thisFileName + '.csv'
#         if chpac.iloc[i].alcoholic:
#             dx_folder = 'alcoholic/'
#         else:
#             dx_folder = 'nonalcoholic/'
#         if os.path.exists(thisPathFileName):
#             data = np.loadtxt(thisPathFileName, delimiter=',', skiprows=1)
#         else:
#             continue
        
#         time_intervals = list(range(0,len(data),sample_rate*epoch_dur))
        
#         for t in range(0,len(time_intervals)-1): 
#             start = time_intervals[t]
#             end = time_intervals[t+1]
#             segment = data[start:end]
            
#             p = Pac(idpac=(pac_method, surrogate_method, norm_method), 
#                     f_pha=(f_pha[0], f_pha[1], 1, 0.1), 
#                     f_amp=(f_amp[0], f_amp[1], 1, 0.1),
#                     dcomplex='wavelet', width=7, verbose=None)
            
#             # Now, extract all of the phases and amplitudes
#             phases = p.filter(sample_rate, segment, ftype='phase')
#             amplitudes = p.filter(sample_rate, segment, ftype='amplitude')
#             xpac = p.fit(phases, amplitudes, n_perm=200, p=0.05, mcp='fdr')
#             x = xpac.mean(-1)
            
#             mn.append(x.min())
#             mx.append(x.max())          
            
#             img = sns.heatmap(np.flip(x,0), cmap='Reds',vmin=vmin, vmax=vmax, xticklabels=False,yticklabels=False, cbar=False)
#             fig = plt.Axes.get_figure(img)
            
#             # FINALLY WE SAVE IT AS A JPG -    THIS WILL BE IMPORTANT FOR RESIZING 
#             # THIS IMAGE FOR RESNET-50 USING PIL PACKAGE 
#             fig.savefig(write_dir + 'pac_figures_segmented/' + dx_folder + thisFileName + '_t' + str(t) + '.jpg', bbox_inches='tight')
#             plt.close(fig)
            
    
#     print(str(min(mn)) + ' to ' + str(max(mx)) + '\n')
