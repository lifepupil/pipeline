# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 13:32:42 2023

@author: CRichard
"""

import matplotlib.pyplot as plt

# import os
# import urllib

import numpy as np
import pandas as pd
import seaborn as sns

# from scipy.io import loadmat
# from scipy.signal import welch
# from scipy.integrate import simps

from tensorpac import Pac #, EventRelatedPac, PreferredPhase
# from tensorpac.utils import PeakLockedTF, PSD, ITC, BinAmplitude
# from tensorpac.signals import pac_signals_wavelet

# import mne

# filename = os.path.join(os.getcwd(), 'seeg_data_pac.npz')
# if not os.path.isfile(filename):
#     print('Downloading the data')
#     url = "https://www.dropbox.com/s/dn51xh7nyyttf33/seeg_data_pac.npz?dl=1"
#     urllib.request.urlretrieve(url, filename=filename)

# arch = np.load(filename)
# data = arch['data']       # data of a single sEEG contact
# sf = float(arch['sf'])    # sampling frequency
# times = arch['times']     # time vector

read_dir = "D:\\COGA_eec\\"
vmin = -3
vmax = 7

f_pha = [0, 13]       # frequency range phase for the coupling
f_amp = [8, 50]      # frequency range amplitude for the coupling
# n_epochs = 20   # number of trials
# n_times = 4000  # number of time points
# sample_rate = 500       # sampling frequency
# data, time = pac_signals_wavelet(sf=sample_rate, f_pha=f_pha, f_amp=f_amp, noise=3.,
#                                  n_epochs=n_epochs, n_times=n_times)

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

pac_method = 5 # USES Phase-Locking Value (PLV) TO GENERATE PAC VALUES
surrogate_method = 2 # METHOD FOR COMPUTING SURROGATES - Swap amplitude time blocks
norm_method = 4 # normalization method for correction - z-scores
# FOR ALL POSSIBLE SETTINGS, SEE:
#  https://etiennecmb.github.io/tensorpac/generated/tensorpac.Pac.html#tensorpac.Pac

mn = []
mx = []

pacdat = pd.read_csv(read_dir + 'pacdat.csv')

for c in range(0,len(chanList_10_20)):
    chpac = pacdat[pacdat.channel==chanList_10_20[c]]
    
    for i in range(0,len(chpac)):
        sample_rate = int(chpac.iloc[i].eeg_file_name.split('_')[-1])
        thisFileName = chpac.iloc[i].eeg_file_name    
        thisPathFileName = read_dir + 'cleaned_data\\' + thisFileName + '.csv'
        if chpac.iloc[i].alcoholic:
            img_folder = 'alcoholic\\'
        else:
            img_folder = 'nonalcoholic\\'
        print('Working on ' + thisFileName + ', ' + str(i+1) + ' of ' + str(len(chpac)) + ' files' )
        data = np.loadtxt(thisPathFileName, delimiter=',', skiprows=1)
        
        # if 0:
        #     band_rng = [0.5, 4]
        
        #     # GENERATES FIGURE OF EEG SIGNAL
        #     time = np.arange(data.size)/sample_rate
        #     plt.plot(time, data, lw=1.5,color='k')
        #     plt.show()
        #     # GENERATES PSD FIGURE
        #     win_length = (2/band_rng[0])*sample_rate
        #     freqs, psd = welch(data , sample_rate, nperseg=win_length)
        #     plt.plot(freqs[freqs<80],psd[freqs<80])
        #     plt.show()
        
        #     fres = band_rng[1] - band_rng[0]
        #     ib = np.logical_and(freqs>=band_rng[0], freqs<band_rng[1])
        #     bp = simps(psd, dx=fres)
    
        p = Pac(idpac=(pac_method, surrogate_method, norm_method), 
                f_pha=(f_pha[0], f_pha[1], 1, 1), 
                f_amp=(f_amp[0], f_amp[1], 2, 2),
                dcomplex='wavelet', width=7, verbose=None)
        
        # Now, extract all of the phases and amplitudes
        phases = p.filter(sample_rate, data, ftype='phase')
        amplitudes = p.filter(sample_rate, data, ftype='amplitude')
        xpac = p.fit(phases, amplitudes, n_perm=200, p=0.05, mcp='fdr')
        x = xpac.mean(-1)
        
        mn.append(x.min())
        mx.append(x.max())
        print(str(mn[-1]) + ' to ' + str(mx[-1]) + '\n')
        # if mx[-1]>vmax:
        #     print('Make vmax larger than ' + str(mx[-1]))
        #     break
        # if mn[-1]<vmin:
        #     print('Make vmin less than ' + str(mn[-1]))
        #     break
        
        
        # plt.figure(figsize=(16, 12))
        # for i, k in enumerate(range(4)):
        #     # change the pac method
        #     p.idpac = (5, k, 4)
        #     # compute only the pac without filtering
        #     xpac = p.fit(phases, amplitudes, n_perm=200, p=0.05, mcp='fdr')
        #     # plot
        #     title = p.str_surro.replace(' (', '\n(')
        #     plt.subplot(2, 2, k + 1)
        #     p.comodulogram(xpac.mean(-1), title=title, cmap='Reds',
        #                     fz_labels=18, fz_title=20, fz_cblabel=18)
        # plt.tight_layout()
        # plt.show()
        

        
        # sns.heatmap(np.flip(x,0), cmap='Reds')
        img = sns.heatmap(np.flip(x,0), cmap='Reds',vmin=vmin, vmax=vmax, xticklabels=False,yticklabels=False, cbar=False)
        # img = sns.heatmap(np.flip(x,0), cmap='Reds',vmin=vmin, vmax=vmax, xticklabels=True,yticklabels=True, cbar=True)
        fig = plt.Axes.get_figure(img)
        # FINALLY WE SAVE IT AS A JPG -    THIS WILL BE IMPORTANT FOR RESIZING 
        # THIS IMAGE FOR RESNET-50 USING PIL PACKAGE 
        fig.savefig(read_dir + 'pac_figures_new\\' + img_folder + thisFileName + '.jpg', bbox_inches='tight')
        plt.close(fig)
        # title = p.str_surro.replace(' (', '\n(')
        # ch = thisFileName.split('_')[0]
        # vst = thisFileName.split('_')[3]
        # sbj = thisFileName.split('_')[4]
        # aud = img_folder.split('\\')[0]
        # title = ch + ' from ' + sbj + ', visit ' + vst + '\n' + aud
        # p.comodulogram(xpac.mean(-1), title=title, cmap='Reds', vmin=0, fz_labels=14, fz_title=18, fz_cblabel=14)
        # # p.savefig(read_dir + 'pac_figures\\' + thisFileName + '.jpg')
        # del p