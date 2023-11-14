# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 13:32:42 2023

@author: CRichard
"""

import matplotlib.pyplot as plt

import os
import urllib

import numpy as np
from scipy.io import loadmat
from scipy.signal import welch
from scipy.integrate import simps

from tensorpac import Pac, EventRelatedPac, PreferredPhase
from tensorpac.utils import PeakLockedTF, PSD, ITC, BinAmplitude
from tensorpac.signals import pac_signals_wavelet

import pandas as pd
import mne

# filename = os.path.join(os.getcwd(), 'seeg_data_pac.npz')
# if not os.path.isfile(filename):
#     print('Downloading the data')
#     url = "https://www.dropbox.com/s/dn51xh7nyyttf33/seeg_data_pac.npz?dl=1"
#     urllib.request.urlretrieve(url, filename=filename)

# arch = np.load(filename)
# data = arch['data']       # data of a single sEEG contact
# sf = float(arch['sf'])    # sampling frequency
# times = arch['times']     # time vector



# f_pha = 6       # frequency phase for the coupling
# f_amp = 70      # frequency amplitude for the coupling
# n_epochs = 20   # number of trials
# n_times = 4000  # number of time points
sf = 500       # sampling frequency
# data, time = pac_signals_wavelet(sf=sf, f_pha=f_pha, f_amp=f_amp, noise=3.,
#                                  n_epochs=n_epochs, n_times=n_times)

# fn = "D:\\COGA\\data_for_mwt\\CZ_ant_7_l1_40039009_32.cnt_500.csv"
# data = np.array(eeg)

# pth = "C:\\Users\\crichard\\Downloads\\data.txt"

# pth = "D:\\COGA_eec\\\cleaned_data\\CZ_eec_1_a1_10006015_cnt_256.csv"
# pth = "D:\\COGA_eec\\\cleaned_data\\CZ_eec_3_b1_10006015_32_cnt_500.csv"
# pth = "D:\\COGA_eec\\\cleaned_data\\CZ_eec_3_c1_10006015_32_cnt_500.csv"
# pth = "D:\\COGA_eec\\\cleaned_data\\CZ_eec_3_d1_10006015_32_cnt_500.csv"
pth = "D:\\COGA_eec\\\cleaned_data\\CZ_eec_4_e1_10006015_32_cnt_500.csv"
# pth = "D:\\COGA_eec\\\cleaned_data\\CZ_eec_4_f1_10006015_32_cnt_500.csv"

# pth = "D:\\COGA_eec\\\cleaned_data\\CZ_eec_1_a1_10158001_cnt_256.csv"
# pth = "D:\\COGA_eec\\\cleaned_data\\CZ_eec_4_l1_10158001_32_cnt_500.csv"

band_rng = [0.5, 4]

data = np.loadtxt(pth, delimiter=',', skiprows=1)

if 1:
    time = np.arange(data.size)/500
    plt.plot(time, data, lw=1.5,color='k')
    plt.show()
    
    win_length = (2/band_rng[0])*sf
    freqs, psd = welch(data , sf, nperseg=win_length)
    plt.plot(freqs[freqs<80],psd[freqs<80])
    plt.show()

    fres = band_rng[1] - band_rng[0]
    ib = np.logical_and(freqs>=band_rng[0], freqs<band_rng[1])
    bp = simps(psd, dx=fres)

# ch_types = ["eeg"]*data.shape[1]
# data = data.reshape(1, len(data))
# ch_types = ["eeg"]
# info = mne.create_info(ch_names=['Cz'], sfreq=500,ch_types=ch_types)
# raw = mne.io.RawArray(data, info)

# win_length = (2/band_rng[0])*sf
# freqs, psd = welch(data , sf, nperseg=win_length)


# define a :class:`tensorpac.Pac` object and use the MVL as the main method
# for measuring PAC
p = Pac(idpac=(5, 2, 4), f_pha=(1, 8, 1, .2), f_amp=(12, 50, 5, 1),
        dcomplex='wavelet', width=12)

# Now, extract all of the phases and amplitudes
phases = p.filter(sf, data, ftype='phase')
amplitudes = p.filter(sf, data, ftype='amplitude')


plt.figure(figsize=(16, 12))
for i, k in enumerate(range(4)):
    # change the pac method
    p.idpac = (5, k, 4)
    # compute only the pac without filtering
    xpac = p.fit(phases, amplitudes, n_perm=400, p=0.05, mcp='fdr')
    # plot
    title = p.str_surro.replace(' (', '\n(')
    plt.subplot(2, 2, k + 1)
    p.comodulogram(xpac.mean(-1), title=title, cmap='Reds', vmin=0,
                   fz_labels=18, fz_title=20, fz_cblabel=18)

plt.tight_layout()

plt.show()