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

from tensorpac import Pac, EventRelatedPac, PreferredPhase
from tensorpac.utils import PeakLockedTF, PSD, ITC, BinAmplitude
from tensorpac.signals import pac_signals_wavelet

import matplotlib.pyplot as plt


# filename = os.path.join(os.getcwd(), 'seeg_data_pac.npz')
# if not os.path.isfile(filename):
#     print('Downloading the data')
#     url = "https://www.dropbox.com/s/dn51xh7nyyttf33/seeg_data_pac.npz?dl=1"
#     urllib.request.urlretrieve(url, filename=filename)

# arch = np.load(filename)
# data = arch['data']       # data of a single sEEG contact
# sf = float(arch['sf'])    # sampling frequency
# times = arch['times']     # time vector



f_pha = 6       # frequency phase for the coupling
f_amp = 70      # frequency amplitude for the coupling
n_epochs = 20   # number of trials
n_times = 4000  # number of time points
sf = 512.       # sampling frequency
data, time = pac_signals_wavelet(sf=sf, f_pha=f_pha, f_amp=f_amp, noise=3.,
                                 n_epochs=n_epochs, n_times=n_times)


# define a :class:`tensorpac.Pac` object and use the MVL as the main method
# for measuring PAC
p = Pac(idpac=(1, 0, 0), f_pha=(3, 10, 1, .2), f_amp=(50, 90, 5, 1),
        dcomplex='wavelet', width=12)

# Now, extract all of the phases and amplitudes
phases = p.filter(sf, data, ftype='phase')
amplitudes = p.filter(sf, data, ftype='amplitude')


plt.figure(figsize=(16, 12))
for i, k in enumerate(range(4)):
    # change the pac method
    p.idpac = (5, k, 1)
    # compute only the pac without filtering
    xpac = p.fit(phases, amplitudes, n_perm=20)
    # plot
    title = p.str_surro.replace(' (', '\n(')
    plt.subplot(2, 2, k + 1)
    p.comodulogram(xpac.mean(-1), title=title, cmap='Reds', vmin=0,
                   fz_labels=18, fz_title=20, fz_cblabel=18)

plt.tight_layout()

plt.show()