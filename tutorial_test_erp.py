# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 17:41:16 2025

@author: lifep
"""

import os
import urllib

import numpy as np
from scipy.io import loadmat

from tensorpac import Pac, EventRelatedPac, PreferredPhase
from tensorpac.utils import PeakLockedTF, PSD, ITC, BinAmplitude

import matplotlib.pyplot as plt

filename = os.path.join(os.getcwd(), 'seeg_data_pac.npz')
if not os.path.isfile(filename):
    print('Downloading the data')
    url = "https://www.dropbox.com/s/dn51xh7nyyttf33/seeg_data_pac.npz?dl=1"
    urllib.request.urlretrieve(url, filename=filename)

arch = np.load(filename)
data = arch['data']       # data of a single sEEG contact
sf = float(arch['sf'])    # sampling frequency
times = arch['times']     # time vector

print(f"DATA: (n_trials, n_times)={data.shape}; SAMPLING FREQUENCY={sf}Hz; "f"TIME VECTOR: n_times={len(times)}")


rp_obj = EventRelatedPac(f_pha=[8, 12], f_amp=(30, 160, 30, 2), dcomplex='wavelet', width=5) 
erpac = rp_obj.filterfit(sf, data, method='gc', smooth=100)