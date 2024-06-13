"""
==============
Long-range PAC
==============

One thing you may have noticed, both the fit() and filterfit() methods take as
inputs data and again data. The reason is that the first data refer to the
signal to extract the phase (xpha) and the second, the amplitude (xamp).
If you want to extract local coupling (i.e. on a source/electrode) both xpha
and xamp have to be set to data.
"""
from tensorpac import Pac
from tensorpac.signals import pac_signals_tort, pac_signals_wavelet
import pandas as pd
import numpy as np
from scipy.signal import hilbert

import matplotlib.pyplot as plt
# plt.style.use('seaborn-poster')
n_perm=1000
f_p = 4
f_a = 25

# First, we generate 2 datasets of signals artificially coupled between 10hz
# and 100hz. By default, those datasets are organized as (n_epochs, n_times)
# where n_times is the number of time points.
n_epochs = 100 # 20  # number of datasets
prefpha = (np.pi/4)
d1, time = pac_signals_wavelet(f_pha=f_p, f_amp=f_a, noise=0, sf=1024, n_times=10*1024,n_epochs=n_epochs, pp=prefpha)
# d1, time = pac_signals_tort(f_pha=f_p, f_amp=f_a, noise=0, n_epochs=n_epochs,
#                             dpha=0, damp=0, chi=0, n_times=10*1024)
# d2, time = pac_signals_tort(f_pha=10, f_amp=100, noise=3, n_epochs=n_epochs,
#                             dpha=20, damp=5, chi=0.3, n_times=3000)

# Define the model of PAC to use :
p = Pac(idpac=(4, 0, 0), f_pha=(1, 12, 1, 0.5), f_amp=(12, 50, 1, 0.1),
        dcomplex='wavelet', width=12)
# Now, compute PAC by taking the phase of the first dataset and the amplitude
# of the second

phases = p.filter(1024, d1, ftype='phase')
amplitudes = p.filter(1024, d1, ftype='amplitude')
xpac12 = p.fit(phases, amplitudes) #, n_perm=n_perm)
# xpac12 = p.filterfit(1024, d1, d2)
# Invert by taking the phase of the second dataset and the amplitude of the
#  first one :
# xpac21 = p.filterfit(1024, d2, d1)

st , en = [0,len(d1[0])]
st , en = [2000,3024]

# np.shape(amplitudes)
# np.shape(phases)
plt.figure(figsize=(10, 20), layout='constrained')
indx_frqs = range(0,20,3)
pltindx = 0
for i in indx_frqs:
    pltindx += 1
    plt.subplot(len(indx_frqs), 1, pltindx)
    
    plt.plot(time[st:en],phases.mean(1)[i][st:en])
    plt.ylabel(str(i), fontsize='xx-large')
# plt.show()

# Plot signals and PAC :
plt.figure(figsize=(10, 12), layout='constrained')

fbool = pd.DataFrame([f[0]<f_p<f[1] for f in p.f_pha], columns=['val'])
pfq =  fbool[fbool.val==True].index[0]

fbool = pd.DataFrame([f[0]<f_a<f[1] for f in p.f_amp], columns=['val'])
afq = fbool[fbool.val==True].index[0]

print('\n' + str(np.shape(amplitudes)[0]) + ' amplitude frequency bands, ' + str(np.shape(phases)[0])+ ' phase frequency bands')

afq = 5
pfq = 0

plt.subplot(4, 1, 1)
plt.plot(time[st:en], d1.mean(0)[st:en], color='k')
plt.xlabel('Time')
plt.ylabel('Amplitude [uV]')
plt.title('Raw Signal')
plt.axis('tight')

plt.subplot(4, 1, 2)
plt.plot(time[st:en], amplitudes[afq].mean(0)[st:en], color='k')
plt.xlabel('Time')
plt.ylabel('Amplitude [uV]')
plt.title('Frequency Amplitude')
plt.axis('tight')

plt.subplot(4, 1, 3)
gg = phases[pfq].mean(0)[st:en]
plt.plot(time[st:en], np.cos(gg), color='k')
plt.xlabel('Time')
plt.ylabel('Amplitude [uV]')
plt.title('Frequency Phase')
plt.axis('tight')

plt.subplot(4, 1, 4)
p.comodulogram(xpac12.mean(-1), title="Phase of the first dataset and "
               "amplitude of the second", cmap='Reds')

# plt.subplot(2, 2, 4)
# p.comodulogram(xpac21.mean(-1), title="Phase of the second dataset and "
#                "amplitde of the second", cmap='Reds')
plt.show()
