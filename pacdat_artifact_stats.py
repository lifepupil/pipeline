# -*- coding: utf-8 -*-
"""
Created on Tue May  7 15:40:34 2024

@author: lifep
"""

import pandas as pd
# import matplotlib.pyplot as plt

logy = False
# logy = True

base_dir = 'D:\\COGA_eec\\' #  BIOWIZARD
which_pacdat = 'pacdat_MASTER.pkl'

pacdat = pd.read_pickle(base_dir + which_pacdat)

fz = pacdat[(pacdat.channel=='FZ')]

# fz[['max_flat']].plot.hist(bins=50,xlabel='seconds', title='Duration of maximum flat interval\n(by EEG channel)',logy=logy)
# fz[['max_noise']].plot.hist(bins=10,xlabel='seconds', title='Duration of maximum noise interval\n(by EEG channel)',logy=logy)
fz[['max_flat_slip0']].plot.hist(bins=50,xlabel='seconds', title='Max duration of flat interval with slip0\n(by EEG channel from eec)',logy=logy)
# fz[['max_flat_slip1']].plot.hist(bins=50,xlabel='seconds', title='Max duration of flat interval with slip1\n(by EEG channel from eec)',logy=logy)
fz[['avg_flat_slip0']].plot.hist(bins=50,xlabel='seconds', title='Avg duration of flat interval with slip0\n(by EEG channel from eec)',logy=logy)
fz[['perc_flat_slip0']].plot.hist(bins=50,xlabel='percentage', title='Percent flat interval with slip0\n(by EEG channel from eec)',logy=logy)
# fz[['perc_flat_slip1']].plot.hist(bins=50,xlabel='percentage', title='Percent flat interval with slip1\n(by EEG channel from eec)',logy=logy)
fz[['max_noise_slip0']].plot.hist(bins=50,xlabel='seconds', title='Max duration of noise interval with slip0\n(by EEG channel from eec)',logy=logy)
# fz[['max_noise_slip1']].plot.hist(bins=50,xlabel='seconds', title='Max duration of noise interval with slip1\n(by EEG channel from eec)',logy=logy)
fz[['avg_noise_slip0']].plot.hist(bins=50,xlabel='seconds', title='Avg duration of noise interval with slip0\n(by EEG channel from eec)',logy=logy)
# fz[['avg_noise_slip1']].plot.hist(bins=50,xlabel='seconds', title='Avg duration of noise interval with slip1\n(by EEG channel from eec)',logy=logy)
fz[['perc_noise_slip0']].plot.hist(bins=50,xlabel='percentage', title='Percent noise interval with slip0\n(by EEG channel from eec)',logy=logy)
# fz[['perc_noise_slip1']].plot.hist(bins=50,xlabel='percentage', title='Percent noise interval with slip1\n(by EEG channel from eec)',logy=logy)


# ss = list(set(fz.site))
# for i in range(0,len(ss)): print(ss[i]+' '+ str(len(fz[(fz.max_flat_slip0==0) & (fz.site==ss[i])])))