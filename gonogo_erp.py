# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 12:48:44 2025

@author: lifep
"""

# TO DO
# 1. WRITE FUNCTION TO CHECK EACH ERP EPOCH IF IT'S BAD (DELTA AMPLITUDE >100 uV | DELTA AMPLITUDE <0.5 uV)
# 2. ARE WAVEFORMS FROM EARLY ERPs INDISTINGUISHABLE FROM LATE ERP WAVEFORMS?
# 3. FIND THE MISSING ERP, ONLY FOUND 99 (49 GO, 50 NOGO) TRIALS
# 4. WILL HAVE TO DOWN SAMPLE ALL EEG TO THE LOWEST COMMON DENOMINATOR (256 Hz SAMPLING RATE?) BEFORE PASSING ERPs TO TENSORPAC
# 5. GET ORDER OF GO/NOGO TRIALS AS RESPONSE MIGHT VARY DEPENDING ON HOW LONG SINCE LAST GO TRIAL
# 6. GET FREQUENCY PAIRS AT WHICH SIGNIFICANT (OR TRENDING) PAC WAS FOUND FOR EACH SUBJECT-CHANNEL-EPOCH 
# 7. MAKE OUTPUT TABLE CONTAINING SIGNIFICANT PHASE AND AMPLITUDE FREQUENCIES, ASSOCIATED PAC, TRIAL TYPE, PAC METHOD, CHANNEL, EPOCH, 

# The subject is presented with visual stimuli. 
# The stimuli are triangles which either point upward, downward, towards the right, or towards the left. 
# This experiment consists of two phases; a practice phase and an experimental phase. 
# The practice phase runs for approximately 1 minute and the experimental phase runs for 4 minutes. 
# During the practice phase the subject is presented with 20 stimuli, 5 in each direction. 
# In the experimental phase the subject is presented with 100 stimuli, 25 in each direction. 
# The triangles that point upwards and downwards are called the Go stimuli, 
# for which the subject is asked to respond with a button press. 
# The triangles that point towards the right and left are the No-Go stimuli, 
# for which the subject is told not to respond. 
# If the subject responds correctly to the Go stimuli, 
# or appropriately refrains from responding to the No-Go stimuli, 
# a dollar sign will appear on the screen. 
# If the subject responds incorrectly, an “X” will appear on the screen. 
# No data collection is required for the practice phase. 
# The total inter-trial interval is 2400 ms. 
# The time sequences of stimuli presentation for the Go and No-Go trials are outlined below.

# Trigger       Case	  Description		               Suffix
# 10		    Go		    Triangle pointing up/down       G
# 20		    No-Go		Triangle pointing right/left    NG
# 40		    $ - Go      Dollar sign for Go              D-G
# 50		    X - Go      Cross sign for Go               C-G
# 60		    X - NoGo    Cross sign for Go               C-NG
# 70		    $ - NoGo    Dollar sign for Go              D-NG

# Go Trials:			
# 		Stimulus presented for 100 ms	
# 		500 ms response window		
# 		Feedback ($ or X) = 200 ms
# 		ITI = 1200 ms
# No-Go Trials:
# 		Stimulus presented for 100 ms
# 		1000 ms response inhibition window
# 		Feedback ($ or X) = 200 ms
# 		ITI = 1200 ms


# PROGRAMMING NOTES
# * WARNING WHEN GENERATING metadata DOES NOT APPEAR TO POSE ANY PROBLEMS WITH DATA PROCESSING 
#   (SEE: https://github.com/mne-tools/mne-python/issues/12346)
# *


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import pandas as pd 
import mne
# from mne.preprocessing import ICA
# from mne_icalabel import label_components
import matplotlib.pyplot as plt
# import coga_support_defs as csd

from tensorpac import Pac, EventRelatedPac, PreferredPhase
from tensorpac.utils import PeakLockedTF, PSD, ITC, BinAmplitude


np.set_printoptions(legacy="1.25")

# INSTANCE VARIABLES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# P = 'D:\\COGA_eec\\0_gonogo\\test_data\\'
pkl_path = 'D:\\COGA_eec\\0_gonogo\\pkl\\'
P = 'D:\\COGA_eec\\0_gonogo\\cleaned_data\\'
Fs = [
      # 'gng_2_a1_a0001385_32.fif', 
      # 'gng_2_a1_40205010_32.fif',
      'gng_2_a1_p0000882_32.fif'
      ]
# F = 'gng_2_a1_a0001385_32.fif'
# F = 'gng_2_a1_40205010_32.cnt'
# F = 'gng_2_a1_p0000882_32.fif'

alpha = 0.05

# FOR ERP PAC
epochs_tmin, epochs_tmax = -0.4, 2
metadata_tmin, metadata_tmax = 0, 2
reject = {"eeg": 250e-6}  # exclude epochs with strong artifacts

f_pha = [2, 12, 1, 1]       # frequency range phase for the coupling; start, stop, width, step
f_amp = [12, 40, 1, 1]      # frequency range amplitude for the coupling
width = 4 # default 7
n_perm = 400 # iterations to generate surrogate PAC data

# CONSTANTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
all_event_id = {
    'response': 0,
    "go": 10,
    "nogo": 20,
    "outcome/wingo": 40,
    "outcome/lossgo": 50,
    "outcome/lossnogo": 60,
    "outcome/winnogo": 70
}

event_dict_trials = {
    "go": 10,
    "nogo": 20
}

# IF USING events_from_annotations FUNCTION
# all_event_id = {
#     'resp': 1,
#     "go": 2,
#     "nogo": 3,
#     "go-$": 4,
#     "go-X": 5,
#     "nogo-X": 6,
#     "nogo-$": 7
# }

# event_dict_trials = {
#     "go": 2,
#     "nogo": 3
# }



# CODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

sig_pac = []
result_df = pd.read_pickle(pkl_path + 'test.pkl')
freq_pairs = tuple(zip(result_df.f_p, result_df.f_a))
result_df['p_a'] = freq_pairs
freq_pairs = sorted(list(set(freq_pairs)))


for F in Fs:
    
    # N.B. -- int16 AND int32 FOR THE data_format IN THE read_raw_cnt FUNCTION WILL GIVE DIFFERENT EVENT MARKER TIMES. 
    #       LOOKS LIKE int32 IS CORRECT AS IT ALIGNS WITH THE 4 MINUTE TASK DURATION LISTED FOR GNG IN MANUAL
    if F[-3:]=='fif':
        raw = mne.io.read_raw_fif(P + F, preload=False, verbose=False)
    else:
        raw = mne.io.read_raw_cnt(P + F, data_format='int32', preload=False, verbose=False)

    info = raw.info
    eeg_indices = mne.pick_types(info, meg=False, eeg=True, ecg=False, eog=False, exclude=['BLANK','X','Y'])
    mne.pick_info(info, eeg_indices, copy=False)
    # raw.set_eeg_reference('average',projection=True)
    
    sn = [raw.time_as_index(s)[0] for s in raw.annotations.onset]
    ff = pd.DataFrame({'onset': sn, 'duration':raw.annotations.duration, 'event': raw.annotations.description})
    ff.event = ff.event.astype(int)
    ff.duration = ff.duration.astype(int)
    all_events = ff.to_numpy()
    
    # all_events, all_event_id = mne.events_from_annotations(raw)
    
    epochs = mne.Epochs(raw, all_events, tmin=epochs_tmin, tmax=epochs_tmax, event_id=event_dict_trials, preload=False)
    # epochs = mne.Epochs(raw, all_events, baseline=(0,0), event_id=event_dict_trials, preload=False)
    
    tt = raw.annotations.description
    print('response : ' + str(sum(tt=='0')))
    print('go : ' + str(sum(tt=='10')))
    print('nogo : ' + str(sum(tt=='20')))
    print('g$ : ' + str(sum(tt=='40')))
    print('gX : ' + str(sum(tt=='50')))
    print('ng$ : ' + str(sum(tt=='70')))
    print('ngX : ' + str(sum(tt=='60')))
    
    
    
    # THIS NEXT LINE ALLOWS US TO GIVE US METADATA BY TRIAL SO WE CAN 
    # DISTINGUISH BETWEEN CORRECT AND INCORRECT RESPONSES (AND REACTION TIMES) FOR EACH TRIAL
    row_events = ['go','nogo']
    keep_first = 'outcome'
    metadata, events, event_id = mne.epochs.make_metadata(
        events=all_events,
        event_id=all_event_id,
        tmin=metadata_tmin,
        tmax=metadata_tmax,
        sfreq=raw.info["sfreq"],
        row_events=row_events,
        keep_first=keep_first
    )
    
    # MAKE THIS A METHOD FOR CLASS gonogo_proc FOR BEST CODING PRACTICES
    metadata['response'].plot.hist(bins=50, title='reaction times', xlabel='seconds', xlim=[0,1])
    
    metadata = metadata.drop(['go', 'nogo', 'outcome/wingo', 'outcome/lossgo','outcome/lossnogo', 'outcome/winnogo', 'outcome'], axis=1)
    # metadata.loc[np.isnan(metadata.resp),'resp'] = 0
    metadata.loc[:,'response_correct'] = ''
    metadata.loc[ metadata['first_outcome'].isin(['wingo','winnogo']), 'response_correct'] = True
    metadata.loc[ metadata['first_outcome'].isin(['lossgo','lossnogo']), 'response_correct'] = False
    metadata.reset_index(drop=True, inplace=True)
    # metadata['epoch_order'] = metadata.index
    
    epochs = mne.Epochs(
        raw=raw,
        tmin=epochs_tmin,
        tmax=epochs_tmax,
        events=events,
        event_id=event_id,
        metadata=metadata,
        reject=reject,
        preload=True,
    )
    
    
    # epochs.pick_channels(['CP6','P4','P8','TP7','C1'])
           
    erp_go = epochs[(epochs.metadata.event_name=='go') & (epochs.metadata.response_correct==True)].average()
    erp_nogo = epochs[(epochs.metadata.event_name=='nogo') & (epochs.metadata.response_correct==True)].average()
    
    fig, ax = plt.subplots(2, figsize=(6, 8), layout="constrained")
    erp_go.plot(gfp=True, spatial_colors=True, axes=ax[0], show=False)
    erp_nogo.plot(gfp=True, spatial_colors=True, axes=ax[1], show=False)
    ax[0].set_title("GO trials")
    ax[1].set_title("NOGO trials")
    fig.suptitle("Visual ERPs", fontweight="bold")
    ax[1].xaxis.set_visible(False)
    plt.show()
    plt.close(fig)
    
    
    df = epochs.to_data_frame()
    cols = df.columns.to_list()
    sample_rate = raw.info["sfreq"]

    time_i = cols.index('time')
    cond_i = cols.index('condition')
    ep_i = cols.index('epoch')
    time = df[df.epoch==0].time.values
    
    # freq_pairs = [[2,3,11,14]]
  
    for pa in range(len(freq_pairs)):
        this_fp = freq_pairs[pa]
        erps_this_fp = result_df[result_df.p_a==this_fp]
        # erps_this_fp = result_df[(result_df.f_p > 2) & (result_df.f_p < 3) & (result_df.f_a > 11) & (result_df.f_a <=14)]
        erp_subset = []
        
        for idx in erps_this_fp.index:
            this_ch = erps_this_fp[erps_this_fp.index==idx].chan.values[0]
            this_ep = erps_this_fp[erps_this_fp.index==idx].epoch.values[0]
            this_tr = erps_this_fp[erps_this_fp.index==idx].trial.values[0]
            if (np.any(df['epoch']==this_ep)):
            
                dat = df.loc[(df['epoch']==this_ep)][this_ch].values
                erp_subset.append(dat)
        data = np.array(erp_subset)
        # data = data.transpose()

        # from tensorpac.signals import pac_signals_wavelet
        # n_epochs = 61
        # n_times = 1201
        # x1, tvec = pac_signals_wavelet(f_pha=10, f_amp=100, n_epochs=n_epochs, noise=.8, n_times=n_times, sf=sample_rate)
        # # Second signal : one second of random noise
        # x2 = np.random.rand(n_epochs, 1000)
        # x = np.concatenate((x1, x2), axis=1)
        # time = np.arange(x.shape[1]) / sample_rate
        # erps = df[df.epoch==0].iloc[:,3:]
        # erps.to_numpy().T
        # ee = (erps.to_numpy().T)
        # # SHAPE OF ee IS (61, 1201)
        # time = np.arange(ee.shape[1]) / sample_rate
        
        # ERP ~~~~~~~~~~~~~~~~~~~~~        
        phf = [this_fp[0]-0.5, this_fp[0]+0.5, 0.25, 0.1]
        amf = [this_fp[1]-2.5, this_fp[1]+2.5, 0.5, 0.25]
        p = EventRelatedPac(f_pha=phf, f_amp=amf, dcomplex='wavelet', width=width) #, verbose=False)
        
        pha = p.filter(sample_rate, data, ftype='phase', n_jobs=1)
        amp = p.filter(sample_rate, data, ftype='amplitude', n_jobs=1)
        erpac = p.fit(pha, amp, method='circular', n_perm=n_perm, mcp='fdr', verbose=False)
        # erpac = p.filterfit(sample_rate, data, method='circular', mcp='fdr').squeeze()
        # erpac = p.filterfit(sample_rate, data, method='gc', mcp='fdr', n_perm=n_perm, p=0.05).squeeze()

        pvalues = p.pvalues
        minp = pvalues.min()
        print('P-A frequency pair ' + str(this_fp) + '\t minimum p-value = ' + str(round(minp,5)) + '    N = ' + str(len(data)))
    
        if minp<=alpha:
            erpac[pvalues > alpha] = np.nan
            vmin, vmax = np.nanmin(erpac), np.nanmax(erpac)
            if not(np.isnan(vmin)):
                # for i in range(erpac.shape[0]):
                for i in ([erpac.shape[0]-1]):
                    # THIS WILL PLOT THE MEAN ERP FOR THIS CONDITION
                    # plt.plot(time, np.average(data, axis=0), lw=0.5)
                    title = 'n_pha=' + str(i) + ' ' + str(this_fp) + ' minp=' + str(round(minp,5))
                    p.pacplot(erpac[i, : ,:], time, p.yvec, xlabel='Time (second)',
                              cmap='Spectral_r', ylabel='Amplitude frequency', title=title,
                              cblabel='ERPAC', rmaxis=True) #, vmin=vmin, vmax=vmax)
                    # plt.axvline(1., linestyle='--', color='k', linewidth=2)
                    p.show()
            
            # itc = ITC(data, sample_rate, f_pha=(1, 20, 1, 0.2))
            # itc.plot(times=time, cmap='plasma', fz_labels=15, fz_title=18)
            # plt.show()
            # plt.close()  
        
        # for i in range(8):
        #     for j in range(8):
        #         plt.plot(erpac[i,j, :])
        #         plt.title('n_amp=' + str(i) + ', n_pha=' + str(j))
        #         plt.show()
        #         plt.close()
        

        
