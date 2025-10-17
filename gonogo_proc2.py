# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 16:13:22 2025

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
pkl_path = 'D:\\COGA_eec\\0_gonogo\\pkl\\'
# P = 'D:\\COGA_eec\\0_gonogo\\test_data\\'
P = 'D:\\COGA_eec\\0_gonogo\\cleaned_data\\'
Fs = ['gng_2_a1_p0000882_32_eeg.fif', 'gng_2_a1_a0001385_32.fif', 'gng_2_a1_40205010_32.cnt']
# F = 'gng_2_a1_a0001385_32.fif'
# F = 'gng_2_a1_40205010_32.cnt'
# F = 'gng_2_a1_p0000882_32.fif'

alpha = 0.05

# FOR ERP PAC
epochs_tmin, epochs_tmax = -0.4, 2
metadata_tmin, metadata_tmax = 0, 2
reject = {"eeg": 250e-6}  # exclude epochs with strong artifacts
phf = [6,7]
amf = [16,20,0.5,0.1]

# FOR COMODULOGRAMS
epoch_dur = 2.4 # how many seconds in each epoch
pac_method = 5 # MeanVectorLength=1 modulation index=2 heightsRatio=3 ndPAC=4 Phase-Locking Value=5 GaussianCopulaPAC=6 
surrogate_method = 2 # METHOD FOR COMPUTING SURROGATES - Swap amplitude time blocks
norm_method = 4 # normalization method for correction - z-scores
f_pha = [2, 12, 1, 1]       # frequency range phase for the coupling; start, stop, width, step
f_amp = [12, 40, 1, 1]      # frequency range amplitude for the coupling
width = 3 # default 7
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


for F in Fs:
    
    # N.B. -- int16 AND int32 FOR THE data_format IN THE read_raw_cnt FUNCTION WILL GIVE DIFFERENT EVENT MARKER TIMES. 
    #       LOOKS LIKE int32 IS CORRECT AS IT ALIGNS WITH THE 4 MINUTE TASK DURATION LISTED FOR GNG IN MANUAL
    # raw = mne.io.read_raw_cnt(P + F, data_format='int32', preload=False, verbose=True)
    raw = mne.io.read_raw_fif(P + F, preload=False, verbose=False)
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
    
    # epochs = mne.Epochs(
    #     raw=raw,
    #     tmin=0,
    #     tmax=2.4,
    #     # baseline=(None,0), # DOES NOT WORK 
    #     # baseline=(0,None), # SAME AS (None,None)
    #     baseline=(None,None),
    #     # baseline=(0,0),
    #     events=events,
    #     event_id=event_id,
    #     metadata=metadata,
    #     reject=reject,
    #     preload=True,
    # )
    
    
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
    # cols = ['time', 'condition', 'epoch', 'FP1', 'FP2', 'F7', 'F8', 'AF1', 'AF2',
    #        'FZ', 'F4', 'F3', 'FC6', 'FC5', 'FC2', 'FC1', 'T8', 'T7', 'CZ', 'C3',
    #        'C4', 'CP5', 'CP6', 'CP1', 'CP2', 'P3', 'P4', 'PZ', 'P8', 'P7', 'PO2',
    #        'PO1', 'O2', 'O1', 'AF7', 'AF8', 'F5', 'F6', 'FT7', 'FT8', 'FPZ', 'FC4',
    #        'FC3', 'C6', 'C5', 'F2', 'F1', 'TP8', 'TP7', 'AFZ', 'CP3', 'CP4', 'P5',
    #        'P6', 'C1', 'C2', 'PO7', 'PO8', 'FCZ', 'POZ', 'OZ', 'P2', 'P1', 'CPZ']
    # cols[3:] = np.sort(cols[3:len(cols)])
    
    # from tensorpac.signals import pac_signals_wavelet
    # n_epochs = 61
    # n_times = 1201
    sample_rate = raw.info["sfreq"]
    # x1, tvec = pac_signals_wavelet(f_pha=10, f_amp=100, n_epochs=n_epochs,
    #                                noise=.8, n_times=n_times, sf=sample_rate)
    # # Second signal : one second of random noise
    # x2 = np.random.rand(n_epochs, 1000)
    # x = np.concatenate((x1, x2), axis=1)
    # time = np.arange(x.shape[1]) / sample_rate
    # erps = df[df.epoch==0].iloc[:,3:]
    # erps.to_numpy().T
    # ee = (erps.to_numpy().T)
    # time = np.arange(ee.shape[1]) / sample_rate
    
    time_i = cols.index('time')
    cond_i = cols.index('condition')
    ep_i = cols.index('epoch')
    time = df[df.epoch==0].time.values
    
    p = Pac(idpac=(pac_method, surrogate_method, norm_method), 
            f_pha=f_pha, 
            f_amp=f_amp,
            dcomplex='wavelet', width=width, verbose=None)
    pac_meth = str(p)
    
    for c in cols[3:]:
        print(c)
        chan_i = cols.index(c)
        dffz = df.iloc[:,[time_i,cond_i,ep_i,chan_i]]
        dffz.pivot(index='epoch', columns='time',values=c)
        
        trial_type = 'go'
        dfgo = dffz[dffz.condition==trial_type]    
        go_epoch_list = list(set(dfgo.epoch))
        g = dfgo.pivot(index='epoch', columns='time',values=c)
        # g = gg.to_numpy()
        
        segments = np.array(g.query('epoch==' + str(go_epoch_list[0])).values[0])
        for epoch_i in go_epoch_list[1:]:
            # segment = g[epoch_i]
            segment  = g.query('epoch==' + str(epoch_i)).values[0]
            segments = np.vstack([segment, segments])
        title = 'Go trial channel ' + c 
        
        # p = Pac(idpac=(pac_method, surrogate_method, norm_method), 
        #         f_pha=f_pha, 
        #         f_amp=f_amp,
        #         dcomplex='wavelet', width=width, verbose=None)
        
        # Now, extract all of the phases and amplitudes
        # phases = p.filter(sample_rate, segment, ftype='phase')
        # amplitudes = p.filter(sample_rate, segment, ftype='amplitude')
        # xpac = p.fit(phases, amplitudes, n_perm=n_perm, p=0.05, mcp='fdr')
        xpac = p.filterfit(sample_rate, segments, n_perm=n_perm, n_jobs=-1, random_state=0, verbose=None)
        # surro = p.surrogates.squeeze()
        # print(f"Surrogates shape (n_perm, n_amp, n_pha) : {surro.shape}")
        # get the maximum of the surrogates across (phase, amplitude) pairs
        # surro_max = surro.max(axis=(1, 2))
        pval = p.infer_pvalues(p=alpha, mcp='fdr')
    
        if np.any(pval<=alpha):
            # get the mean pac values where it's detected as significant
            xpac_smean = xpac[pval < alpha].mean()
            # plt.figure()
            # # p.comodulogram(xpac.mean(-1), cmap='Spectral_r', plotas='contour', ncontours=5, title=title, fz_title=14, fz_labels=13)
            # p.comodulogram(xpac, cmap='Spectral_r', plotas='contour', ncontours=5, title=title, fz_title=14, fz_labels=13)
            # # export the figure
            # # plt.savefig('readme.png', bbox_inches='tight', dpi=300)
            # p.show()
            # plt.close()
            
            # FIGURE -  COMODULOGRAM FOR THIS EPOCH
            plt.figure()
            p.comodulogram(xpac, title=pac_meth + '\n' + title, cmap='Spectral_r', vmin=0., pvalues=pval, levels=0.95)
            p.show()
            plt.close()
            
            # FIGURE SHOWING PHASE AND AMPLITUDE FREQUENCIES WHERE P-VALUES ARE SIGNIFICANT
            plt.figure()
            p.comodulogram(pval, title='P-values - ' + title, cmap='rainbow', vmin=1/n_perm, vmax=0.10, over='lightgray')
            p.show()
            plt.close()  
        
            # FIGURE SHOWS NULL PAC DISTRIBUTIONS AND CORRECTED PAC DISTRIBUTIONS WITH CUTOFF
            # plt.figure()
            # plt.hist(xpac.max(axis=1), bins=30)
            # plt.hist(surro_max, bins=4)
            # plt.title('Corrected distribution of surrogates\n' + title)
            # plt.axvline(xpac_smean, lw=1, color='red', linestyle='--')
            # p.show()
            # plt.close()
            
            amp, phi = np.where(pval<alpha)
            for sig_i in range(len(amp)):
                rt = metadata[metadata.index==epoch_i].response.values[0]
                f_p = p.xvec[phi[sig_i]]
                f_a = p.yvec[amp[sig_i]]
                pv = pval[amp[sig_i], phi[sig_i]]
                pac = xpac[amp[sig_i], phi[sig_i]]
                dat = pd.DataFrame({'epoch': epoch_i, 
                                    'trial': trial_type,
                                    'chan': c,
                                    'f_p': f_p, 
                                    'f_a': f_a, 
                                    'p': pv, 
                                    'pac': pac,
                                    'rt': rt,
                                    'fname': F,
                                    'method': pac_meth}, index=[0])
                sig_pac.append(dat)
    
        trial_type = 'nogo'
        dfnogo = dffz[dffz.condition==trial_type]
        nogo_epoch_list = list(set(dfnogo.epoch))
        ng = dfnogo.pivot(index='epoch', columns='time',values=c)
        # ng = ng.to_numpy()
        
        for epoch_i in nogo_epoch_list:
            segment  = ng.query('epoch==' + str(epoch_i)).values[0]
            # segment = ng[epoch_i]
            title = 'NoGo trial channel ' + c + ' epoch ' + str(epoch_i)
        
            # p = Pac(idpac=(pac_method, surrogate_method, norm_method), 
            #         f_pha=f_pha, 
            #         f_amp=f_amp,
            #         dcomplex='wavelet', width=width, verbose=None)
            
            # Now, extract all of the phases and amplitudes
            # phases = p.filter(sample_rate, segment, ftype='phase')
            # amplitudes = p.filter(sample_rate, segment, ftype='amplitude')
            # xpac = p.fit(phases, amplitudes, n_perm=n_perm, p=0.05, mcp='fdr')
            xpac = p.filterfit(sample_rate, segment, n_perm=n_perm, n_jobs=-1, random_state=0).squeeze()    
            surro = p.surrogates.squeeze()
            # print(f"Surrogates shape (n_perm, n_amp, n_pha) : {surro.shape}")
            # get the maximum of the surrogates across (phase, amplitude) pairs
            surro_max = surro.max(axis=(1, 2))
            pval = p.infer_pvalues(p=alpha)
        
        
            if np.any(pval<=alpha):
                # get the mean pac values where it's detected as significant
                xpac_smean = xpac[pval < alpha].mean()
                # plt.figure()
                # # p.comodulogram(xpac.mean(-1), cmap='Spectral_r', plotas='contour', ncontours=5, title=title, fz_title=14, fz_labels=13)
                # p.comodulogram(xpac, cmap='Spectral_r', plotas='contour', ncontours=5, title=title, fz_title=14, fz_labels=13)
                # # export the figure
                # # plt.savefig('readme.png', bbox_inches='tight', dpi=300)
                # p.show()
                # plt.close()
            
                plt.figure()
                p.comodulogram(xpac, title=str(p) + '\n' + title, cmap='Spectral_r', vmin=0., pvalues=pval, levels=0.95)
                p.show()
                plt.close()
                
                plt.figure()
                p.comodulogram(pval, title='P-values - ' + title, cmap='rainbow', vmin=1/n_perm, vmax=0.10, over='lightgray')
                p.show()
                plt.close()  
            
                # plt.figure()
                # plt.hist(xpac.max(axis=1), bins=30)
                # plt.hist(surro_max, bins=4)
                # plt.title('Corrected distribution of surrogates\n' + title)
                # plt.axvline(xpac_smean, lw=1, color='red', linestyle='--')
                # p.show()
                # plt.close()
                
                amp, phi = np.where(pval<alpha)
                for sig_i in range(len(amp)):
                    rt = metadata[metadata.index==epoch_i].response.values[0]
                    f_p = p.xvec[phi[sig_i]]
                    f_a = p.yvec[amp[sig_i]]
                    pv = pval[amp[sig_i], phi[sig_i]]
                    pac = xpac[amp[sig_i], phi[sig_i]]
                    dat = pd.DataFrame({'epoch': epoch_i, 
                                        'trial': trial_type,
                                        'chan': c,
                                        'f_p': f_p, 
                                        'f_a': f_a, 
                                        'p': pv, 
                                        'pac': pac,
                                        'rt': rt,
                                        'fname': F,
                                        'method': pac_meth}, index=[0])
                    # sig_pac.append(pd.DataFrame(dat, index=[0]))
                    sig_pac.append(dat)
    
    
    
    
        # # ERP ~~~~~~~~~~~~~~~~~~~~~
        # p = EventRelatedPac(f_pha=phf, f_amp=amf, dcomplex='wavelet', verbose=False)
        # mcp = 'fdr'
        # erpac = p.filterfit(sample_rate, g, method='circular', mcp=mcp).squeeze()
        # pvalues = p.pvalues.squeeze()
        # minp = pvalues.min()
        # print(c + '\t' + str(round(minp,2)))
    
        # if minp<=alpha:
        #     erpac[pvalues > alpha] = np.nan
        #     vmin, vmax = np.nanmin(erpac), np.nanmax(erpac)
        #     title = c + '   ' + str(round(minp,3))
        #     p.pacplot(erpac, time, p.yvec, xlabel='Time (second)',
        #               cmap='Spectral_r', ylabel='Amplitude frequency', title=title,
        #               cblabel='ERPAC', rmaxis=True, vmin=vmin, vmax=vmax)
        #     # plt.axvline(1., linestyle='--', color='k', linewidth=2)
        #     p.show()
            
            # itc = ITC(g, sample_rate, f_pha=(1, 20, 1, 0.2))
            # itc.plot(times=time, cmap='plasma', fz_labels=15, fz_title=18)
            # plt.show()
            # plt.close()
        
result_df = pd.concat(sig_pac, ignore_index=True)
result_df.to_pickle(pkl_path + 'test.pkl')

