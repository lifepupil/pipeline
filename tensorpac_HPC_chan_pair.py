# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 13:32:42 2023

@author: CRichard
"""
import os 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import seaborn as sns
from tensorpac import Pac #, EventRelatedPac, PreferredPhase
# from tensorpac.utils import PeakLockedTF, PSD, ITC, BinAmplitude
# from tensorpac.signals import pac_signals_wavelet
# import mne
# import pickle5 as pickle

channel_1 = ['FC2', '/'] # EEG channel 1 to process Linux format
channel_2 = ['O2','/'] # EEG channel 2 to process Linux format
# channel_1 = ['FC2', '\\'] # EEG channel 1 to process Windows format
# channel_2 = ['O2','\\'] # EEG channel 2 to process Windows format
not_for_humans = True # IF False THEN KEEPS AXIS LABELS
# Dynamic definition of f_pha and f_amp:
# (start freq., stop freq., width, step)
f_pha = [0, 13, 2, 0.5]    # frequency range phase for the coupling
f_amp = [4, 50, 6, 1]      # frequency range amplitude for the coupling
vmin = -4
vmax = 8
epoch_dur = 300 # how many seconds in each epoch
pac_method = 5 # Phase-Locking Value=5, modulation index=2
surrogate_method = 2 # METHOD FOR COMPUTING SURROGATES - Swap amplitude time blocks
norm_method = 4 # normalization method for correction - z-scores
# FOR ALL POSSIBLE SETTINGS, SEE:
#  https://etiennecmb.github.io/tensorpac/generated/tensorpac.Pac.html#tensorpac.Pac

alpha = 0.2
allsig = True # SET TO True TO SAVE FILE REGARDLESS OF ALPHA, False IF NOT
mcp = 'fdr' # 'maxstat', 'fdr', 'bonferroni'
n_perm=1000

which_pacdat = 'pacdat_MASTER.pkl'

# read_dir = "D:\\COGA_eec\\"
# write_dir = "D:\\COGA_eec\\FC2-O2\\"
# sub_dir = "new_pac\\"
#read_dir = "/ddn/crichard/eeg_csv/"
#write_dir = "/ddn/crichard/pipeline/processed/"
read_dir = os.environ['TMPDIR'] + '/input/'
write_dir = os.environ['TMPDIR'] + '/results/'




# GET COGA META DATA
pacdat = pd.read_pickle(read_dir + which_pacdat)
chpac = pacdat[pacdat.channel==channel_1[0]]

# # SET UP OUR PHASE-AMPLITUDE COUPLING MODEL
# p = Pac(idpac=(pac_method, surrogate_method, norm_method), 
#         f_pha=f_pha, 
#         f_amp=f_amp,
#         dcomplex='wavelet', width=7, verbose=None)

debug_i = 0
for i in range(debug_i,len(chpac)):
    file_name_components = chpac.iloc[i].eeg_file_name.split('_')
    base_filename = '_'.join(file_name_components[1:])
    base_id = '_'.join(file_name_components[3:5])

    ch1FileName = chpac.iloc[i].eeg_file_name + '.csv'
    ch2FileName = channel_2[0] + '_' + base_filename + '.csv'
    
    ch1PathFileName = read_dir + channel_1[0] + channel_1[1] + ch1FileName 
    ch2PathFileName = read_dir + channel_2[0] + channel_2[1] + ch2FileName 

    ch1_exist = os.path.exists(ch1PathFileName)
    ch2_exist = os.path.exists(ch2PathFileName)
    if ch1_exist & ch2_exist:
        sample_rate = int(file_name_components[-1])

        data_1 = np.loadtxt(ch1PathFileName, delimiter=',', skiprows=1)
        data_2 = np.loadtxt(ch2PathFileName, delimiter=',', skiprows=1)
        print('\nWorking on ' + base_filename + ', ' + str(i+1) + ' of ' + str(len(chpac)) + ' files' )
    else:
        # mf = [ch1PathFileName, ch2PathFileName]
        # missingfn = mf[ch1_exist, ch2_exist]
        # print('Missing file ' + missingfn)
        continue
    
    # GET DIAGNOSIS TO ADD TO FIGURE TITLE
    if chpac.iloc[i].AUD_this_visit:
        dx = 'AUD'
    else:
        dx = 'UNAFF'
    
    # JUST IN CASE EPOCH DURATION IS GRETER THAN THE LENGTH OF THE SIGNAL
    if sample_rate*epoch_dur<= len(data_1):        
        time_intervals = list(range(0,len(data_1),sample_rate*epoch_dur))
    else:
        time_intervals = [0,len(data_1)]

    
    for t in range(0,len(time_intervals)-1): 
        
        start = time_intervals[t]
        end = time_intervals[t+1]
        
        segment_1 = data_1[start:end]
        segment_2 = data_2[start:end]
        
        ssr = str(int(start/sample_rate))
        esr = str(int(end/sample_rate))
        sig_diff = str(round(np.mean(segment_1 - segment_2),14))
        title = 'dx: ' + dx + ', ' + ssr + '-' + esr + ' sec, ' + channel_1[0] + '-' + channel_2[0]  

        # p = Pac(idpac=(pac_method, surrogate_method, norm_method), 
        #         f_pha=(f_pha[0], f_pha[1], 1, 0.5), 
        #         f_amp=(f_amp[0], f_amp[1], 1, 0.2),
        #         dcomplex='wavelet', width=7, verbose=None)
        p = Pac(idpac=(pac_method, surrogate_method, norm_method), 
                f_pha=f_pha, 
                f_amp=f_amp,
                dcomplex='wavelet', width=7, verbose=None)
        
        # Now, extract channel 1 phases and channel 2 amplitudes
        print('\n' + channel_1[0] + ' phase, ' + channel_2[0] + ' amplitude, \n' + ssr + '-' + esr + ' sec, ' + base_id)
        phases = p.filter(sample_rate, segment_1, ftype='phase')
        amplitudes = p.filter(sample_rate, segment_2, ftype='amplitude')
        xpac12 = p.fit(phases, amplitudes, n_perm=n_perm)
        
        pval = p.infer_pvalues(p=alpha, mcp=mcp)
        anysig = len(pval[pval<1])>0
        if anysig | allsig:
            
            print(title)
            print('SIG ' + base_id + ', epoch ' + str(t) + '\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
            savename = channel_1[0] + channel_2[0] +'\\' + channel_1[0] + '-' + channel_2[0] + '_' + base_filename
            
            xpac = xpac12.squeeze()
            plt.figure(figsize=(5, 5))
            pac_ns = xpac.copy()
            pac_ns[pval <= alpha] = np.nan
            pac_s = xpac.copy()
            pac_s[pval > alpha] = np.nan
            plt.subplot(1, 1, 1)

            if not_for_humans:
                p.comodulogram(pac_ns, cmap='gray', colorbar=False, vmin=np.nanmin(pac_ns), vmax=np.nanmax(pac_ns), rmaxis=True)
                p.comodulogram(pac_s, cmap='Reds', vmin=np.nanmin(pac_s), vmax=np.nanmax(pac_s), colorbar=False, rmaxis=True)
                plt.gca().invert_yaxis()
                plt.gca().get_xaxis().set_visible(False)
                plt.gca().get_yaxis().set_visible(False)
                plt.savefig(write_dir + savename + '_t' + str(t) + '_' + str(epoch_dur) + '' + '.jpg', bbox_inches='tight')
            else:
                p.comodulogram(pac_ns, cmap='gray', colorbar=False, vmin=np.nanmin(pac_ns), vmax=np.nanmax(pac_ns))
                p.comodulogram(pac_s, title=title + base_id, cmap='Reds', vmin=np.nanmin(pac_s), vmax=np.nanmax(pac_s))
                plt.gca().invert_yaxis()
                plt.savefig(write_dir + savename + '_t' + str(t) + '_' + str(epoch_dur) + '' + '.jpg')
            plt.close()

        
        # REVERSE CHANNELS PHASE AND AMPLITUDE ASSIGNMENTS
        print(channel_2[0] + ' phase, ' + channel_1[0] + ' amplitude, \n' + ssr + '-' + esr + ' sec, ' + base_id)
        phases = p.filter(sample_rate, segment_2, ftype='phase')
        amplitudes = p.filter(sample_rate, segment_1, ftype='amplitude')
        xpac21 = p.fit(phases, amplitudes, n_perm=n_perm)
        
        pval = p.infer_pvalues(p=alpha, mcp=mcp)
        anysig = len(pval[pval<1])>0
        if anysig | allsig:
            
            print(title)
            print('SIG ' + base_id + ', epoch ' + str(t) + '\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
            savename = channel_2[0] + channel_1[0] +'\\' + channel_2[0] + '-' + channel_1[0] + '_' + base_filename

            xpac = xpac12.squeeze()
            plt.figure(figsize=(5, 5))
            pac_ns = xpac.copy()
            pac_ns[pval <= alpha] = np.nan
            pac_s = xpac.copy()
            pac_s[pval > alpha] = np.nan
            plt.subplot(1, 1, 1)
            
            if not_for_humans:
                p.comodulogram(pac_ns, cmap='gray', colorbar=False, vmin=np.nanmin(pac_ns), vmax=np.nanmax(pac_ns), rmaxis=True)
                p.comodulogram(pac_s, cmap='Reds', vmin=np.nanmin(pac_s), vmax=np.nanmax(pac_s), colorbar=False, rmaxis=True)
                plt.gca().invert_yaxis()
                plt.gca().get_xaxis().set_visible(False)
                plt.gca().get_yaxis().set_visible(False)
                plt.savefig(write_dir + savename + '_t' + str(t) + '_' + str(epoch_dur) + '' + '.jpg', bbox_inches='tight')
            else:
                p.comodulogram(pac_ns, cmap='gray', colorbar=False, vmin=np.nanmin(pac_ns), vmax=np.nanmax(pac_ns))
                p.comodulogram(pac_s, title=title + base_id, cmap='Reds', vmin=np.nanmin(pac_s), vmax=np.nanmax(pac_s))
                plt.gca().invert_yaxis()
                plt.savefig(write_dir + savename + '_t' + str(t) + '_' + str(epoch_dur) + '' + '.jpg')     
            plt.close()

            


            
    

                
    
            
