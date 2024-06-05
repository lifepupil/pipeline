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
from tensorpac import Pac #, EventRelatedPac, PreferredPhase
# from tensorpac.utils import PeakLockedTF, PSD, ITC, BinAmplitude
# from tensorpac.signals import pac_signals_wavelet
import mne
# import pickle5 as pickle

epoch_dur = 30 # how many seconds in each epoch
pac_method = 5 # Phase-Locking Value=5, modulation index=2
surrogate_method = 2 # METHOD FOR COMPUTING SURROGATES - Swap amplitude time blocks
norm_method = 4 # normalization method for correction - z-scores
# FOR ALL POSSIBLE SETTINGS, SEE:
#  https://etiennecmb.github.io/tensorpac/generated/tensorpac.Pac.html#tensorpac.Pac

read_dir = "D:\\COGA_eec\\"
write_dir = "D:\\COGA_eec\\FC2-O2 pval\\"
# sub_dir = "new_pac\\"
#read_dir = "/ddn/crichard/eeg_csv/"
#write_dir = "/ddn/crichard/pipeline/processed/"
# read_dir = os.environ['TMPDIR'] + '/input/'
# write_dir = os.environ['TMPDIR'] + '/results/'
which_pacdat = 'pacdat_MASTER.pkl'
vmin = -3
vmax = 7
alpha = 0.2
f_pha = [0, 13]       # frequency range phase for the coupling
f_amp = [4, 50]      # frequency range amplitude for the coupling

channel_1 = ['FC2', '\\'] # EEG channel 1 to process Windows format
channel_2 = ['O2','\\'] # EEG channel 2 to process Windows format

pacdat = pd.read_pickle(read_dir + which_pacdat)

chpac = pacdat[pacdat.channel==channel_1[0]]

debug_i = 14
for i in range(debug_i,len(chpac)):
    file_name_components = chpac.iloc[i].eeg_file_name.split('_')
    base_filename = '_'.join(file_name_components[1:])
    base_id = '_'.join(file_name_components[3:5])

    ch1FileName = chpac.iloc[i].eeg_file_name + '.csv'
    ch2FileName = channel_2[0] + '_' + base_filename + '.csv'
    
    ch1PathFileName = read_dir + channel_1[0] + channel_1[1] + ch1FileName 
    ch2PathFileName = read_dir + channel_2[0] + channel_2[1] + ch2FileName 

    
    if os.path.exists(ch1PathFileName) & os.path.exists(ch2PathFileName):
        sample_rate = int(file_name_components[-1])

        data_1 = np.loadtxt(ch1PathFileName, delimiter=',', skiprows=1)
        data_2 = np.loadtxt(ch2PathFileName, delimiter=',', skiprows=1)
        print('\nWorking on ' + base_filename + ', ' + str(i+1) + ' of ' + str(len(chpac)) + ' files' )
    else:
        continue

    # GET DIAGNOSIS TO ADD TO FIGURE TITLE
    if chpac.iloc[i].AUD_this_visit:
        dx = 'AUD'
    else:
        dx = 'Unaff'
    
    time_intervals = list(range(0,len(data_1),sample_rate*epoch_dur))
    
    for t in range(0,len(time_intervals)-1): 
        
        # p = Pac(idpac=(pac_method, surrogate_method, norm_method), 
        #         f_pha=(f_pha[0], f_pha[1], 1, 0.5), 
        #         f_amp=(f_amp[0], f_amp[1], 1, 0.2),
        #         dcomplex='wavelet', width=7, verbose=None)
        p = Pac(idpac=(pac_method, surrogate_method, norm_method), 
                f_pha=(f_pha[0], f_pha[1], 1, 0.5), 
                f_amp=(f_amp[0], f_amp[1], 3, 1),
                dcomplex='wavelet', width=7, verbose=None)
        
        start = time_intervals[t]
        end = time_intervals[t+1]
        
        segment_1 = data_1[start:end]
        segment_2 = data_2[start:end]
        
        ssr = str(int(start/sample_rate))
        esr = str(int(end/sample_rate))
        sig_diff = str(round(np.mean(segment_1 - segment_2),14))
        title1 = channel_1[0] + ' EEG segment ' + ssr + '-' + esr + ' sec, Dx: ' + dx + ''
        title2 = channel_2[0] + ' EEG segment ' + ssr + '-' + esr + ' sec, ' + channel_1[0] + '-' + channel_2[0]  + ' Diff =' + sig_diff 

        
        # Now, extract channel 1 phases and channel 2 amplitudes
        print(channel_1[0] + ' phase, ' + channel_2[0] + ' amplitude \n')
        phases = p.filter(sample_rate, segment_1, ftype='phase')
        amplitudes = p.filter(sample_rate, segment_2, ftype='amplitude')
        # xpac12 = p.fit(phases, amplitudes, n_perm=200, p=0.05, mcp='fdr')
        xpac12 = p.fit(phases, amplitudes, n_perm=1000)
            
        # REVERSE CHANNELS PHASE AND AMPLITUDE ASSIGNMENTS
        print('\n' + channel_2[0] + ' phase, ' + channel_1[0] + ' amplitude')
        phases = p.filter(sample_rate, segment_2, ftype='phase')
        amplitudes = p.filter(sample_rate, segment_1, ftype='amplitude')
        xpac21 = p.fit(phases, amplitudes, n_perm=1000)
        
        # WE USE THIS TO SEE WHETHER OR NOT THERE ARE ANY SIGNIFICANT PAC TO PLOT
        bools12 = []
        for mcp in (['maxstat', 'fdr', 'bonferroni']):
            # get the corrected p-values
            pval = p.infer_pvalues(p=alpha, mcp=mcp)
            #  IS THERE AT LEAST ONE P-VALUE THAT SATISFIES THE ABOVE THRESHOLD?
            bools12.append(len(pval[pval<1])>0)
        bools21 = []
        for mcp in (['maxstat', 'fdr', 'bonferroni']):
            # get the corrected p-values
            pval = p.infer_pvalues(p=alpha, mcp=mcp)
            #  IS THERE AT LEAST ONE P-VALUE THAT SATISFIES THE ABOVE THRESHOLD?
            bools21.append(len(pval[pval<1])>0)
        bools = bools12 + bools21
        
        print(bools)
        print('\n')
        if np.array(bools).any():
            # Plot signals and PAC :
            time = np.array(range(0,len(segment_1)))/sample_rate
            x12_max = str(round(xpac12.mean(-1).max(),1))
            x12_min = str(round(xpac12.mean(-1).min(),1))
            x21_max = str(round(xpac21.mean(-1).max(),1))
            x21_min = str(round(xpac21.mean(-1).min(),1))
            any_diff = str((segment_1 - segment_2).any())
            
            plt.figure(figsize=(18, 12))
            plt.subplot(2, 2, 1)
            plt.plot(time,segment_1, color='k')
            plt.xlabel('Time')
            plt.ylabel('Amplitude [uV]')
            plt.title(title1)
            plt.axis('tight')
            
            plt.subplot(2, 2, 2)
            plt.plot(time,segment_2, color='k')
            plt.xlabel('Time')
            plt.ylabel('Amplitude [uV]')
            plt.title(title2)
            plt.axis('tight')
            
            plt.subplot(2, 2, 3)
            title = channel_1[0] + ' phase ' + channel_2[0] + ' amplitude from ' + base_id + ' (' + x12_min + ' to ' + x12_max + ')'
            p.comodulogram(xpac12.mean(-1), title=title, cmap='Reds', vmin=vmin, vmax=vmax)
            
            plt.subplot(2, 2, 4)
            title = channel_2[0] + ' phase ' + channel_1[0] + ' amplitude from ' + base_id + ' (' + x21_min + ' to ' + x21_max + ')'
            p.comodulogram(xpac21.mean(-1), title=title, cmap='Reds', vmin=vmin, vmax=vmax)
            plt.show()
        

            if np.array(bools12).any():
                #  GO AHEAD AND PLOT FIGURES FOR ANY LOW P-VALUE PAC HEATMAPS
                xpac = xpac12.squeeze()

                plt.figure(figsize=(16, 5))
                for n_mcp, mcp in enumerate(['maxstat', 'fdr', 'bonferroni']):
                    # get the corrected p-values
                    pval = p.infer_pvalues(p=alpha, mcp=mcp)   
                    # set to gray non significant p-values and in color significant values
                    pac_ns = xpac.copy()
                    pac_ns[pval <= alpha] = np.nan
                    pac_s = xpac.copy()
                    pac_s[pval > alpha] = np.nan
                
                    plt.subplot(1, 3, n_mcp + 1)
                    p.comodulogram(pac_ns, cmap='gray', colorbar=False, vmin=np.nanmin(pac_ns), vmax=np.nanmax(pac_ns))
                    title = dx.upper() + ' -- ' + base_id + f', {mcp} (' + channel_1[0] + '-' + channel_2[0] + ', ' + ssr + '-' + esr + ' sec)'
                    p.comodulogram(pac_s, title=title, cmap='Reds', vmin=np.nanmin(pac_s), vmax=np.nanmax(pac_s))
                    plt.gca().invert_yaxis()

                    plt.savefig(write_dir + 'test.jpg')
                    from PIL import Image
                    img2 = Image.open(write_dir + 'test.jpg')
                    img2.show()
                
                plt.tight_layout()
                plt.show()

                # plt.figure(figsize=(5, 5))
                # pval = p.infer_pvalues(p=alpha, mcp='fdr')   
                # pac_ns = xpac.copy()
                # pac_ns[pval <= alpha] = np.nan
                # pac_s = xpac.copy()
                # pac_s[pval > alpha] = np.nan
                # plt.subplot(1, 1, 1)
                # p.comodulogram(pac_ns, cmap='gray', colorbar=False, vmin=np.nanmin(pac_ns), vmax=np.nanmax(pac_ns), rmaxis=True)
                # p.comodulogram(pac_s, cmap='Reds', vmin=np.nanmin(pac_s), vmax=np.nanmax(pac_s), colorbar=False, rmaxis=True)
                # plt.gca().invert_yaxis()
                # plt.gca().get_xaxis().set_visible(False)
                # plt.gca().get_yaxis().set_visible(False)
                # plt.savefig(write_dir + 'test.jpg', bbox_inches='tight')                
                
                
                # UNCOMMENT TO PRINT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # x = xpac12.mean(-1)
                # img = sns.heatmap(np.flip(x,0), cmap='Reds',vmin=vmin, vmax=vmax, xticklabels=False,yticklabels=False, cbar=False)
                # fig = plt.Axes.get_figure(img)
                # # FINALLY WE SAVE IT AS A JPG -    THIS WILL BE IMPORTANT FOR RESIZING 
                # # THIS IMAGE FOR RESNET-50 USING PIL PACKAGE 
                # savename = channel_1[0] + '-' + channel_2[0] + '_' + base_filename
                # fig.savefig(write_dir + savename + '_t' + str(t) + '.jpg', bbox_inches='tight')
                # plt.close(fig)
                
            if np.array(bools21).any():
                #  GO AHEAD AND PLOT FIGURES FOR ANY LOW P-VALUE PAC HEATMAPS
                xpac = xpac21.squeeze()
                # pac_ns = xpac.copy()
                # pac_ns[pval <= .05] = np.nan
                # pac_s = xpac.copy()
                # pac_s[pval > .05] = np.nan
                # p.comodulogram(pac_ns, cmap='gray', colorbar=False, vmin=np.nanmin(pac_ns), vmax=np.nanmax(pac_ns))
                # p.comodulogram(pac_s, title=f'MCP={mcp}', cmap='viridis', vmin=np.nanmin(pac_s), vmax=np.nanmax(pac_s))
                plt.figure(figsize=(16, 5))
                for n_mcp, mcp in enumerate(['maxstat', 'fdr', 'bonferroni']):
                    # get the corrected p-values
                    pval = p.infer_pvalues(p=alpha, mcp=mcp)    
                    # set to gray non significant p-values and in color significant values
                    pac_ns = xpac.copy()
                    pac_ns[pval <= alpha] = np.nan
                    pac_s = xpac.copy()
                    pac_s[pval > alpha] = np.nan
                
                    plt.subplot(1, 3, n_mcp + 1)
                    p.comodulogram(pac_ns, cmap='gray', colorbar=False, vmin=np.nanmin(pac_ns), vmax=np.nanmax(pac_ns))
                    title = dx.upper() + ' -- ' + base_id + f', {mcp} (' + channel_2[0] + '-' + channel_1[0] + ', ' + ssr + '-' + esr + ' sec)'
                    p.comodulogram(pac_s, title=title, cmap='Reds', vmin=np.nanmin(pac_s), vmax=np.nanmax(pac_s))
                    plt.gca().invert_yaxis()
                
                plt.tight_layout()
                p.show()
                
                # UNCOMMENT TO PRINT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # x = xpac21.mean(-1)
                # img = sns.heatmap(np.flip(x,0), cmap='Reds',vmin=vmin, vmax=vmax, xticklabels=False,yticklabels=False, cbar=False)
                # fig = plt.Axes.get_figure(img)
                # # FINALLY WE SAVE IT AS A JPG -    THIS WILL BE IMPORTANT FOR RESIZING 
                # # THIS IMAGE FOR RESNET-50 USING PIL PACKAGE 
                # savename = channel_2[0] + '-' + channel_1[0] + '_' + base_filename
                # fig.savefig(write_dir + savename + '_t' + str(t) + '.jpg', bbox_inches='tight')
                # plt.close(fig)

        else:
            print('Not ' + base_id + ', epoch ' + str(t))
            print('Not ' + base_id + ' ' + str(t) + '\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            continue
            


            
    

                
    
            
