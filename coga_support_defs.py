# -*- coding: utf-8 -*-
"""
Created on Fri May 26 15:14:17 2023

@author: Christian Richard

A quick and easy function to convert .sas7pdat files into tables saved as .CSV files

usage:
    - if output_folder = '' then it will save the output file to the working directory
    
"""

def convert_sas_tables(this_file, output_folder, output_fname):
    import pandas as pd
    this_sas7pdat = pd.read_sas(this_file)
    if not output_folder:
        output_folder = '.\\'
    this_sas7pdat.to_csv(output_folder + output_fname + '.csv',index=False)
    

def get_file_list(base_dir,file_type):
    # WE WANT TO FIND ALL OF THE .CNT FILES IN THIS FOLDER STRUCTURE SO THAT 
    # WE HAVE THE FULL PATH AND FILENAME TO PERMIT DOWNSTREAM PROCESSING
    # built directory walk, as generic as possible so will work for TUH and COGA folder structures
    # create functions to open either EDF or CNT files
    import os
    eegList = []
    for root, dirs, files in os.walk(base_dir, topdown=True):
        for file in files:
            if file[len(file)-3:len(file)]==file_type:
                eegList += [[root , file]]
    return eegList
    

def plot_eeg_mwt(freqlist, mwt_path, mwt_file, eeg_path, eeg_file):
    #  PRIMARILY USING THIS FOR MANUAL INSPECTION OF DATA
    #  ADD FUNCTIONALITY TO SAVE PLOTS AT LATER DATE
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # freqlist CONTAINS THE FREQUENCIES USED IN THE COMPLEX MORLET WAVELET TRANSFORM
    y_freqs = pd.read_csv(freqlist)
    indx = y_freqs.values.T[0]
    indx = np.flip(indx)
    indx = [round(num, 1) for num in indx]
    
    # NOW WE OPEN THE TIME-FREQUENCY SCALOGRAM, mwt
    chunk = pd.read_csv(mwt_path + mwt_file, chunksize=1000,sep='\t', header=0)
    mwt_df = pd.concat(chunk)
    # THEN ADD FREQUENCIES ANALYZED SO THAT THEY CAN BE ADDED TO OUTPUT FIGURE
    mwt_df.index=indx
    # NOW FIND THE ASSOCIATED EEG SIGNAL FILE
    chunk = pd.read_csv(eeg_path + eeg_file, chunksize=1000,sep='\t')
    eeg_df = pd.concat(chunk)
    sample_rate = int(eeg_file[-7:-4])
    this_channel = eeg_file.split('_')[0]
    this_eeg = eeg_df[this_channel].values
    
    # t1 = 30000
    # t2 = 31000
    
    t1 = 0
    t2 = len(this_eeg)

    # ADD COLUMN HEADER FOR TIME
    cnames = range(len(this_eeg))
    cnames = [round((num/sample_rate)/60,1) for num in cnames]
    mwt_df.columns = cnames
    
    # NOW WE MAKE THE PLOT    
    fig, ax  = plt.subplots(2, 1, sharex='col')
    ax[0].set_ylabel('Voltage')
    ax[0].plot(this_eeg[t1:t2], linewidth=0.2, color='black')
    sns.heatmap(mwt_df.iloc[:,t1:t2],  ax=ax[1], cbar=False)  
    ax[1].set_ylabel('Frequency (Hz)')
    ax[1].set_xlabel('Time (minutes)')
    
def do_pac(data):
    # 1. NEED TO HAVE COLLECTED ALL FILENAMES OF RECORDINGS FROM SAME EEG CHANNEL
    #       BEST TO MAKE SMALL HELPER FUNCTION TO SPLIT FILENAMES TO GET CHANNEL INFO
    # 2. EACH OF THESE FILES NEEDS TO BE OPENED AND PUT INTO AN ARRAY
    # 3. 
    
    sf = 500
    
    import matplotlib.pyplot as plt
    from tensorpac import Pac, EventRelatedPac, PreferredPhase
    from tensorpac.utils import PeakLockedTF, PSD, ITC, BinAmplitude
    from tensorpac.signals import pac_signals_wavelet
    
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
    
def get_sub_from_fname(whichSub, whichIndx):
    return whichSub.split('_')[whichIndx]

def get_diagnosis(srow, visit_list, thisVisit, dx, int_dx):
    # DSM4 DIAGNOSTIC CRITERIA FOR ALCOHOL ABUSE AT LEAST 1 OF 4 IN PAST 12 MONTHS
    # DSM4 DIAGNOSTIC CRITERIA FOR ALCOHOL DEPENDENCE AT LEAST 3 OF 7 IN PAST 12 MONTHS
    # DSM5 DIAGNOSTIC CRITERIA FOR AUD AT LEAST 2 OF 11 IN PAST 12 MONTHS
    if srow.iloc[0][dx] == 5:
        dx_now = True

        visit_of_dx_code = srow.iloc[0][int_dx].replace('_', '').replace('\'', '').replace('b', '').replace('C', '').replace('S', '')
        visit_of_dx = visit_list.index(visit_of_dx_code)
        # IN ORDER TO FIND VISITS TAKING PLACE BEFORE SUBJECTS DEVELOP AUD, ALCOHOL DEPENDENCE, OR ALCOHOL ABUSE
        # WE CALCULATE HOW MANY VISITS BEFORE DIAGNOSIS TO MAKE IT EASY TO FIND THEM DOWNSTREAM
        dx_visits_ago = visit_of_dx - thisVisit
        # NOW WE CAN DETERMINE WHETHER THE FIRST VISIT OF DIAGNOSIS IS BEFORE THIS VISIT OR NOT
        if thisVisit >= visit_of_dx:
            dx_then = True
        else:
            dx_then = False   
    else:
        dx_now = False
        dx_then = False
        dx_visits_ago = 999
    return [dx_then, dx_now, dx_visits_ago]
        