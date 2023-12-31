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

def remove_completed_files_from_list(cntList, completedPath, institutionDir):
    # N.B. - IF EVEN ONE CSV FILE OF CHANNEL DATA IS WRITTEN FOR A SUBJECT VISIT
    # THEN THIS WILL EXCLUDE THAT SUBJECT VISIT FROM BEING FULLY PROCESSED. SO 
    # IF MANUALLY STOPPING CODE THEN DO SO BETWEEN WRITING EVENTS - THESE EVENTS  
    # CAN BE READILY SEEN IN THE CONSOLE FYI
    import os
    
    # FIRST WE GET A LIST ONLY OF THE FILE NAMES. SINCE ALL FILENAMES ARE UNIQUE
    # REGARDLESS OF WHICH SITE THEY ARE COLLECTED WE ONLY HAVE TO COMPARE FILENAMES
    cntListFN = [f[1] for f in cntList]
    # WE WANT TO SEE WHAT SUBJECT FILES HAVE ALREADY BEEN PROCESSED SO WE FIRST GET ALL OUTPUT FILES
    completedList = os.listdir(completedPath)
    # NEXT WE REMOVE CHANNEL AND FILE EXTENSION FROM EACH 
    # THEN WE REJOIN CORE FILE NAMES WITH THE APPROPRIATE EXTENSION IN THIS CASE, .cnt (LOWER CASE)
    # NOW WE REMOVE DUPLICATE CNT FILE NAMES
    coreCompletedList = _get_subject_list(completedList, True)
    
    # UNCOMMENT NEXT TWO LINE TO GET SUBJECT COUNT FOR METHODS SECTION
    # subList = _get_subject_list(cntListFN, False)
    # print(institutionDir + ': ' +  str(len(subList)) )

    # NEXT WE DETERMINE WHICH CNT FILE NAMES ARE IN BOTH
    isect = set(cntListFN).intersection(coreCompletedList)
    # THEN GET THE INDICES OF THE FILES FROM coreCompletedList THAT ARE ALSO IN cntList
    completed_idx = [cntListFN.index(i) for i in isect]
    # FINALLY WE REMOVE THEM FROM cntList WHICH SHOULD BE IDENTICAL TO THE INDICES 
    # FOR THE cntListFN LIST
    cntList = [f for i, f in enumerate(cntList) if i not in completed_idx]
    # WE GET THE TOTAL NUMBER OF FILES TO PROCESS SO WE CAN PROVIDE INFORMATION 
    # IN THE CONSOLE ABOUT HOW MANY FILES ARE LEFT TO PROCESS
    return cntList
        
def _get_subject_list(thisList, removeChanBool):
    # NEXT WE REMOVE CHANNEL AND FILE EXTENSION FROM EACH 
    splitList = [str.split(fn,'_') for fn in thisList]
    # THEN WE REJOIN CORE FILE NAMES WITH THE APPROPRIATE EXTENSION IN THIS CASE, .cnt (LOWER CASE)
    # SINCE WE NEED TO REMOVE THE CHANNEL INFORMATION FROM THE coreCompletedList 
    # BUT NOT cntListFN WE HAVE TO USE DIFFERENT RANGES WHEN DOING JOIN OPERATION
    if removeChanBool:
        coreList = ['_'.join(fn[1:len(fn)-2])+'.cnt' for fn in splitList]
    else:
        coreList = ['_'.join(fn[0:len(fn)-1])+'.cnt' for fn in splitList]
    # NOW WE REMOVE DUPLICATE CNT FILE NAMES
    coreList = set(coreList)
    return coreList

def print_demo_vals(tbl):
    # COUNTS BY SEX, RACE, HISPANIC, AND SITE OF DATA ACQUISITION
    num_m = len(tbl[tbl.sex=='M'])
    num_f = len(tbl[tbl.sex=='F'])
    num_w = len(tbl[tbl.race=='WHITE'])
    num_ww = len(tbl.loc[(tbl['race']=='WHITE') & (tbl['hispanic']==False)])
    num_wh = len(tbl.loc[(tbl['race']=='WHITE') & (tbl['hispanic']==True)])
    num_b = len(tbl[tbl.race=='BLACK'])
    num_bb = len(tbl.loc[(tbl['race']=='BLACK') & (tbl['hispanic']==False)])
    num_bh = len(tbl.loc[(tbl['race']=='BLACK') & (tbl['hispanic']==True)])
    num_o = len(tbl[tbl.race=='OTHER'])
    num_oo = len(tbl.loc[(tbl['race']=='OTHER') & (tbl['hispanic']==False)])
    num_oh = len(tbl.loc[(tbl['race']=='OTHER') & (tbl['hispanic']==True)])
    num_a = len(tbl[tbl.race=='ASIAN'])
    num_aa = len(tbl.loc[(tbl['race']=='ASIAN') & (tbl['hispanic']==False)])
    num_ah = len(tbl.loc[(tbl['race']=='ASIAN') & (tbl['hispanic']==True)])
    num_h = len(tbl.loc[(tbl['hispanic']==True)])
    num_nh = len(tbl.loc[(tbl['hispanic']==False)])
    num_su = len(tbl[tbl.site=='SUNY'])
    num_io = len(tbl[tbl.site=='IOWA'])
    num_iu = len(tbl[tbl.site=='IU'])
    num_uc = len(tbl[tbl.site=='UCONN'])
    num_sd = len(tbl[tbl.site=='UCSD'])
    num_wu = len(tbl[tbl.site=='WASHU'])
    num_alc = len(tbl[tbl.alcoholic==True])
    num_ctl = len(tbl[tbl.alcoholic==False])

    # PERCENTAGES OF THE ABOVE
    prc_m = num_m/len(tbl)*100
    prc_f = num_f/len(tbl)*100
    prc_o = num_o/len(tbl)*100
    prc_oo = num_oo/len(tbl)*100
    prc_oh = num_oh/len(tbl)*100
    prc_a = num_a/len(tbl)*100
    prc_aa = num_aa/len(tbl)*100
    prc_ah = num_ah/len(tbl)*100
    prc_b = num_b/len(tbl)*100
    prc_bb = num_bb/len(tbl)*100
    prc_bh = num_bh/len(tbl)*100
    prc_w = num_w/len(tbl)*100
    prc_ww = num_ww/len(tbl)*100
    prc_wh = num_wh/len(tbl)*100
    prc_h = num_h/len(tbl)*100
    prc_nh = num_nh/len(tbl)*100
    prc_su = num_su/len(tbl)*100
    prc_io = num_io/len(tbl)*100
    prc_iu = num_iu/len(tbl)*100
    prc_uc = num_uc/len(tbl)*100
    prc_sd = num_sd/len(tbl)*100
    prc_wu = num_wu/len(tbl)*100
    prc_alc = num_alc/len(tbl)*100
    prc_ctl = num_ctl/len(tbl)*100

    # SEND RESULTS TO CONSOLE
    print('Males: ' + str(num_m) + ' (' + str(round(prc_m,1)) + ' %)')
    print('Females: ' + str(num_f) + ' (' + str(round(prc_f,1)) + ' %)\n')
    print('blacks: ' + str(num_b) + ' (' + str(round(prc_b,1)) + ' %)')
    print('blacks non-hispanic: ' + str(num_bb) + ' (' + str(round(prc_bb,1)) + ' %)')
    print('black hispanics: ' + str(num_bh) + ' (' + str(round(prc_bh,1)) + ' %)\n')
    print('asians: ' + str(num_a) + ' (' + str(round(prc_a,1)) + ' %)')
    print('asians non-hispanics: ' + str(num_aa) + ' (' + str(round(prc_aa,1)) + ' %)')
    print('asian hispanics: ' + str(num_ah) + ' (' + str(round(prc_ah,1)) + ' %)\n')
    print('whites: ' + str(num_w) + ' (' + str(round(prc_w,1)) + ' %)')
    print('white non-hispanic: ' + str(num_ww) + ' (' + str(round(prc_ww,1)) + ' %)')
    print('white hispanic: ' + str(num_wh) + ' (' + str(round(prc_wh,1)) + ' %)\n')
    print('other: ' + str(num_o) + ' (' + str(round(prc_o,1)) + ' %)')
    print('other non-hispanic: ' + str(num_oo) + ' (' + str(round(prc_oo,1)) + ' %)')
    print('other hispanic: ' + str(num_oh) + ' (' + str(round(prc_oh,1)) + ' %)\n')
    
    print('hispanic: ' + str(num_h) + ' (' + str(round(prc_h,1)) + ' %)')
    print('non-hispanic: ' + str(num_nh) + ' (' + str(round(prc_nh,1)) + ' %)\n')
    
    print('SUNY: ' + str(num_su) + ' (' + str(round(prc_su,1)) + ' %)')
    print('IOWA: ' + str(num_io) + ' (' + str(round(prc_io,1)) + ' %)')
    print('IU: ' + str(num_iu) + ' (' + str(round(prc_iu,1)) + ' %)')
    print('UCONN: ' + str(num_uc) + ' (' + str(round(prc_uc,1)) + ' %)')
    print('UCSD: ' + str(num_sd) + ' (' + str(round(prc_sd,1)) + ' %)')
    print('WASHU: ' + str(num_wu) + ' (' + str(round(prc_wu,1)) + ' %)\n')
    
    print('alcoholic: ' + str(num_alc) + ' (' + str(round(prc_alc,1)) + ' %)')
    print('control: ' + str(num_ctl) + ' (' + str(round(prc_ctl,1)) + ' %)\n')


    print('TOTAL SAMPLE: ' + str(len(tbl)))
    
    
def get_recording_duration(f, sfreq):
    import numpy as np
    pth = f[0] + f[1] + '.csv'
    data = np.loadtxt(pth, delimiter=',', skiprows=1)
    eeg_dur = len(data)/sfreq 
    return eeg_dur
    
def get_band_psds(f, sfreq, FREQ_BANDS):
    import numpy as np
    # from scipy.signal import welch
    # from mne.time_frequency import psd_array_multitaper
    # from scipy.integrate import simps
    # import matplotlib.pyplot as plt
    import mne
    
    pth = f[0] + f[1] + '.csv'
    # TOP ROW OF THE CSV FILES CONTAINING EEG CHANNEL DATA IS CHANNEL NAME SO WE EXCLUDE IT
    data = np.loadtxt(pth, delimiter=',', skiprows=1)
    # WE NEED TO MAKE A MNE info OBJECT TO GET THE RAW DATA FROM THE CSV FILE 
    # IN ORDER TO MAKE AN MNE OBJECT TO GET PSD
    info = mne.create_info(ch_names=['chan'], sfreq=sfreq, ch_types=["eeg"])
    data = data.reshape(1,len(data))
    raw = mne.io.RawArray(data, info)
    # mne.viz.plot_raw_psd(raw)
    # NOW WE CALCULATE PSD FOR THIS MNE OBJECT
    pspect = raw.compute_psd(fmin=1.0, fmax=50.0)
    psds, freqs = pspect.get_data(return_freqs=True)
    # AND NORMALIZE TO GET RELATIVE EEG VALUES
    psds /= np.sum(psds, axis=-1, keepdims=True)
    # FINALLY WE AVERAGE SPECTRAL POWER (IN uV^2/Hz, I.E. IN dB)
    # FOR THE FREQ_BANDS PASSED TO THIS FUNCTION WHICH USE STANDARD EEG
    # FREQUENCY BANDS BY DEFAULT
    band_powers = []
    for fmin, fmax in FREQ_BANDS.values():
        psds_band = psds[0][ (freqs >= fmin) & (freqs < fmax)].sum()
        band_powers.append(psds_band.reshape(len(psds), -1))
    # FINALLY WE RETURN THOSE dB VALUES FOR INCLUSION INTO OUTPUT TABLE
    return np.concatenate(band_powers, axis=1)[0]
    
    # pth = "C:\\Users\\crichard\\Downloads\\data.txt"
    # pth = "D:\\COGA\\data_for_mwt\\O1_eec_5_l1_40039009_32.cnt_500.csv"
    # band_rng = [8, 12]
    # time = np.arange(data.size)/sample_rate
    # plt.plot(time, data, lw=1.5,color='k')
    # # SET SIZE (DURATION LENGTH) OF WINDOW FOR CALCULATING POWER SPECTRAL DENSITY
    # # BASED ON AT LEAST TWO COMPLETE CYCLES OF THE LOW FREQUENCY BOUNDARY 
    # # win_length = (2/band_rng[0])*sample_rate
    # # NOW WE GET PDSs
    # # freqs, psd = welch(data, sample_rate, nperseg=win_length)
    # psd, freqs = psd_array_multitaper(data, sample_rate, fmin=0,fmax=50)
    # plt.figure(figsize=(8,4))
    # plt.plot(freqs,psd)
    # plt.yscale('log')
    # plt.xlim([-1,20])
    # freq_resolution = band_rng[1] - band_rng[0]
    # # GETS THE INDICES BETWEEN WHICH REPRESENT THE HIGH AND LOW FREQUENCIES DEFINING THE EEG BAND 
    # freqBand_i = np.logical_and(freqs>=band_rng[0], freqs<band_rng[1])
    # band_power = simps(psd[freqBand_i], dx=freq_resolution)
    # bp_relative = band_power/simps(psd, dx=freq_resolution)
    
def convert_visit_code(vc):
    import string
    vc = vc.lower()
    visit_letters = list(string.ascii_lowercase)
    try:
        thisVisit = [visit_letters.index(vc)][0] + 1
    except:
        thisVisit = 999
    return thisVisit
