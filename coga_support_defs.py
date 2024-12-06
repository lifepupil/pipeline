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




def match_age2(group1,group2,seeds, ttl):
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    
    group1 = group1.sample(frac=1, random_state=seeds[0]).reset_index(drop=True)
    group2 = group2.sample(frac=1, random_state=seeds[1]).reset_index(drop=True)
    
    
    len_grp2 = len(group2)
    print('N of AUD ' + str(len_grp2))
    
    group2_ind = []
    group1_ind = []
    
    minage = int(min(  int(min(group1.age_this_visit)),int(min(group2.age_this_visit))  ))
    maxage = int(max(  int(max(group1.age_this_visit)),int(max(group2.age_this_visit))  ))
    # ttl = 'Ages ' + str(minage) + '-' + str(maxage) + ', AUD N=' + str(len(group2)) + ', ctl N=' + str(len(group1))
    pltlog = True
        
    bins = np.linspace(minage, maxage, maxage-minage+1)
    ga1 = group1[['age_this_visit']].values[:,0]
    ga2 = group2[['age_this_visit']].values[:,0]
    N = len(ga2)
    
    # pd.concat([group1,group2]).plot.hist(column=["age_this_visit"], by="AUD_this_visit", title='AUD and non-AUD age distributions')
    # plt.hist([ga1,ga2], bins=bins, label=['unaffected','AUD'], log=pltlog)
    plt.hist(ga1, bins=bins, label='unaffected', edgecolor='black', color='w', log=pltlog) #, align='right')
    plt.hist(ga2, bins=bins, label='AUD', alpha=0.5, color='b', log=pltlog) #, ax=ax1) #, align='right')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.ylim([1,1000])
    plt.title(ttl + ', AUD_N=' + str(N))
    plt.legend()
    plt.show()
    
    # pd.concat([group1,group2]).plot.hist(column=["age_this_visit"], by="AUD_this_visit", title='AUD and non-AUD age distributions')
    # plt.hist(group1[['age_this_visit']], bins=(maxage-minage), label='unaffected', edgecolor='black', color='w', log=pltlog)
    # plt.hist(group2[['age_this_visit']], bins=(maxage-minage), label='AUD', alpha=0.5, color='b', log=pltlog)
    # plt.xlabel('Age')
    # plt.ylabel('Frequency')
    # plt.title(ttl)
    # plt.legend()
    # plt.show()
    
    ages = range(minage,maxage+1,1)
    
    for age in ages:
        g1age = group1[group1.age_this_visit==age].reset_index(drop=True)
        g2age = group2[group2.age_this_visit==age].reset_index(drop=True)
        print('age is ' + str(age) + ' -- length g1age = ' + str(len(g1age)) + ', length g2age = ' + str(len(g2age)))
        if len(g2age)>0:
            if len(g2age)>len(g1age): 
                for s in range(0,len(g1age)):
                    group1_ind.append(g1age.iloc[s].copy())
                    group2_ind.append(g2age.iloc[s].copy())
            else:
                for s in range(0,len(g2age)):
                    group1_ind.append(g1age.iloc[s].copy())
                    group2_ind.append(g2age.iloc[s].copy())
                    
    print('final group1 N = ' + str(len(group1_ind)) + ' \nfinal group2 N = ' + str(len(group2_ind)))    
    group1_ind = pd.DataFrame(group1_ind)
    group2_ind = pd.DataFrame(group2_ind)
    return group1_ind, group2_ind
 
    
    
    
    
    
    
    
    
def match_age1(group1,group2, seeds):
    import math
    import pandas as pd
    
    print('match_age1 started')

    # group1 = group1.sort_values(by=['age_this_visit'])
    # group2 = group2.sort_values(by=['age_this_visit'])
    # group2 = group2.reset_index(drop=True)
    
    group1 = group1.sample(frac=1, random_state=seeds[0]).reset_index(drop=True)
    group2 = group2.sample(frac=1, random_state=seeds[1]).reset_index(drop=True)
    

    
    # hdr = group1.columns()
    len_grp1 = len(group1)
    len_grp1_now = len(group1)

    len_grp2 = len(group2)
    print('N of AUD ' + str(len_grp2))
    
    group2_ind = []
    group1_ind = []

    while len(group1)>0:
        print('SECOND PASS - still ' + str(len(group1)) + ' ')
        for row2_i, age2 in enumerate(group2.age_this_visit):
            
            # JUST IN CASE WE HAVE NAN
            if math.isnan(age2): # continue if age is missing
                print('nan found' + ' ' + str(row2_i) )
                continue
            
            group1 = group1.reset_index(drop=True)
            # age2 = group2.age_this_visit.values[row2_i]
            # WE WANT TO GET ALL group1 ROW INDICES THAT HAVE SAME AGE AS SUBJECT TO BE MATCHED FROM group2
            g1_age2 = group1[group1.age_this_visit==age2].index
            # JUST IN CASE WE RUN OUT, AND IF SO THIS HELPS TO EXPLAIN HOW MANY SUBJECTS ARE LOST FROM LACK OF AN AGE MATCH
            if len(g1_age2)==0:
                print('no more matchable subjects for age ' + str(age2))
                break
            
            row1_i = 0
            while row1_i < len(g1_age2):
                # WE USE g1a2_i TO KNOW WHICH ROW TO REMOVE FROM group1
                g1a2_i = g1_age2[row1_i]
                age1 = group1.age_this_visit.values[g1a2_i]
                if math.isnan(age1):# continue if age is missing
                    print('nan found' + (str(row1_i) + ' ' + str(row2_i) ))
                    continue
                
                # WE WANT TO MAKE SURE WE ARE AGE MATCHING TWO DIFFERENT PEOPLE, NOT THE SAME PERSON
                if (group1.iloc[g1a2_i].ID == group2.iloc[row2_i].ID):
                    print(str(row2_i) + ' of ' + str(len_grp2) + ' alc, ' + str(row1_i) + ' of ' + str(len_grp1_now) + ' ctl (' + str(len_grp1) +  '), bad match ' + group1.iloc[row1_i].eeg_file_name + ' ' +  group2.iloc[row2_i].eeg_file_name)
                    row1_i += 1
                    continue


                if (age2==age1):
                    group1_ind.append(group1.iloc[g1a2_i].copy())
                    group2_ind.append(group2.iloc[row2_i].copy())
                    # print(str(group1_ind[-1].age_this_visit) + ' ' + str(group2_ind[-1].age_this_visit))
                    group1.drop(index=g1a2_i, inplace=True)
                    if len(group1[group1.ID==group1_ind[-1].ID])>0:
                        # len_before = len(group1)
                        group1 = group1[group1.ID!=group1_ind[-1].ID]
                        # len_after = len(group1)
                        # len_change = len_before - len_after
                        # print('removing other instances of matched subject ' + str(group1_ind[-1].ID) + ', N before=' + str(len_before) + ', removed ' + str(len_change) )

                    len_grp1_now = len(group1)

                    # group2.drop(index=row2_i, inplace=True)
                    # print(str(row2_i) + ' '  + str(row1_i))
                    break
                    
                row1_i += 1

    print('MATCHED: ' + str(row2_i) + ' of ' + str(len_grp2) + ' alc, ' + str(len_grp1 - len(group1)) + ' of ' + str(len_grp1) + ' ctl')    
    print('final group1 N = ' + str(len(group1_ind)) + ' \nfinal group2 N = ' + str(len(group2_ind)))    
    group1_ind = pd.DataFrame(group1_ind)
    group2_ind = pd.DataFrame(group2_ind)
    
    return group1_ind, group2_ind
    # filename='group2_ind_aud.txt'
    # np.savetxt(filename, group2_ind)
    # filename='group1_ind_cntl.txt'
    # np.savetxt(filename, group1_ind)
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    # The two groups include ID and session,
    # and age (it can have more features of course)
    # the return are the indices from both goups that are
    # matched
    # group2_ind = []
    # group1_ind = []
    # # age_diffs = [0]
    # age_diffs = [0,1,-1,2,-2,3,-3]
    # for this_ad in age_diffs:
    #     for row2_i, age2 in enumerate(group2.age_this_visit):
    #         matched = False
    #         if math.isnan(age2): # continue if age is missing
    #             continue
    #         for row1_i, age1 in enumerate(group1.age_this_visit):
    #             # WE WANT TO MAKE SURE WE ARE AGE MATCHING TWO DIFFERENT PEOPLE, NOT THE SAME PERSON
    #             if (group1.iloc[row1_i].ID == group2.iloc[row2_i].ID):
    #                 continue
    #             if math.isnan(age1):# continue if age is missing
    #                 continue
    #             if row1_i in group1_ind: # continue if age is missing
    #                 continue
    #             if (age2==age1 + this_ad):
    #                 group1_ind.append(row1_i)
    #                 group2_ind.append(row2_i)
    #                 # group1.drop(index=row1_i, inplace=True)
    #                 # group2.drop(index=row2_i, inplace=True)
    #                 matched = True
    #                 break
    #             else:
    #                 if row1_i == len(group1):
    #                     break
    #             if matched:
    #                 break

    #     return group1_ind, group2_ind
    #     # filename='group2_ind_aud.txt'
    #     # np.savetxt(filename, group2_ind)
    #     # filename='group1_ind_cntl.txt'
    #     # np.savetxt(filename, group1_ind)