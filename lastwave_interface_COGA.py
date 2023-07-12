# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 15:59:34 2023

@author: lifep
"""
# when either EDF or CNT file is found, have it passed to function for opening that file type
# pull out info about file and folder location then put together a unique filename for output used by LastWave
#       - SAMPLE RATE
#       - SUBJECT ID
#       - DATE AND TIME OF RECORDING


# import required modules
import pandas as pd
import numpy as np
from scipy import signal
from scipy.signal import sosfiltfilt, butter
import seaborn as sns
import string

import mne
from mne.preprocessing import ICA
import torch
from mne_icalabel import label_components
import os
import sys
import matplotlib.pyplot as plt
import coga_support_defs as csd


# INSTANCE VARIABLES
drive_letter = 'E'
do_sas_convert = False             # TO CONVERT .SAS7PDAT FILES TO TABLES SO THAT SUBJECT METADATA CAN BE USED DOWNSTREAM
do_plot_eeg_signal_and_mwt = False # TO PLOT SIGNAL AND HEATMAP FOR A GIVEN FILE
do_filter_eeg_signal_cnt = True    # TO DO LOW PASS, HIGH PASS, NOTCH FILTER TO REMOVE LINE NOISE FROM SIGNAL, AND ICA
do_pac = False                      # PHASE AMPLITUDE COUPLING USING TENSORPAC

# PARAMETERS
if drive_letter=='C':
    base_dir = "C:\\Users\\CRichard\\Documents\\COGA_eec\\"
    eeg_dir = "C:\\Users\\CRichard\\Documents\\COGA_eec\\data_for_mwt\\"
else:
    base_dir = drive_letter + ":\\Documents\\COGA_eec\\"
    eeg_dir = drive_letter + ":\\Documents\\COGA_eec\\data_for_mwt\\"
    
# PARAMETERS FOR do_pac PHASE AMPLITUDE COUPLING USING TENSORPAC
if do_pac:
    pac_path = 'C:\\Users\\CRichard\\Documents\\COGA_eec\\tensorpac\\'
    core_pheno_list = 'C:\\Users\\CRichard\\Documents\\COGA_sub_info\\core_pheno_20201120.csv'
    # WHEN core_pheno FILE IS UPDATED WITH NEW INTERVIEW COLUMNS MUST ADD THEM HERE FOR CODE TO SCAN THROUGH ALL INTERVIEWS
    interview_col_names = ['intvw_p1', 
                           'intvw_p2', 
                           'intvw_p3', 
                           'intvw_p4', 
                           'intvw_p41',
                           'intvw_p42', 
                           'intvw_p43', 
                           'intvw_p44', 
                           'intvw_p45', 
                           'intvw_p46',
                           'intvw_p47']
    # TO CONVERT COGA SITE WHERE RECORDING TAKEN INTO HUMAN READABLE FORM
    coga_sites = {'1':'UCONN',
                  '2': 'IU',
                  '3': 'IOWA',
                  '4': 'SUNY',
                  '5': 'WASHU',
                  '6': 'UCSD'}
    # WE ALSO WANT TO CONVERT SEX CODES (SEX: 1=M,2=F) TO HUMAN READABLE ONES
    sex_dict = {1: 'M', 2: 'F'}
    # AND CONVERT CODE FOR HISPANIC ETHNICITY (1=False, 5=True)
    hisp_dict = {1: False, 5: True}
    
    chan_pos = 0
    task_pos = 1
    visit_pos = 3
    id_pos = 4
    
# PARAMETERS FOR do_sas_convert
if do_sas_convert:
    # sas_path_and_file = 'C:\\Users\\CRichard\\OneDrive - Downstate Medical Center\\coga_data\\PhaseIV\\ssaga_dx\\dx_ssaga4.sas7bdat'
    sas_path_and_file = 'C:\\Users\\CRichard\\OneDrive - Downstate Medical Center\\coga_data\\core_pheno_20201120\\core_pheno_20201120.sas7bdat'
    output_path = '' # IF THIS STRING IS EMPTY THEN IT WRITES TABLE TO WORKING DIRECTORY
    output_filename = 'core_pheno_20201120'

# PARAMETERS FOR do_plot_eeg_signal_and_mwt
if do_plot_eeg_signal_and_mwt:
    mwt_path = 'D:\\COGA\\curry_neuroscan\\mwt_RESULTS\\phase\\'
    eeg_path = 'C:\\Users\\crichard\\Documents\\COGA\\data_for_mwt\\'
    mwt_file = 'CPZ_sit_2_a1_a0001560_32.cnt_500_phi.csv'
    eeg_file = 'CPZ_sit_2_a1_a0001560_32.cnt_500.csv'
    freqlist_path_file = 'D:\\COGA\\freqList.txt'

# PARAMETERS FOR do_filter_eeg_signal_cnt 
if do_filter_eeg_signal_cnt:
    notch_freq = 60.0       # FREQUENCY (Hz) TO REMOVE LINE NOISE FROM SIGNAL 
    lowfrq = 0.1            # LOW PASS FREQUENCY
    hifrq = 100              # HIGH PASS FREQUENCY
    do_plot_channels = True
    
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# TO CONVERT .SAS7PDAT FILES TO TABLES SO THAT SUBJECT METADATA CAN BE USED DOWNSTREAM
if do_sas_convert: 
    csd.convert_sas_tables(sas_path_and_file, output_path, output_filename)

# TO PLOT TIME-FREQUENCY SCALOGRAM AND ASSOCIATED EEG SIGNAL
if do_plot_eeg_signal_and_mwt:
    csd.plot_eeg_mwt(freqlist_path_file, mwt_path, mwt_file, eeg_path, eeg_file)

# TO FILTER SIGNALS THEN SAVE THEM IN CSV AND PNG FILES
if do_filter_eeg_signal_cnt:
    # GET ALL THE .CNT FILE NAMES AND PATHS AND PUT INTO LIST 
    cntList = csd.get_file_list(base_dir, 'cnt')
    #  WE GO THROUGH EACH OF THE FILES IN cntList 
    for f in range(len(cntList)):
        fname = cntList[16][1]
        # fname = cntList[f][1]
        path_and_file = cntList[f][0] + '\\' + fname
        try:
            data = mne.io.read_raw_cnt(path_and_file, preload=True)
            # print(data.info["bads"])
            # data.plot()
            # a=1
        except Exception as e:
            with open(base_dir + 'errors_from_core_pheno.txt', 'a') as f:
                f.write(str(fname) + '\t ' + str(e) + '\n')
            continue                
                
        info = data.info
        # WE EXCLUDE THE BLANK CHANNEL AND RELABEL CHANNEL TYPES OF THE TWO EYE CHANNELS TO eog
        # ASSUMES THAT ALL CHANNELS ARE LABELED AS EEG WHETHER THEY ARE OR NOT
        data.set_channel_types({'X': 'eog', 'Y': 'eog'})
        # IN CASE YOU WANT TO SEE WHAT THE CHANNEL TYPE ASSIGNMENTS ARE JUST RUN LINE OF CODE BELOW
        # for i in range(len(info['chs'])): print([str(i) + ' ' + str(filtered_data.info['chs'][i]['kind'])])
        
        # WE WANT TO EXCLUDE ANY CHANNELS THAT AREN'T EEG OR eog JUST IN CASE WE HAVE ANY THERE
        eeg_indices = mne.pick_types(info, meg=False, eeg=True, ecg=False, eog=True, exclude=['bads', 'BLANK'])
        mne.pick_info(info, eeg_indices, copy=False)
        # NOW WE GET SAMPLE RATE AND CHANNEL INFO
        samp_freq = int(info['sfreq'])  # sample rate (Hz)
        channels = data.ch_names
        # NOW WE PERFORM PREPROCESSING STEPS ON THE (COMPLETELY) RAW DATA FROM THE .CNT FILES
        # LOW AND HIGH PASS FILTERING THAT SATISFIES ZERO-PHASE DESIGN
        filtered_data = data.copy().filter(lowfrq, hifrq)
        # REMOVE 60 HZ LINE NOISE FROM SIGNAL WITH NOTCH FILTER
        filtered_data.notch_filter(60, filter_length='auto', phase='zero')
        # WE NEED TO APPLY A COMMON AVERAGE REFERENCE TO USE MNE-ICALabel
        # UNCLEAR WHETHER AVERAGE SHOULD INCLUDE OR EXCLUDE THE EYE CHANNELS 
        # ALSO, WE WANT TO EXCLUDE BAD CHANNELS BEFORE THIS STEP SO WE MUST 
        # HAVE A PRELIMINARY CHECK OF CONSPICUOUSLY BAD CHANNELS
        
        # ******************** PUT QUICK EEG SCREENING FUNCTION HERE

        filtered_data = filtered_data.set_eeg_reference("average")
        # NOW WE DO ICA FOR ARTIFACT REMOVAL
        # UNCLEAR WHETHER EYE CHANNELS SHOULD BE INCLUDED OR NOT IN ICA
        ica = ICA(
            n_components=15,
            max_iter="auto",
            method="infomax",
            random_state=42,
            fit_params=dict(extended=True),
        )
        ica.fit(filtered_data)
        # NOW WE GET THE PREDICTED CLASSIFICATIONS OF THE INDEPENDENT COMPONENTS 
        ic_labels = label_components(filtered_data, ica, method="iclabel")
        # THEN EXCLUDE ANY ICs THAT ARE NOT CLASSIFIED AS 'BRAIN' OR 'OTHER'
        labels = ic_labels["labels"]
        print(ic_labels["labels"])
        exclude_idx = [idx for idx, label in enumerate(labels) if label not in ["brain", "other"]]
        # AND FINALLY WE RECONSTRUCT THE SIGNAL USING THE INCLUDED ICs
        reconst_data = filtered_data.copy()
        ica.apply(reconst_data, exclude=exclude_idx)
        # # TO DISPLAY THE CHANNEL PSDs, RAW DATA AND INDEPENDENT COMPONENTS
        # filtered_data.plot()
        # data.plot_psd()
        # filtered_data.plot_psd()
        ica.plot_sources(filtered_data, show_scrollbars=False, show=True)
        filtered_data.plot(show_scrollbars=False, title='filtered')
        reconst_data.plot(show_scrollbars=False, title='reconstr')
        for ch in channels:
            # NOW WE EXTRACT FILTERED SIGNAL FROM EACH CHANNEL AND SAVE IT IN .CSV FILE
            # AND AS .PNG, THE FORMER FOR USE IN MWT, AND THE LATTER FOR VISUAL INSPECTION
            fname = fname.replace('.','_')
            print(ch + ', '  + ' -- ' + fname[:-4])
            this_chan = reconst_data.get_data([ch])[0]
            datFN = ch + '_' + fname + '_' + str(samp_freq) + '.csv'
            np.savetxt(eeg_dir + datFN, this_chan.T , delimiter=',', header=ch, comments='')
            if do_plot_channels:
                figFN = ch + '_' + fname[:-4] + '_' + str(samp_freq) + '.png'
                plt.plot(this_chan)
                plt.title(ch + ', ' + cntList[f][0][-2:] + ' -- ' + fname[:-4])
                plt.savefig(base_dir + 'eeg_figures\\' + figFN)
                plt.clf()

if do_pac:    
    # GET A LIST OF ALL THE EEG FILES THAT HAVE BEEN EXTRACTED FROM .CNT FILES
    # AND HIGH/LOW PASS FILTERED
    eegList = csd.get_file_list(eeg_dir, 'csv')
    core_pheno = pd.read_csv(core_pheno_list)
    # SINCE VISIT INFORMATION IN THE FILE NAME IS DESIGNATED WITH LOWER CASE ALPHABET LETTERS
    # LET'S CREATE AN ALPHABET LIST
    visit_letters = list(string.ascii_lowercase)
    # NOW WE CREATE A DATAFRAME TO PUT VALUES INTO FOR DOWNSTREAM ANALYSIS AND/OR MACHINE LEARNING
    pacdat = pd.DataFrame()
    # WE GO THROUGH EACH OF THE EEG FILES FOUND AT eeg_path TO COLLECT ALL RELEVANT
    # META DATA AND TO CALCULATE PHASE AMPLITUDE COUPLING VALUES FOR DOWNSTREAM
    # ANALYSIS USING DEEP LEARNING NETWORKS AS WELL AS FOR TRADITIONAL STATISTICS
    for f in eegList:
        # FIRST WE NEED TO EXTRACT INFO CONTAINED IN THE FILE NAME
        # EXAMPLE FILE NAME: 'F7_eec_3_d1_10037021_32_cnt_500.csv'
        thisSub = int(csd.get_sub_from_fname(f[1], id_pos))
        thisVisitCode = csd.get_sub_from_fname(f[1], visit_pos)[0]
        thisRun = csd.get_sub_from_fname(f[1], visit_pos)[1]
        thisTask = csd.get_sub_from_fname(f[1], task_pos)
        thisChan = csd.get_sub_from_fname(f[1], chan_pos)
        
        # NEXT WE LOOK IN THE core_pheno TABLE TO LOOK UP THE SUBJECT ID FROM THE EEG FILE NAME
        # THE srow DATAFRAME CONTAINS MOST OF THE INFO WE NEED FOR COGA ANALYSIS PURPOSES
        srow = core_pheno[core_pheno['ID'] == thisSub]
        # IT IS POSSIBLE THAT THERE IS NO ENTRY FOR A SUBJECT SO WE WANT TO SKIP IF THERE ISN'T ONE IN core_pheno
        if len(srow) == 0:
            with open(base_dir + 'errors_from_core_pheno.txt', 'a') as ff:
                ff.write(str(thisSub) + '\t MISSING FROM core_pheno \n')
            continue
        # GET TOTAL NUMBER OF VISITS BY COUNTING THE NUMBER OF FIVES IN THE INTERVIEW COLUMNS. WE DO THIS TO BE
        # DYNAMICALLY COMPATIBLE WITH FUTURE VERSIONS OF core_pheno FILE
        interview_columns = np.where(np.isin(srow.columns, interview_col_names))[0]
        # WE MAKE A LIST OF BOOLEANS FOR ALL INTERVIEW VISITS THAT HAVE OCCURRED FOR THIS SUBJECT, 1=FALSE, 5=TRUE
        which_visits = srow.iloc[0, interview_columns] == 5
        # NOW USE THE BOOLEAN LIST TO GET THE COLUMN NAMES FOR THOSE INTERVIEW VISITS
        visit_list = [interview for (interview, visit_bool) in zip(interview_col_names, which_visits) if visit_bool]
        # ALL WE NEED ARE THE NUMERIC DESIGNATIONS OF INTERVIEW VISIT TYPES
        visit_list = [v.replace('intvw_p', '') for v in visit_list ]
        # NOW WE CAN MATCH UP THE VISIT INFORMATION FROM THE FILE NAME TO GET THE
        # CORRESPONDING NUMERIC VISIT DESIGNATION
        thisVisit = [visit_letters.index(thisVisitCode)][0]
        # NOW WE WANT TO GET THE CORRECT VALUE FROM visit_list SO WE CAN EXTRACT INFO FROM core_pheno FOR THE
        # QUALITY OF THE INTERVIEW DURING THE VISIT REPRESENTED BY THIS CURRENT EEG RECORDING FILE. QUALITY DEFINED AS
        # HOW TRUTHFUL, ACCURATE, RELIABLE BY THE INTERVIEWER, HIGH TO LOW, 1-4
        # interviewers rating (1=NO DIFFICULTY,2=SOME PROBLEMS,3=MAJOR DIFFICULTY,4=IMPOSSIBLE TO RATE,.A=Not Interviewed)
        # IT IS POSSIBLE THAT THE FILE NAME CONTAINS INCORRECT VISIT CODE, 
        # E.G., eec_1_b1_40133020_cnt_256 HAS ONLY ONE VISIT LISTED IN core_pheno
        # SO WE WRITE THAT INTO OUR ERROR TEXT FILE FOR LATER REVIEW AND SKIP THIS FILE
        try:
            visit_conversion = visit_list[thisVisit]
        except Exception as e:
            with open(base_dir + 'errors_from_core_pheno.txt', 'a') as ff:
                ff.write(str(thisSub) + '\t' + str(e) + ' - FILENAME ' + f[1] + ' LISTED AS 2ND RECORDING WHEN ONLY 1 RECORDING IN core_pheno\n')
            continue    
        thisVisitInterviewQuality = srow.iloc[0]['int_rating_p' + visit_list[thisVisit]]
        thisVisitAge = srow.iloc[0]['age_p' + visit_list[thisVisit]]
        
        # DETERMINE WHETHER SUBJECT HAS BEEN DIAGNOSED WITH ALCOHOLISM TYPES
        # IF ald5dx THEN PROCESS ald5_first_whint
        [aud_then, aud_now, aud_visits_ago] = csd.get_diagnosis(srow, visit_list, thisVisit, 'ald5dx', 'ald5_first_whint')
        # IF alc_abuse THEN PROCESS alab_first_whint
        [alab_then, alab_now, alab_visits_ago] = csd.get_diagnosis(srow, visit_list, thisVisit, 'alc_abuse', 'alab_first_whint')
        # IF alc_dep_dx THEN PROCESS aldp_first_whint
        [ald_then, ald_now, ald_visits_ago] = csd.get_diagnosis(srow, visit_list, thisVisit, 'alc_dep_dx', 'aldp_first_whint')

        
        # NEED TO OPEN EACH FILE AND PUT EEG VOLTAGE VALUES INTO AN ARRAY
        #       MUST INCLUDE CODE TO MAKE SURE ALL SIGNALS ARE THE SAME LENGTH
        # WE CAN THEN COMPARE PHASE AMPLITUDE COUPLING IN EEG OF SUBJECTS WITH AUD
        #       COMPARED TO THOSE THAT DO NOT HAVE AUD
        
    
        # FINALLY WE PUT ALL THE INFO AND PAC CALCULATIONS INTO A ONE ROW DATAFRAME 
        # TO ADD TO THE BIG DATAFRAME pacdat.
        df = pd.DataFrame({'ID': [srow.iloc[0]['ID']],
                           'site': [coga_sites[str(srow.iloc[0]['ID'])[0]]],
                           'sex': [sex_dict[srow.iloc[0]['sex']]],
                           'race': [srow.iloc[0]['race'][2:len(srow.iloc[0]['race'])-1]],
                           'hispanic': [hisp_dict[srow.iloc[0]['hisp']]],
                           'task': [thisTask],
                           'channel': [thisChan],
                           'this_visit': [thisVisit + 1],
                           'this_run': [thisRun],
                           'total_visits': [len(visit_list)],
                           'age_this_visit': [thisVisitAge],
                           'AUD_this_visit': [aud_then],
                           'ALAB_this_visit': [alab_then],
                           'ALD_this_visit': [ald_then],
                           'visits_before_AUD': [aud_visits_ago],
                           'visits_before_ALAB': [alab_visits_ago],
                           'visits_before_ALD': [ald_visits_ago],
                           'AUD_now': [aud_now],
                           'ALAB_now': [alab_now],
                           'ALD_now': [ald_now],
                           'interview_quality': [thisVisitInterviewQuality],
                           'eeg_file_name': [f[1]]

                           })
        pacdat = pd.concat([pacdat, df])
    # FINALLY WE SAVE THE pacdat TABLE
    pacdat.to_csv('C:\\Users\\CRichard\\Documents\\COGA_eec\\pacdat.csv', index=False)