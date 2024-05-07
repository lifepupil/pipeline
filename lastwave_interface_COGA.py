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
# from scipy import signal
# from scipy.signal import sosfiltfilt, butter
# import seaborn as sns
import string
# import statistics as st
import scipy.stats as stats
# import sympy as sp
# import pyprep as pp

import mne
from mne.preprocessing import ICA
# import torch
# import torch.nn as nn
# import torch.optim as optim

# import mne_icalabel as mica
# from mne_icalabel import label_components
import os
# import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import coga_support_defs as csd


# INSTANCE VARIABLES
do_sas_convert =                    False              # TO CONVERT .SAS7PDAT FILES TO TABLES SO THAT SUBJECT METADATA CAN BE USED DOWNSTREAM
do_plot_eeg_signal_and_mwt =        False  # TO PLOT SIGNAL AND HEATMAP FOR A GIVEN FILE
do_filter_eeg_signal_cnt =          False    # TO DO LOW PASS, HIGH PASS, NOTCH FILTER TO REMOVE LINE NOISE FROM SIGNAL, AND ICA
make_data_table =                   False              # GENERATED A DATA TABLE WITH DEMOGRAPHIC INFO, ALCOHOLISM STATUS, AND MACHINE LEARNING INPUTS, E.G. BAND POWER
do_stats =                          False                   # FOR TRADITIONAL STATISTICAL ANALYSIS 
do_reshape_by_subject =             False       # RESHAPES pacdat SO THAT EACH ROW IS ONE SUBJECT-VISIT WITH ALL CHANNELS AT ALL FREQUENCY BANDS
relocate_images_by_alcoholism =     False
do_deep_learn =                     False               # USES DATA TABLE AS INPUT TO DEEP LEARNING NETWORK TRAINING AND TESTING
generate_pac_images =               False
do_resnet_chanxfreq =               False
do_bad_channel_check_table_gen =    False
do_bad_channel_figure_gen =         False
do_bad_channel_pacdat_update =      False
do_bad_channel_removal =            False
do_bad_channel_check =              False
do_filter_figures_by_subject =      False
do_filter_figures_by_condition =    False
do_resnet_image_conversion =        False
do_filter_by_subject =              True
do_cnn_pac =                        False
do_resnet_pac_regularization =      True
do_resnet_pac =                     False
resnet_to_logistic =                False


# PARAMETERS
base_dir = "E:\\Documents\\COGA_eec\\data\\"
write_dir = "D:\\COGA_eec\\"
    
base_dir = "E:\\Documents\\COGA_eec\\data\\"

# specific frequency bands
FREQ_BANDS = {"delta": [0.5, 4],
              "theta": [4, 8],
              "alpha": [8, 12],
              "low_beta": [12, 20],
              "high_beta": [20, 30],
              'low_gamma': [30, 50]}



# PARAMETERS FOR make_data_table PHASE AMPLITUDE COUPLING USING TENSORPAC
if make_data_table:
    # TO GET SUBJECT-WISE STATS POINT source_dir TO A data FOLDER AND SET whichEEGfileExtention TO cnt
    # OR TO GENERATE TABLE FOR EACH EEG CHANNEL FOR USE IN DEEP LEARNING NETWORKS THEN
    # POINT source_dir TO A cleaned_data FOLDER AND SET whichEEGfileExtention TO cnt
    whichEEGfileExtention = 'csv'
    core_pheno_list = 'C:\\Users\\lifep\\OneDrive\\Documents\\COGA_sub_info\\core_pheno_20201120.csv'
    # core_pheno_list = 'D:\\COGA\\core_pheno_20201120.csv'
    
    if whichEEGfileExtention=='csv':
        source_dir = write_dir + "cleaned_data\\"
        chan_pos = 0
        task_pos = 1
        visit_pos = 3
        id_pos = 4
    elif whichEEGfileExtention=='cnt':
        source_dir = write_dir + "data\\"
        chan_pos = 0
        task_pos = 0
        visit_pos = 2
        id_pos = 3

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
    lowfrq = 1              # LOW PASS FREQUENCY, RECOMMENDED SETTING TO 1 HZ IF USING mne-icalabel
    hifrq = 100             # HIGH PASS FREQUENCY
    maxZeroPerc = 0.5       # PERCENTAGE OF ZEROS IN SIGNAL ABOVE WHICH CHANNEL IS LABELED 'BADS'
    do_plot_channels = True # TO GENERATE PLOTS OF THE CLEANED EEG SIGNAL
    mpl.rcParams['figure.dpi'] = 300 # DETERMINES THE RESOLUTION OF THE EEG PLOTS
    eye_blink_chans = ['X', 'Y'] # NAMES OF CHANNELS CONTAINING EOG
    institutionDir = 'uconn' # suny, indiana, iowa, uconn, ucsd, washu


if generate_pac_images:
    pac_path = 'D:\\COGA_eec\\pac_figures\\'

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
    cntList = csd.get_file_list(base_dir + institutionDir + '\\', 'cnt')
    # IN CASE THIS CODE HAS BEEN PARTIALLY RUN ON RAW DATA ALREADY
    # WE WANT TO FIND OUT WHAT OUTPUT FILES HAVE ALREADY BEEN GENERATED 
    # SO WE CAN EXCLUDE THEM FROM THE ORIGINAL LIST AND ONLY PROCESS RAW DATA
    # FILES THAT HAVEN'T YET BEEN PROCESSED. 
    # NOW WE REMOVE COMPLETED FILES FROM THE MAIN LIST OF FILES TO PROCESS, I.E., FROM cntList
    cntList = csd.remove_completed_files_from_list(cntList, write_dir + 'cleaned_data\\', institutionDir)
    totalFileCount = str(len(cntList))
    #  WE GO THROUGH EACH OF THE FILES IN cntList 
    for f in range(len(cntList)):
        fname = cntList[f][1]
        print('\n\n\nWORKING ON ' + str(f+1) + ' OF THE ' + totalFileCount + ' CNT FILES\n' + fname + '\n\n')
        path_and_file = cntList[f][0] + '\\' + fname
        try:
            data = mne.io.read_raw_cnt(path_and_file, preload=True, verbose=False)
            # print(data.info["bads"])
            # data.plot()
            # a=1
        except Exception as e:
            with open(base_dir + 'errors_from_core_pheno.txt', 'a') as bf:
                bf.write(str(fname) + '\t ' + str(e) + '\n')
            continue                
        
        data.drop_channels(['BLANK'], on_missing='warn')
        
        # ATTEMPT AT USING pyprep BUT IT WAS REJECTING CHANNELS THAT VISIBLY WERE NOT BAD
        # SHOULD BE TRIED AGAIN BY PLAYING WITH THE ARGUMENTS FOR THE pyprep.PrepPipeline
        # FUNCTION CALL, OR BY USING THE pyprep.NoisyChannels CLASS
        # data.set_channel_types({'X': 'eog', 'Y': 'eog'})
        # filtered_data = data.copy().filter(lowfrq, hifrq)
        # filtered_data.notch_filter(60, filter_length='auto', phase='zero')
        # prep_params = {
        #     "ref_chs": "eeg",
        #     "reref_chs": "eeg",
        #     "line_freqs": np.arange(60, int(data.info['sfreq']) / 2, 60),
        # }
        # prep = pp.PrepPipeline(filtered_data, prep_params, filtered_data.get_montage())
        # prep.fit()
        # print("Bad channels: {}".format(prep.interpolated_channels))
        # print("Bad channels original: {}".format(prep.noisy_channels_original["bad_all"]))
        # print("Bad channels after interpolation: {}".format(prep.still_noisy_channels))
        # filtered_data.plot()
        
        channels = data.ch_names
        info = data.info
        # WE EXCLUDE THE BLANK CHANNEL AND RELABEL CHANNEL TYPES OF THE TWO EYE CHANNELS TO eog
        # ASSUMES THAT ALL CHANNELS ARE LABELED AS EEG WHETHER THEY ARE OR NOT
        for ch in eye_blink_chans:
            if ch in channels:
                data.set_channel_types({ch: 'eog'})
        # HOWEVER THE Y CHANNEL IS OFTEN FLAT SO THIS NEXT LINE IS OPTIONAL FOR DEBUG
        # WOULD BE NICE TO HAVE A FLAT CHANNEL DETECTOR THAT WORKS WELL UNLIKE pyprep ABOVE
        # data.info['bads'] = ['BLANK']
        # IN CASE YOU WANT TO SEE WHAT THE CHANNEL TYPE ASSIGNMENTS ARE JUST RUN LINE OF CODE BELOW
        # for i in range(len(info['chs'])): print([str(i) + ' ' + str(filtered_data.info['chs'][i]['kind'])])
        
        # ******************** EEG SCREENING FOR BAD CHANNELS BEFORE ARTIFACT IDENTIFICATION AND REMOVAL
        # for ch in channels:
        #     # SIMPLE WAY TO EXCLUDE FLAT CHANNELS ALTHOUGH SOME CHANNELS CAN BE 
        #     # FLAT BUT HAVE FEW ACTUAL ZEROS IN THE SIGNAL
        #     percZeros = len(np.where(data.get_data([ch])[0]==0))/len(data.get_data([ch])[0])
        #     if percZeros>=maxZeroPerc:
        #         data.set_channel_types({ch: 'bads'})        
        
        # WE WANT TO EXCLUDE ANY CHANNELS THAT AREN'T EEG OR eog JUST IN CASE WE HAVE ANY THERE
        eeg_indices = mne.pick_types(info, meg=False, eeg=True, ecg=False, eog=True, exclude=['bads'])
        mne.pick_info(info, eeg_indices, copy=False)
        # NOW WE GET SAMPLE RATE AND CHANNEL INFO
        samp_freq = int(info['sfreq'])  # sample rate (Hz)
        # NOW WE PERFORM PREPROCESSING STEPS ON THE (COMPLETELY) RAW DATA FROM THE .CNT FILES
        # LOW AND HIGH PASS FILTERING THAT SATISFIES ZERO-PHASE DESIGN
        filtered_data = data.copy().filter(lowfrq, hifrq)
        # REMOVE 60 HZ LINE NOISE FROM SIGNAL WITH NOTCH FILTER
        filtered_data.notch_filter(60, filter_length='auto', phase='zero', verbose=False)
        # WE NEED TO APPLY A COMMON AVERAGE REFERENCE TO USE MNE-ICALabel
        # UNCLEAR WHETHER AVERAGE SHOULD INCLUDE OR EXCLUDE THE EYE CHANNELS 
        # ALSO, WE WANT TO EXCLUDE BAD CHANNELS BEFORE THIS STEP SO WE MUST 
        # HAVE A PRELIMINARY CHECK OF CONSPICUOUSLY BAD CHANNELS            
        
        # SETS AVERAGE REFERENCE 
        filtered_data = filtered_data.set_eeg_reference("average")
        
        # NOW WE DO ICA FOR ARTIFACT REMOVAL
        # UNCLEAR WHETHER EYE CHANNELS SHOULD BE INCLUDED OR NOT IN ICA
        ica = ICA(
            n_components=15,
            max_iter="auto",
            random_state=42,
            method="infomax",
            fit_params=dict(extended=True),
            verbose=False
        )
        ica.fit(filtered_data)
        # IN AN ATTEMPT TO IMPROVE THE EYE BLINK IDENTIFICATION I TRIED USING THE 
        # create_eog_epochs FUNCTION AND IT FOUND NO EYE BLINKS EVEN THOUGH THE 
        # EOG MATCHED ALMOST PERFECTLY ONE OF THE ICs SO ABANDONING FOR NOW
        # eog_epochs = create_eog_epochs(filtered_data,
        #                                flat={'eog': 250e-6}
        #                                )
        
        # SINCE ica HAS THE EOG CHANNEL LABELED AS EEG (THIS CAN BE CONFIRMED WITH 
        # ica.get_channel_types() AT THE CONSOLE) THE PREDICTION FROM mne_icalabel 
        # LABELS WHAT IS CLEARLY EYE BLINK COMPONENT AS BRAIN SO HERE WE ARE USING THE 
        # find_bads_eog FUNCTION TO FIND THE MOST LIKELY EYE BLINK IC AND EXCLUDES IT
        # BUT FIRST WE NEED TO SEE WHETHER THERE ARE ANY eog CHANNELS IN data AND
        # IF NOT THEN WE USE THE FRONTAL POLAR CHANNELS FOR EYE BLINK DETECTION
        if 'eog' not in data.get_channel_types():
            # IT TURNS OUT THAT THERE ARE SOME FILES THAT HAVE NEITHER 
            # LABELED EOG NOR FP1 OR FP2 SO WE SKIP THESE FILES 
            # ONLY EXAMPLE FOUND SO FAR IS ONE WHERE CHANNELS ARE ALL NUMBERS
            # SEE: eec_1_a1_50302007.cnt
            if any(ch in channels for ch in ('FP1', 'FP2')):
                eog_indices, eog_scores = ica.find_bads_eog(filtered_data,
                                                            ch_name=['FP1', 'FP2'],
                                                            verbose=False
                                                            )
            else:
                with open(base_dir + 'errors_from_core_pheno.txt', 'a') as bf:
                    bf.write(str(fname) + '\t ' + 'MISSING EOG AND FRONTAL CHANNELS ' + '\n')
                continue 
        else:
            eog_indices, eog_scores = ica.find_bads_eog(filtered_data,
                                                        verbose=False
                                                        )

        # THIS SECTION EMPLOYS THE mne-icalabels AND IS ALSO BEING LEFT ALONE FOR NOW
        # BECAUSE IT ALSO MISSED CONSPICUOUSLY REAL ICs FOR EYE BLINKS HOWEVER
        # WE COMBINE THE NON-BRAIN ICs FROM BOTH eog_indices AND exclude_idx TO 
        # COVER ALL POSSIBLE NON-BRAIN ARTIFACTS FOR REMOVAL
        ic_labels = label_components(filtered_data, ica, method="iclabel")
        labels = ic_labels["labels"]
        # TO PRINT OUT LIST OF LABELS BY INDEX FOR QUICK INTERPRETATION OF IC PLOT
        # for i in range(len(labels)): print([str(i) + ' ' + labels[i]])
        
        # THEN EXCLUDE ANY ICs THAT ARE NOT CLASSIFIED AS 'BRAIN' OR 'OTHER'
        exclude_idx = [idx for idx, label in enumerate(labels) if label not in ["brain", "other"]]
        
        # AND FINALLY WE RECONSTRUCT THE SIGNAL USING THE INCLUDED ICs
        # reconst_data = filtered_data.copy()
        # COMBINING ALL NON-BRAIN ICs AND REMOVING THEM
        ic_to_remove = [*set(exclude_idx + eog_indices)]
        ica.exclude = ic_to_remove
        ica.apply(filtered_data)
        
        # # UNCOMMENT TO DISPLAY THE CHANNEL PSDs, RAW DATA AND INDEPENDENT COMPONENTS
        # filtered_data.plot()
        # data.plot_psd()
        # filtered_data.plot_psd()
        
        # # UNCOMMENT THE NEXT THREE LINES TO GET PLOTS OF ICs, AND BEFORE AND AFTER RECONSTRUCTION OF SIGNALS
        # ica.plot_sources(filtered_data, show_scrollbars=False, show=True)
        # filtered_data.plot(show_scrollbars=False, title='filtered')
        # reconst_data.plot(show_scrollbars=False, title='reconstr')
        
        for ch in channels:
            # NOW WE EXTRACT FILTERED SIGNAL FROM EACH CHANNEL AND SAVE IT IN .CSV FILE
            # AND AS .PNG, THE FORMER FOR USE IN MWT, AND THE LATTER FOR VISUAL INSPECTION
            fname = fname.replace('.','_')
            print(ch + ', '  + ' -- ' + fname[:-4])
            this_chan = filtered_data.get_data([ch])[0]
            datFN = ch + '_' + fname + '_' + str(samp_freq) + '.csv'
            try:
                np.savetxt(write_dir + 'cleaned_data\\' + datFN, this_chan.T , delimiter=',', header=ch, comments='')
            except Exception as e:
                with open(base_dir + 'errors_from_core_pheno.txt', 'a') as bf:
                    bf.write(str(datFN) + '\t ' + str(e) + '\n')
                continue        
            # THIS CONTROLS WHETHER CHANNEL PLOTS ARE GENERATED OR NOT
            if do_plot_channels:
                figFN = ch + '_' + fname[:-4] + '_' + str(samp_freq) + '.png'
                plt.plot(this_chan)
                # plt.ylim((-50/1000000),(50/1000000))
                plt.title(ch + ', ' + institutionDir.upper() + ' -- ' + fname[:-4])
                # plt.show()
                plt.savefig(write_dir + 'eeg_figures\\' + figFN)
                plt.clf()

if make_data_table:    
    source_dir = 'D:\\COGA_eec\\cleaned_data\\'
    base_dir = "D:\\COGA_eec\\"

    # GET A LIST OF ALL THE .CSV FILES OF SUBJECT-VISIT-EEG CHANNELS 
    # THAT HAVE BEEN EXTRACTED FROM .CNT FILES AND HIGH/LOW PASS FILTERED
    # OR CAN USE TO GENERATE INFO TABLE FROM .CNT FILES BY PASSING 'cnt' INSTEAD OF 'csv'
    eegList = csd.get_file_list(source_dir, whichEEGfileExtention)
    core_pheno = pd.read_csv(core_pheno_list)
    # SINCE VISIT INFORMATION IN THE FILE NAME IS DESIGNATED WITH LOWER CASE ALPHABET LETTERS
    # LET'S CREATE AN ALPHABET LIST
    visit_letters = list(string.ascii_lowercase)
    # NOW WE CREATE A DATAFRAME TO PUT VALUES INTO FOR DOWNSTREAM ANALYSIS AND/OR MACHINE LEARNING
    pacdat = pd.DataFrame()
    # WE GO THROUGH EACH OF THE EEG FILES FOUND AT base_dir TO COLLECT ALL RELEVANT
    # META DATA AND TO CALCULATE PHASE AMPLITUDE COUPLING VALUES FOR DOWNSTREAM
    # ANALYSIS USING DEEP LEARNING NETWORKS AS WELL AS FOR TRADITIONAL STATISTICS
    for f in eegList:
        # f = eegList[3886]
        print(f[1])
        # REMOVE THE FILE EXTENSION
        f[1]  = f[1][0:-4]
        # WE NEED TO EXTRACT INFO CONTAINED IN THE FILE NAME
        # EXAMPLE FILE NAME: 'F7_eec_3_d1_10037021_32_cnt_500.csv'
        thisSub = csd.get_sub_from_fname(f[1], id_pos)
        # SINCE THERE ARE SOMETIMES SUBJECT INFO THAT DON'T CONTAIN ALL NUMBERS
        # AND WE WANT TO IGNORE THOSE FILES WE HAVE THIS TEST BELOW
        if any([x.isalpha() for x in thisSub]):
            continue
        else:
            thisSub = int(thisSub)
        thisVisitCode = csd.get_sub_from_fname(f[1], visit_pos)[0].lower()
        thisRun = csd.get_sub_from_fname(f[1], visit_pos)[1]
        thisTask = csd.get_sub_from_fname(f[1], task_pos)
        thisChan = csd.get_sub_from_fname(f[1], chan_pos)
        thisSampFreq = f[1].split('_')
        thisSampFreq = int(thisSampFreq[len(thisSampFreq)-1])
        # THERE ARE VISITS THAT ARE LABELED WITH 'L' THAT DON'T FIT THE 
        # FILENAMING CONVENTION THAT I WAS GIVEN INITIALLY SO FOR NOW
        # WE ARE IGNORING .CNT FILES THAT HAVE 'L' FOR VISIT INFO
        if thisVisitCode=='l':
            with open(base_dir + 'errors_from_core_pheno.txt', 'a') as ff:
                ff.write(str(thisSub) + '\ta LIFESPAN project visit \n')
            continue
        # NEXT WE LOOK IN THE core_pheno TABLE TO LOOK UP THE SUBJECT ID FROM THE EEG FILE NAME
        # THE srow DATAFRAME CONTAINS MOST OF THE INFO WE NEED FOR COGA ANALYSIS PURPOSES
        srow = core_pheno[core_pheno['ID'] == thisSub]
        # ALL IDs IN core_pheno ARE UNIQUE SO NO POSSIBILITY OF THERE BEING len(srow) > 1
        # IT IS POSSIBLE THAT THERE IS NO ENTRY FOR A SUBJECT SO WE WANT TO SKIP IF THERE ISN'T ONE IN core_pheno
        if len(srow) == 0:
            with open(base_dir + 'errors_from_core_pheno.txt', 'a') as ff:
                ff.write(str(thisSub) + '\tMISSING FROM core_pheno \n')
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
        # THERE IS AT LEAST ONE SUBJECT-VISIT THAT HAS AN ERROR IN core_pheno_20201120
        # NAMELY eec_3_c1_30025017_32.cnt WHICH HAS NO ENTRY FOR FIRST VISIT IN 
        # PHASE 4, I.E., intvw_p4. THERE ARE FOUR FILES FROM THAT SUBJECT: 
        # b1, c1, d1, AND l1 SO FOR NOW WE JUST WRITE INFO INTO THE ERROR LOG. 
        # THIS NEXT IF STATEMENT ONLY DEALS WITH PHASE 4 VISITS
        if not which_visits['intvw_p4'] and any(which_visits[3:11]):
            with open(base_dir + 'errors_from_core_pheno.txt', 'a') as ff:
                ff.write(str(thisSub) + '\tFILENAME ' + f[1] + ' MISSING VISIT INFORMATION IN core_pheno\n')
            continue 
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
        
        freq_band_psds = csd.get_band_psds(f, thisSampFreq, FREQ_BANDS)
        eeg_dur = csd.get_recording_duration(f, thisSampFreq)
        
        # AF1_eec_1_a1_10197014_256 test flat eeg

        # FINALLY WE PUT ALL THE INFO AND PAC CALCULATIONS INTO A ONE ROW DATAFRAME 
        # TO ADD TO THE BIG DATAFRAME pacdat.
        df = pd.DataFrame({'ID': [srow.iloc[0]['ID']],
                           'site': [coga_sites[str(srow.iloc[0]['ID'])[0]]],
                           'sex': [sex_dict[srow.iloc[0]['sex']]],
                           'race': [srow.iloc[0]['race'][2:len(srow.iloc[0]['race'])-1]],
                           'hispanic': [hisp_dict[srow.iloc[0]['hisp']]],
                           'task': [thisTask],
                           'channel': [thisChan],
                           'duration': [eeg_dur],
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
                           'eeg_file_name': [f[1]],
                           'delta': freq_band_psds[0],
                           'theta': freq_band_psds[1],
                           'alpha': freq_band_psds[2],
                           'low_beta': freq_band_psds[3],
                           'high_beta': freq_band_psds[4],
                           'gamma': freq_band_psds[5]
                           })
        pacdat = pd.concat([pacdat, df])
        
    # # TO LOAD THE FILE FOR DEBUG
    # pacdat = pd.read_csv(base_dir + 'pacdat.csv')
    
    # WE SORT THE DATAFRAME FOR HUMAN READABILITY
    pacdat = pacdat.sort_values(by='ID')
    
    # WE ADD A COLUMN TO SEPARATE ALCOHOLICS NOW FROM CONTROLS NOW FOR DEMOGRAPHICS INFO
    pacdat['alcoholic'] = pacdat.apply(lambda r: any([r['AUD_now'], r['ALAB_now'],r['ALD_now']]), axis=1)
    # pacdat['last_age'] = pacdat.apply(lambda r: r['age_this_visit'] if r['this_visit']==r['total_visits'] else 0, axis=1)
    # pacdat['first_age'] = pacdat.apply(lambda r: r['age_this_visit'] if r['this_visit']==1 else 0, axis=1)
    # pacdat['second_age'] = pacdat.apply(lambda r: r['age_this_visit'] if r['this_visit']==2 else 0, axis=1)
    
    # FINALLY WE SAVE THE pacdat TABLE
    pacdat.to_csv(base_dir + 'pacdat' + '.csv', index=False)


    if whichEEGfileExtention=='cnt':
        # NOW WE EXECUTE THIS CODE TO GET DEMOGRAPHICS FROM SUBJECTS
        tbl = pacdat.copy()
        # WE REMOVE ANY COLUMNS THAT WOULD PREVENT REMOVAL ALL DUPLICATES FOR A GIVEN SUBJECT
        tbl.drop(tbl.iloc[:, 5:22], inplace=True, axis=1)
        # tbl = tbl.loc[tbl['first_age']>0]
        tbl = tbl.drop_duplicates()
        tbl.to_csv(base_dir + 'pd.csv', index=False)
        csd.print_demo_vals(tbl)

if do_stats:
    # import statsmodels.api as sm
    from statsmodels.formula.api import ols
    from pingouin import ancova
    
    # OPEN pacdat DATA TABLE
    pacdat = pd.read_csv(write_dir + 'pacdat.csv')
    g1 = pacdat['delta'][(pacdat['alcoholic']==True) & (pacdat['channel']=='CZ') & (pacdat['sex']=='M')]
    g2 = pacdat['delta'][(pacdat['alcoholic']==False) & (pacdat['channel']=='CZ') & (pacdat['sex']=='M')]
    stats.ttest_ind(g1, g2)

    model = ols('delta ~ alcoholic + age_this_visit + sex', data=pacdat).fit()
    print(model.summary())
    
    dt = pacdat[(pacdat.duration>=180) & (pacdat.channel=='CZ')]
    dt = dt[['sex','channel','age_this_visit','delta','theta','alpha','low_beta','high_beta','gamma','alcoholic']]
    dt['alcoholic'] = dt['alcoholic'].astype(float)
    # IN CASE NEED TO USE DUMMY VARIABLE CODING UNCOMMENT LINE BELOW
    # dt = pd.get_dummies(dt,columns=['sex'],dtype=float)
    ancova(data=dt[dt['sex']=='M'], dv='high_beta', covar='age_this_visit', between='alcoholic')
    
if do_reshape_by_subject:
    # READ ME !!!
    # NOTA BENE:
    # AS OF NOVEMBER 30, 2023 THIS CODE NEEDS TO BE REFACTORED TO PERMIT THE 
    # REASONABLY FAST GENERATION OF CHANNEL X FERQUENCY BAND IMAGES, THE
    # CONVERSION OF THOSE IMAGES TO 224 X 224 SPECIFICATIONS REQUIRED BY 
    # RESNET-50, OR THE GENERATION OF A SUBJECT-VISIT TABLE SAVED AS A PICKLE
    # FILE. iF YOU TRY TO RUN ALL OF THEM AT THE SAME TIME, PROCESSING WILL 
    # MOVE AT A SNAILS PACE, BUT IF THEY ARE RUN SEPARATELY THEN THEY ARE 
    # MUCH, *MUCH* FASTER (HENCE THE NEED TO REFACTOR THIS CODE)
    # FINALLY THE USE OF do_reshape_by_subject STILL HAS MOST LINES, 
    # COMMENTED OUT, TO ADD A VECTOR OF CHANNEL X FREQUENCY BAND TO THE TABLE
    # HOWEVER SINCE IT PERFORMED SO BADLY AS INPUT INTO OUR DEEP LEARNING
    # MODEL, THESE LINES HAVE ONLY BEEN KEPT FOR UNLIKELY LEGACY PURPOSES 
    
    import seaborn as sns
    from PIL import Image
    
    exclude_chans = ['C5', 'TP7', 'P6', 'CP3', 'POZ', 'P1', 'C2', 'C1', 'OZ', 'FC4', 'FPZ', 'F5', 'FT7', 'F6', 'FC3', 'CP4', 'PO7', 'F1', 'TP8', 'CPZ', 'P2', 'FT8', 'PO8', 'AFZ', 'AF7', 'FCZ', 'P5', 'F2', 'AF8', 'C6']
    
    # OPEN pacdat DATA TABLE
    pacdat = pd.read_csv(write_dir + 'pacdat.csv')
    # EXCLUDE ROWS WITH NAs
    pacdat = pacdat.dropna()
    # GET LIST OF ALL FILES USED TO GENERATE pacdat
    paclist = set(['_'.join(il.split('_')[1:]) for il in pacdat.eeg_file_name])
    # LET'S GET A LIST OF ALL THE SUBJECT IDs THAT NEED TO BE PROCESSED
    imgList = csd.get_file_list('D:\\COGA_eec\\chan_hz_figures\\', 'jjpg')
    imgList = set([f[1][:-4] for f in imgList])   
    remaining = paclist.difference(imgList)
    ids = set([int(s.split('_')[3]) for s in remaining])
    
    # CREATE OUR OUTPUT DATAFRAME
    pacColumns = ['ID',
     'site',
     'sex',
     'race',
     'hispanic',
     'duration',
     'this_visit',
     'this_run',
     'total_visits',
     'age_this_visit',
     'AUD_this_visit',
     'ALAB_this_visit',
     'ALD_this_visit',
     'visits_before_AUD',
     'visits_before_ALAB',
     'visits_before_ALD',
     'AUD_now',
     'ALAB_now',
     'ALD_now',
     'interview_quality',
     'alcoholic',
     'chan_num',
     'chan_hz_path']
    
    dat = pd.DataFrame(index=range(len(paclist)), columns=pacColumns)
    # dat = pd.DataFrame(index=range(len(ids), columns=range(len(pacdat.columns)))
    
    # LET'S SET UP A COUNTER SO WE CAN TRACK PROGRESS OF RESHAPING
    sub_count  = 0
    total_sub = len(ids)
    # FIRST WE GET ALL OF THE ROWS FOR A GIVEN SUBJECT
    for id in ids: 
        subj = pacdat[pacdat.ID==id]
        # NOW WE NEED TO FIND OUT WHAT VISITS THERE ARE FOR THIS SUBJECT 
        # IT IS POSSIBLE THAT THERE ARE MISSING VISIT NUMBERS SO WE NEED
        # TO MAKE SURE WE ACCOUNT FOR THAT BY ONLY CYCLING THROUGH EXISTING 
        # VISIT NUMBERS AKA this_visit
        visits = list(set(subj.this_visit))
        for v in visits:
            subvis = subj[subj.this_visit==v]
            
            # NEXT WE NEED TO EXCLUDE NON-EEG CHANNELS WHICH WE WILL DO
            # ONE AT A TIME BECAUSE SOME SUBJECTS MAY OR MAY NOT HAVE 
            # ALL THREE CHNNELS WE WANT TO EXCLUDE 
            subvis = subvis[(subvis.channel!='X')]
            subvis = subvis[(subvis.channel!='Y')]
            subvis = subvis[(subvis.channel!='BLANK')]
            # SINCE NOT ALL RECORDING SESSIONS USED 31 CHANNELS WE NEED TO 
            # EXCLUDE THE EXTRA CHANNELS TO HAVE CONSISTENCE IN THE 
            # CHANNEL X FREQUENCY BAND IMAGES
            orig_channum = len(subvis.channel)
            if orig_channum==61:
                subvis = subvis[subvis.channel.isin(exclude_chans)==False]
            # IT APPEARS THAT THERE IS AT LEAST ONE SUBJECT-VISIT WITH THE 
            # SAME this_visit VALUE FOR TWO DIFFERENT FILES SO WE NEED TO 
            # SCREEN FOR THIS AND AT LEAST FOR NOW NOTE IT IN AN ERROR FILE
            if not subvis['channel'].is_unique:
                with open(base_dir + 'errors_from_reshape.txt', 'a') as ff:
                    fn = '_'.join(subvis['eeg_file_name'].iloc[0].split('_')[1:])
                    ff.write(str(id) + '\tFILENAME ' + fn + ' DUPLICATE VISITS\n')
                continue
            # NEXT WE SORT CHANNELS FOR STANDARDIZATION OF LATER CHAN x FREQ JPG
            subvis = subvis.sort_values(by=['channel'])
            # AND THEN EXCLUDE FILENAME COLUMN SO THAT ALL COLUMN VALUES APART
            # FROM SPECTRAL POWER ARE ALL IDENTICAL
            # BUT FIRST WE GRAB THE SHARED FILE NAME STRING SO WE CAN USE IT 
            # LATER TO SAVE OUR HEATMAP OF THE CHANNEL X FREQUENCY BAND HEATMAP
            figFN =  '_'.join(subvis['eeg_file_name'].iloc[0].split('_')[1:]) + '.jpg'            

            subvis = subvis.loc[:, subvis.columns != 'eeg_file_name']
            
            # THIS BLOCK ALLOWS TO TO PUT ALL CHANNELS AND FREQUENCY BANDS INTO A VECTOR
            if 0:
                # NOW FOR THE RESHAPING INTO A SINGLE ROW
                chan_Hz = subvis.pivot(index='ID',columns='channel',values=['delta','theta','alpha','low_beta','high_beta','gamma'])
                # CREATE COLUMN LABELS OF CHANNEL_FREQUENCY BAND
                chan_Hz.columns = chan_Hz.columns.swaplevel().map('_'.join)
                # CLEAN UP chan_Hz FOR CONCATENATION
                chan_Hz = chan_Hz.reset_index()
                chan_Hz = chan_Hz.drop('ID',axis=1)
            
            # THIS BLOCK CAN BE USED TO TURN MATRIX OF VALUES INTO AN IMAGE
            if 0:
                dta = np.sqrt(subvis[['delta','theta','alpha']])
                lhbg = np.sqrt(subvis[['low_beta','high_beta','gamma']])
                fbs = dta.join(lhbg)
                img = sns.heatmap(fbs,vmin=0,vmax=1, xticklabels=False,yticklabels=False, cbar=False,cmap='hsv')
                # NOW WE SAVE IT
                fig = plt.Axes.get_figure(img)
                # FINALLY WE SAVE IT AS A JPG -    THIS WILL BE IMPORTANT FOR RESIZING 
                # THIS IMAGE FOR RESNET-50 USING PIL PACKAGE 
                fig.savefig(write_dir + 'chan_hz_figures\\' + figFN, bbox_inches='tight')
                
            # THIS BLOCK USED TO RESIZE IMAGES FOR RESNET-50
            if 0:
                img2 = Image.open(write_dir + 'chan_hz_figures\\' + figFN)
                print(figFN)
                img2 = img2.resize((224, 224))
                img2.save(write_dir + 'chan_hz_figures\\' + figFN)
            
            # plt.plot(this_chan)
            # # plt.ylim((-50/1000000),(50/1000000))
            # plt.title(ch + ', ' + institutionDir.upper() + ' -- ' + fname[:-4])
            # # plt.show()
            # plt.savefig(write_dir + 'eeg_figures\\' + figFN)
            # plt.clf()            
            
            # THIS BLOCK GIVES US ALL SUBJECT INFO INTO ONE ROW
            # WE JUST NEED THE FIRST ROW OF THE DATAFRAME FOR THIS SUBJECT-VISIT
            orig_row = pd.DataFrame(data=subvis.iloc[0,:].values,index=subvis.iloc[0,:].index).T
            orig_row = orig_row.drop(['channel','task','delta','theta','alpha','low_beta','high_beta','gamma'], axis=1)
            orig_row['chan_num'] = orig_channum
            
            # # fin = pd.concat([orig_row,chan_Hz],axis=1)
            # img = subvis[['delta','theta','alpha','low_beta','high_beta','gamma']]
            # img = img.to_numpy()
            # orig_row['chan_hz'] = [img]

            orig_row['chan_hz_path'] =  write_dir + 'chan_hz_figures\\' + figFN
            # dat = pd.concat([dat,orig_row])
            dat.iloc[sub_count,:] = orig_row.iloc[0,:]
            sub_count += 1

    # WE SORT THE DATAFRAME FOR HUMAN READABILITY
    dat = dat.sort_values(['ID','this_visit'], ascending=True)
    # FINALLY WE SAVE THE RESHAPED TABLE
    # dat.to_csv(base_dir + 'chan_hz_dat' + '.csv', index=False)
    dat.to_pickle(base_dir  + 'chan_hz_dat.pkl')
    





if relocate_images_by_alcoholism:
    import shutil
    
    which_pacdat = 'pacdat_cutoffs_flat_25_excessnoise_25.pkl'
    whichEEGfileExtention = 'jpg'
    source_folder, targ_folder = 'chan_hz_figures_all','chan_hz'

    read_dir = 'D:\\COGA_eec\\' + source_folder + '\\'  #  BIOWIZARD
    base_dir = 'D:\\COGA_eec\\' #  BIOWIZARD
    
    pth = 'D:\\COGA_eec\\pac_figures\\'
    alcpth = 'D:\\COGA_eec\\pac_figures\\alcoholic\\'
    nonpth = 'D:\\COGA_eec\\pac_figures\\nonalcoholic\\'
    
    # CONSTANTS
    chan_i = 0 
    visit_i = 3 
    id_i = 4 
    
    # OPEN PICKLE FILE
    pacdat = pd.read_pickle(base_dir + which_pacdat)
    # dat = pd.read_pickle(base_dir + 'chan_hz_dat.pkl')
    subset = pacdat[(pacdat.channel=='FZ')]

    # alc = dat[(dat.AUD_this_visit==1) & (dat.channel=='FZ')]
    # nonalc = dat[(dat.AUD_this_visit==0) & (dat.channel=='FZ')]
    # alc = dat[(dat.AUD_this_visit==1) & (dat.age_this_visit>=25) & (dat.age_this_visit<=40) & (dat.sex=='F')]
    # nonalc = dat[(dat.AUD_this_visit==0) & (dat.age_this_visit>=25) & (dat.age_this_visit<=40) & (dat.sex=='F')]
    
    # GET FILE LIST WITH GIVEN EXTENSION FROM SOURCE FOLDER 
    figList = csd.get_file_list(base_dir + source_folder, whichEEGfileExtention)
    # MAKE A DATAFRAME TO ENRICH WITH ADDITIONAL COLUMNS DERIVED FROM FILENAME INFO
    fig_info = pd.DataFrame(figList, columns=['dir','fn'])
    
    c = [f.split('_')[chan_i] for f in  fig_info.fn]
    c = pd.DataFrame(c,columns=['channels'])
    fig_info.insert(0,'channels',c)

    visitCodeList = [f.split('_')[visit_i][0] for f in  fig_info.fn]
    visitCodeList = [csd.convert_visit_code(v) for v in visitCodeList]    
    v = pd.DataFrame(visitCodeList,columns=['this_visit'])
    fig_info.insert(0,'this_visit',v)
    
    v = [f.split('_')[id_i] for f in  fig_info.fn]
    v = pd.DataFrame(v,columns=['ID'])
    fig_info.insert(0,'ID',v)  
    
    # alc = dat[dat.AUD_this_visit==1]
    # nonalc = dat[dat.AUD_this_visit==0]
    # alc.reset_index(drop=True,inplace=True)

    for i in range(0,len(fig_info)):
        this_id = int(fig_info.loc[i,'ID'])
        this_chan = fig_info.loc[i,'channels']
        this_visit = fig_info.loc[i,'this_visit']
        
        this_subj_vis = subset[(subset.ID==this_id) & (subset.channel==this_chan) & (subset.this_visit==this_visit)]
        if this_subj_vis.empty:
            print('Missing ' + fig_info.loc[i,'fn'])
        else:
            this_dir = fig_info.loc[i,'dir']
            this_fn = fig_info.loc[i,'fn']
            
            this_alc_diag = this_subj_vis.AUD_this_visit.values[0]
            if this_subj_vis.AUD_this_visit.values[0]:
                diag_folder = 'alcoholic'
            else:
                diag_folder = 'nonalcoholic'
            
                        
            old_path_fn = this_dir + '\\' + this_fn
            
            new_path_fn = this_dir + '\\' + this_fn
            new_path_fn = new_path_fn.replace(source_folder, targ_folder + '\\' + diag_folder)
            print('Copying file ' + this_fn)
            shutil.copy(old_path_fn, new_path_fn)

    # for i in range(0,len(alc)):
    #     thisjpg = alc.iloc[i].chan_hz_path
    #     fn = thisjpg.split('\\')[-1]
    #     os.rename(thisjpg, alcpth + fn)
    
    # for i in range(0,len(nonalc)):
    #     thisjpg = nonalc.iloc[i].chan_hz_path
    #     fn = thisjpg.split('\\')[-1]
    #     os.rename(thisjpg, nonpth + fn)
    
    
    
    
    
if do_deep_learn:

    # OPEN QUESTIONS 
    # 1. WITH RESPECT TO CHANNEL X FREQ COMBINATIONS HOW MANY INPUTS ARE TOO MANY INPUTS?
    # 2. HOW MANY NODES IN FIRST LAYER GIVEN SIZE OF INPUT, E.G. 30 NODES FOR 185 INPUTS NOT SUFFICIENT
    # 
    # 
    
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Dropout
    from sklearn.model_selection import GroupShuffleSplit
    import keras as ks
    
    from keras.preprocessing.image import image
    from keras.preprocessing.image import img_to_array
    from keras.applications.resnet50 import preprocess_input    
    from keras.applications.imagenet_utils import decode_predictions
    
    write_dir = "E:\\Documents\\COGA_eec\\data\\"
    # OPEN dat DATA TABLE
    # dat = pd.read_csv(write_dir + 'chan_hz_dat.csv')
    dat =  pd.read_pickle(base_dir  + 'chan_hz_dat.pkl')
    
    # EXCLUDE ROWS WITH NAs
    # dat = dat.dropna()
    
    # REMOVE CHANNELS THAT ARE TOO SHORT IN DURATION - DUR > 180 SECONDS  (ALSO TOO LONG?)
    # ALSO NEED AT LEAST RIGHT NOW TO EXCLUDE ANY IMAGES THAT AREN'T MADE UP OF 61 CHANNELS 
    dl = dat[(dat.duration>=90) and (dat.chan_num==61)]
    
    # # REMOVE COLUMNS THAT ARE NOT NEEDED, E.G. UNLESS DOING LSTM LEAVE OUT VISIT ORDER
    # dl = dl[['sex','channel','age_this_visit','delta','theta','alpha','low_beta','high_beta','gamma','alcoholic']]
    
    # RECODE CATEGORICAL VALUES TO NUMERICS
    dl['alcoholic'] = [1 if alc == True else 0 for alc in dl['alcoholic']]
    # SPLIT TABLE INTO SEPARATE TABLES FOR INPUT AND OUTPUT VARIABLES
    outp = dl['alcoholic']
    inp = dl.iloc[:,22:207]
    
    leaky_relu = LeakyReLU(alpha=0.0001)
    
    model = Sequential()
    # model.add(BatchNormalization())
    model.add(Dense(20, input_shape=(len(inp.columns),), activation='elu'))
    # model.add(Dropout(0.2))
    # model.add(BatchNormalization())
    model.add(Dense(20, activation='relu'))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(50, activation='leaky_relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    
    optimizer = ks.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.fit(inp, outp, 
              epochs=100, 
              batch_size=16,
              shuffle=True,
              validation_split=0.2)
    
    loss = np.array(model.history.history['loss'])
    vloss = np.array(model.history.history['val_loss'])
    acc = np.array(model.history.history['accuracy'])
    vacc = np.array(model.history.history['val_accuracy'])
    x = np.linspace(0,len(acc)-1,len(acc))
    params_dl = str(model.history.params)
    
    _, accuracy = model.evaluate(inp, outp)
    print('Accuracy: %.2f' % (accuracy*100))
    
    predictions = (model.predict(inp) > 0.5).astype(int)
            
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(params_dl)
    ax.plot(x, acc, color='red', label='acc')
    ax.plot(x, loss, color='blue', label='loss')
    ax.legend()
    
    
    
if do_resnet_chanxfreq:
    # after unimpressive training using ImageNet,
    # tried setting weights to None,  
    # then tried making each_layer trainable 
    # can also change from categorical to binary label mode and
    # the loss function fom categorical_crossentropy to binary_crossentropy
    # can also check that the images are being read in RBG per input_shape 
    # 224 x 224 x 3 prerequisite using applications.resnet50.preprocess_input
    
    import matplotlib.pyplot as plotter_lib
    import numpy as np
    # import PIL as image_lib
    import tensorflow as tf
    from tensorflow.keras.layers import Flatten
    from keras.layers.core import Dense
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam
    # import cv2

    # base_dir = 'C:\\Users\\crichard\\Documents\\COGA\\' # LAPTOP    
    # base_dir = 'D:\\COGA_eec\\' #  BIOWIZARD
    
    pth = 'D:\\COGA_eec\\chan_hz_figures\\'
    
    img_height,img_width=224,224
    batch_size=32
    epochs=10

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
      pth,
      validation_split=0.2,
      subset="training",
      seed=123,
      label_mode='binary',
      image_size=(img_height, img_width),
      batch_size=batch_size)
    
    validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
        pth,
        validation_split=0.2,
        subset="validation",
        seed=123,
        label_mode='binary',
        image_size=(img_height, img_width),
        batch_size=batch_size)
    
    coga_model = Sequential()

    rn50= tf.keras.applications.ResNet50(include_top=False,
        input_shape=(img_height, img_width,3),
        pooling='avg',
        classes=2,
        weights=None)
    
    for each_layer in rn50.layers:
        each_layer.trainable=True
    coga_model.add(rn50)
        
    coga_model.add(Flatten())
    coga_model.add(Dense(512, activation='relu'))
    coga_model.add(Dense(1, activation='sigmoid'))
    
    coga_model.compile(optimizer=Adam(learning_rate=0.001),loss=tf.keras.losses.BinaryCrossentropy(),metrics=['accuracy'])
    history = coga_model.fit(train_ds, validation_data=validation_ds, epochs=epochs)
    
    plotter_lib.figure(figsize=(8, 8))
    epochs_range= range(epochs)
    plotter_lib.plot( epochs_range, history.history['accuracy'], label="Training Accuracy")
    plotter_lib.plot(epochs_range, history.history['val_accuracy'], label="Validation Accuracy")
    plotter_lib.axis(ymin=0.4,ymax=1)
    plotter_lib.grid()
    plotter_lib.title('Model Accuracy, binary crossentropy')
    plotter_lib.ylabel('Accuracy')
    plotter_lib.xlabel('Epochs')
    plotter_lib.legend(['train', 'validation'])
    
    
    
    
    
    

if do_bad_channel_check_table_gen:
    print('Starting do_bad_channel_check_table_gen\n')

# THIS READS AND PROCESSES THE FLAT CHANNEL AND EXCESSIVE NOISE CHANNEL METICS
# OUTPUTS OF THIS BLOCK ARE LOOKUP TABLES FOR DOWNSTREAM PROCESSING WITH
# SUBJECT ID AND VISIT COLUMNS, ALL BY CHANNEL METRIC, SAVES THEM AS PICKLE FILES
    
    # ~~~~~~~~~~~~ FLAT CHANNEL METRICS ~~~~~~~~~~~~~~~~
    # THE NUMBER OF 1 SECOND INTERVALS (RANGE [0,256]) IN WHICH THE DIFFERENCE BETWEEN THE 
    # MAXIMUM VALUE AND THE MININUM VALUE WAS LESS THAN 5 MICROVOLTS
    
    # ~~~~~~~~~~~~ EXCESSIVE NOISE CHANNEL METRICS ~~~~~~~~~~~~~~~~
    # THE NUMBER OF 1 SECOND INTERVALS IN WHICH THE DIFFERENCE BETWEEN THE MAXIMUM 
    # VALUE AND THE MININUM VALUE WAS GREATER THAN 100 MICROVOLTS
    
    # read_dir = 'C:\\Users\\crichard\\Documents\\' # LAPTOP    
    # base_dir = 'C:\\Users\\crichard\\Documents\\COGA\\' # LAPTOP    
    base_dir = 'D:\\COGA_eec\\' #  BIOWIZARD
    
    demog_hdr = ['ID','this_visit','fname','chan_num']
    
    visit_pos = 2 # FOR USE IN DOWNSTREAM SPLIT OPERATION
    visit_letters = list(string.ascii_lowercase)

    d = pd.read_csv(base_dir + 'eeg_eval_12.15.23.dat', header=None)
    
    clist = pd.read_csv(base_dir + 'chan_64.txt', delimiter='\t', header=None)
    hdr = clist.iloc[:,1].tolist()
    hdr = [h.strip() for h in hdr]

    hdr32 = hdr[0:31]
    hdr32.append('eye')
    hdr32 = demog_hdr + hdr32

    hdr64 = hdr.copy() 
    hdr64.extend(['eye','unknown1','unknown2'])
    hdr64 = demog_hdr + hdr64
    
    hdr = demog_hdr + hdr


    qflat = pd.DataFrame(columns=hdr, index=np.arange(len(d)/3))
    qexc = pd.DataFrame(columns=hdr, index=np.arange(len(d)/3))
    qcount = 0

    for i in range(0, len(d), 3):
        # GET THREE ROW SUBSET AND EXTRACT FILE NAME
        dd = d.iloc[i:i+3,:]
        fn = dd.iloc[0,:][0]
        # info = fn.split('_')
        # info2 = info[-1].split('.')
        # info.pop(-1)
        # info = info + info2
        thisVisitCode = csd.get_sub_from_fname(fn, visit_pos)[0].lower()
        try:
            thisVisit = [visit_letters.index(thisVisitCode)][0] + 1
        except Exception as e:
            with open(base_dir + 'errors_from_eeg_quality_check.txt', 'a') as ff:
                note = 'FILENAME ' + fn + ' visit ' + thisVisitCode + '_' + str(e) + '\n'
                ff.write(note)
                print(note)
            continue
            
        thisID = csd.get_sub_from_fname(fn, 3).split('.')[0]
        
        # THE NUMBER OF 1 SECOND INTERVALS (RANGE [0,256]) IN WHICH THE DIFFERENCE BETWEEN THE 
        # MAXIMUM VALUE AND THE MININUM VALUE WAS LESS THAN 5 MICROVOLTS 
        raw1 = dd.iloc[1,:][0].split(' ')[1:]
        thisrow =  [int(v) for v in raw1] 

        if len(thisrow)==64:
            # In the 32 channel set, chan 32 is the eye channel; in the 64 channel set the eye channel is 62.
            thisrow = [(thisID), thisVisit, fn, 64] + thisrow
            vals1 = pd.DataFrame([thisrow],columns=hdr64)
            vals1.drop(['eye','unknown1','unknown2'],axis=1, inplace=True)
        elif len(thisrow)==32:
            thisrow.pop(31)
            thisrow = [(thisID), thisVisit, fn, 32] + thisrow
            thisrow =  thisrow + [np.nan]*(len(qflat.columns)-len(thisrow))
            vals1 = pd.DataFrame([thisrow],columns=hdr)
        else:
            print('PROBLEM WITH ' + fn + ', ' + str(len(thisrow)) + ' channels?')
            
        # COPY VALUES INTO DATAFRAMES THEN INCREMENT
        qflat.iloc[qcount,:] = vals1.iloc[0,:]    
        
        # THE NUMBER OF INTERVALS IN WHICH THE DIFFERENCE BETWEEN THE MAXIMUM 
        # VALUE AND THE MININUM VALUE WAS GREATER THAN 100 MICROVOLTS
        raw2 = dd.iloc[2,:][0].split(' ')[1:]
        thisrow =  [int(v) for v in raw2] 

        if len(thisrow)==64:
            # In the 32 channel set, chan 32 is the eye channel; in the 64 channel set the eye channel is 62. -Dave Chorlian
            thisrow = [(thisID), thisVisit, fn, 64] + thisrow
            vals2 = pd.DataFrame([thisrow],columns=hdr64)
            vals2.drop(['eye','unknown1','unknown2'],axis=1, inplace=True)
        elif len(thisrow)==32:
            thisrow.pop(31)
            thisrow = [(thisID), thisVisit, fn, 32] + thisrow
            thisrow =  thisrow + [np.nan]*(len(qexc.columns)-len(thisrow))
            vals2 = pd.DataFrame([thisrow],columns=hdr)
        else:
            print('PROBLEM WITH ' + fn)
        
        # COPY VALUES INTO DATAFRAMES THEN INCREMENT
        qexc.iloc[qcount,:] =  vals2.iloc[0,:]
        qcount+=1
    
    qflat.to_pickle(base_dir  + 'coga_eec_channel_quality_FLAT.pkl')
    qexc.to_pickle(base_dir  + 'coga_eec_channel_quality_EXCESSIVE.pkl')
    # dat =  pd.read_pickle(base_dir  + 'chan_hz_dat.pkl')
    
    
# THIS BLOCK GENERATES HISTOGRAM OF PERCENT BAD CHANNELS FOR THRESHOLD SELECTION
# USING FLAT AND NOISY CHANNEL METRICS FROM PICKLE FILES GENERATED ABOVE
if do_bad_channel_figure_gen: 
    # 1. CALCULATE % RETAINED AT ALL POSSIBLE VALUES (0-256) 
    # 3. FIND MATCHING SUBJECT, VISIT, CHANNEL IN PACDAT AND UPDATE
    # 4. SAVE PACDAT

    plot_figs = True
    figtype = 'cumulative' # cumulative histogram
    base_dir = 'D:\\COGA_eec\\' #  BIOWIZARD
    # base_dir = 'C:\\Users\\crichard\\Documents\\COGA\\' # LAPTOP
    
    # pacdat.to_csv(base_dir + 'pacdat' + '.csv', index=False)
    # pacdat = pd.read_csv(base_dir + 'pacdat.csv')
    flat =  pd.read_pickle(base_dir  + 'coga_eec_channel_quality_FLAT.pkl')
    exss =  pd.read_pickle(base_dir  + 'coga_eec_channel_quality_EXCESSIVE.pkl')
    
    flat_metrics = [0]*256 # FLAT CHANNEL METRICS
    exss_metrics = [0]*256 # EXCESSIVE NOISE CHANNEL METRICS
    
    chans = np.array(exss.columns)
    chans = chans.tolist()
    chans = [c.strip() for c in chans][4:]
    
    figlbl = figtype[0].upper()+figtype[1:]    
    if figtype=='histogram':
        figlbl2 = 'Counts'
        legloc = "upper center"
    else:
        figlbl2 = 'Cumulative'
        legloc = "lower center"

    if figtype=='histogram':
        for ch in range(0,len(chans)):
            for i in range(0,256):
                f_bools = flat[[chans[ch]]]==i
                f_nums = f_bools.replace({True: 1, False: 0})
                fchan = np.array(f_nums)
                fcf = [f[0] for f in fchan]
                flat_metrics[i] = sum(fcf)
                x_bools = exss[[chans[ch]]]==i
                x_nums = x_bools.replace({True: 1, False: 0})
                xchan = np.array(x_nums)
                xcf = [x[0] for x in xchan]
                exss_metrics[i] = sum(xcf)
            if plot_figs:
                figFN = figtype + '_flat_noisy_metrics_' + chans[ch] + '.jpg'
                plt.figure()
                plt.yscale("log")   
                plt.plot(flat_metrics, label='flat intervals (delta < 5uV)')
                plt.plot(exss_metrics, label='noisy intervals (delta > 100uV)')
                plt.title('EEG recording quality metrics ' + figlbl + ', channel ' + chans[ch])
                plt.xlabel('# of bad 1 second intervals')
                plt.ylabel('# of EEG recordings (' + figlbl2 + ')')
                plt.legend(loc=legloc)
                # plt.show()
                # plt.clf()
                print('Saving file ' + figFN)
                plt.savefig(base_dir + 'eeg_quality_figures\\' + figFN)
                plt.close()
    if figtype=='cumulative':
        for ch in range(0,len(chans)):
            for i in range(0,256):
                f_bools = flat[[chans[ch]]]<=i
                f_nums = f_bools.replace({True: 1, False: 0})
                fchan = np.array(f_nums)
                fcf = [f[0] for f in fchan]
                flat_metrics[i] = sum(fcf)
    
                x_bools = exss[[chans[ch]]]<=i
                x_nums = x_bools.replace({True: 1, False: 0})
                xchan = np.array(x_nums)
                xcf = [x[0] for x in xchan]
                exss_metrics[i] = sum(xcf)
    
            if plot_figs:
                figFN = figtype + '_flat_noisy_metrics_' + chans[ch] + '.jpg'
                plt.figure()
                # plt.yscale("log")   
                plt.plot(flat_metrics, label='flat intervals (delta < 5uV)')
                plt.plot(exss_metrics, label='noisy intervals (delta > 100uV)')
                plt.title('EEG recording quality metrics ' + figlbl + ', channel ' + chans[ch])
                plt.xlabel('# of bad 1 second intervals')
                plt.ylabel('# of EEG recordings (' + figlbl2 + ')')
                plt.legend(loc=legloc)
                # plt.show()
                # plt.clf()
                print('Saving file ' + figFN)
                plt.savefig(base_dir + 'eeg_quality_figures\\' + figFN)
                plt.close()
     
if do_bad_channel_pacdat_update: 
#  THIS UPDATES THE MASTER DATA TABLE pacdat
    print('Starting do_bad_channel_pacdat_update\n')
    flat_cutoff = 25 
    noise_cutoff = 25 
    startrow = 0 # SET TO 0 UNLESS PICKING UP WHERE LEFT OFF IN pacdat UPDATING
    base_dir = 'D:\\COGA_eec\\' #  BIOWIZARD
    # base_dir = 'C:\\Users\\crichard\\Documents\\COGA\\' # LAPTOP
    
    pacdat = pd.read_csv(base_dir + 'pacdat.csv')
    # pacdat = pd.read_pickle(base_dir + 'pacdat.pkl')
    pacdat[['bad_flat','bad_noisy','flat_score','noise_score']] = np.nan
    bf = pacdat.pop('bad_flat')
    bf = pd.DataFrame(bf,columns=['bad_flat'])
    bn = pacdat.pop('bad_noisy')
    bn = pd.DataFrame(bn,columns=['bad_noisy'])
    fs = pacdat.pop('flat_score')
    fs = pd.DataFrame(fs,columns=['flat_score'])
    ns = pacdat.pop('noise_score')
    ns = pd.DataFrame(ns,columns=['noise_score'])
    
    flat =  pd.read_pickle(base_dir  + 'coga_eec_channel_quality_FLAT.pkl')
    exss =  pd.read_pickle(base_dir  + 'coga_eec_channel_quality_EXCESSIVE.pkl')
    missing_count = 0 
    
    # WE DON'T CARE ABOUT NON-EEG CHANNELS
    exclude_chs = ['X','Y','BLANK','Horizonta','Vertical','HEOG','VEOG']
    for ec in exclude_chs:
        pdi = pacdat[(pacdat.channel==ec)].index
        pacdat.drop(index=pdi, inplace=True)
    # BUT NOW WE NEED TO RESET THE INDEX BECAUSE OF THE ROWS WE EXCLUDED ABOVE
    pacdat.reset_index(drop=True,inplace=True)
    # WE CAN NOW START GOING THROUGH EACH ROW OF pacdat 
    # TO GET BAD CHANNEL VALUES FROM LOOKUP TABLES flat AND exss
    for i in range(startrow,len(pacdat)):
        # GET LOOKUP VALUES FOR THIS ROW IN pacdat
        this_id_pacdat = str(pacdat.ID[i])
        thisvisit_pacdat = pacdat.this_visit[i]
        this_ch_pacdat = pacdat.channel[i]
        # USE LOOKUP VALUES IN BAD CHANNEL TABLE FOR FLAT CHANNEL METRICS
        this_subj_vis = flat[(flat.ID==this_id_pacdat) & (flat.this_visit==thisvisit_pacdat)]
        # WE ONLY NEED TO DO THIS CHECK FOR EMPTY DATAFRAME ONCE BECAUSE IF 
        # IT'S EMPTY FOR flat THEN IT WILL BE EMPTY FOR exss
        if this_subj_vis.empty:
            missing_count+=1
            print('Subject ' + this_id_pacdat + ' visit ' + str(thisvisit_pacdat) + '' +' not found (i =  ' + str(i) + ')')
            continue
        flat_score = this_subj_vis[[this_ch_pacdat]].iloc[0,0]
        # STORE THE FLAT CHANNEL 'SCORE' IN fs, A COLUMN TO BE ADDED BACK TO pacdat ONCE FILLED
        fs.loc[i,('flat_score')] = flat_score
        if flat_score>flat_cutoff:
            bf.loc[i,('bad_flat')] = 1
            # pacdat.loc[i,('bad_flat')] = 1
        else:
            bf.loc[i,('bad_flat')] = 0
        # LOOKUP FOR NOISY CHANNEL METRICS
        this_subj_vis = exss[(exss.ID==this_id_pacdat) & (exss.this_visit==thisvisit_pacdat)]
        noise_score = this_subj_vis[[this_ch_pacdat]].iloc[0,0]
        ns.loc[i,('noise_score')] = noise_score
        if noise_score>noise_cutoff:
            bn.loc[i,('bad_noisy')] = 1
            # pacdat.loc[i,('bad_noisy')] = 1
        else:
            bn.loc[i,('bad_noisy')] = 0
    
    pacdat.insert(2,'bad_flat',bf)
    pacdat.insert(3,'flat_score',fs)
    pacdat.insert(4,'bad_noisy',bn)
    pacdat.insert(5,'noise_score',ns)
    # pacdat.to_pickle(base_dir + 'pacdat2.pkl')
    mcp = (missing_count/len(pacdat))*100
    print(str(mcp) + '% of files in pacdat are missing from flat')
    flatfn = 'pacdat_cutoffs_flat_' + str(flat_cutoff ) + '_excessnoise_' + str(noise_cutoff) + ''    
    # pacdat.to_csv(base_dir + flatfn + '.csv', index=False)
    pacdat.to_pickle(base_dir + flatfn + '.pkl')
    
    
    
    
    
    
if do_bad_channel_removal:
#  THIS MOVES IMAGE FILES DERIVED FROM 'BAD' CHANNELS AS MARKED IN pacdat INTO 
# SEPARATE FOLDERS SO THAT THEY ARE NOT USED IN MACHINE LEARNING

    whichEEGfileExtention = 'jpg'
    read_dir = 'D:\\COGA_eec\\pac_figures\\'  #  BIOWIZARD
    base_dir = 'D:\\COGA_eec\\' #  BIOWIZARD
    # whichEEGfileExtention = 'png'
    # read_dir = 'D:\\COGA_eec\\eeg_figures\\'  #  BIOWIZARD
    # base_dir = 'C:\\Users\\crichard\\Documents\\COGA\\' # LAPTOP
    
    badf_target_path = 'D:\\COGA_eec\\pac_figures\\flat_channels\\'
    badn_target_path = 'D:\\COGA_eec\\pac_figures\\noisy_channels\\'
    # badf_target_path = 'D:\\COGA_eec\\eeg_figures_flat\\'
    # badn_target_path = 'D:\\COGA_eec\\eeg_figures_noise\\'
    # pacdat_target_path = 'D:\\COGA_eec\\eeg_figures_pacdat\\'
    
    chan_i = 0 
    visit_i = 3 
    id_i = 4 
    
    flat_cutoff = 200 # OUT OF 256
    noise_cutoff = 250 # OUT OF 256

    fl_alc = csd.get_file_list(read_dir + 'alcoholic\\', whichEEGfileExtention)
    fl_nonalc = csd.get_file_list(read_dir + 'nonalcoholic\\', whichEEGfileExtention)    
    figList = fl_alc + fl_nonalc
    fig_info = pd.DataFrame(figList, columns=['dir','fn'])
    
    c = [f.split('_')[chan_i] for f in  fig_info.fn]
    c = pd.DataFrame(c,columns=['channels'])
    fig_info.insert(0,'channels',c)

    visitCodeList = [f.split('_')[visit_i][0] for f in  fig_info.fn]
    visitCodeList = [csd.convert_visit_code(v) for v in visitCodeList]    
    v = pd.DataFrame(visitCodeList,columns=['this_visit'])
    fig_info.insert(0,'this_visit',v)
    
    v = [f.split('_')[id_i] for f in  fig_info.fn]
    v = pd.DataFrame(v,columns=['ID'])
    fig_info.insert(0,'ID',v)  
        
    # # THIS BLOCK USED TO RESIZE IMAGES FOR RESNET-50
    # # IN FUTURE VERSION PERHAPS AS A HELPER FUNCTION
    # img2 = Image.open(write_dir + 'chan_hz_figures\\' + figFN)
    # print(figFN)
    # img2 = img2.resize((224, 224))
    # img2.save(write_dir + 'chan_hz_figures\\' + figFN)
    
    # THIS IS HERE TO MOVE ALL FILES IN THE LIST TO SEE WHAT FILES LEFT OVER IN 
    # LAPTOP FOLDER BUT CAN BE USED FOR OTHER PURPOSES
    pacdat = pd.read_pickle(base_dir + 'pacdat_cutoffs_flat_25_excessnoise_25.pkl')
    # for i in range(0,len(pacdat)):
    #     thisid = str(pacdat.iloc[i,0])
    #     thisvisit = pacdat.iloc[i,10]
    #     this_ch = pacdat.iloc[i,8]
    #     this_subj_vis = fig_info[(fig_info.ID==thisid) & (fig_info.this_visit==thisvisit) & (fig_info.channels==this_ch)]
    #     if not this_subj_vis.empty:
    #         pth = this_subj_vis.dir.values[0]
    #         fn = this_subj_vis.fn.values[0]
    #         source_path_fn = pth + fn
    #         os.rename(source_path_fn, pacdat_target_path + fn)
    
    # INDEX POSITIONS IN pacdat COLUMNS
    id_i = 0
    chan_i = 10 
    vis_i = 12 
    alc_diag_i = 33

    missing_flat = 0
    missing_noisy = 0

    # bad = pacdat[pacdat.bad_flat==1]    
    bad = pacdat[pacdat.flat_score>flat_cutoff]    
    print('There are ' + str(len(bad)) + ' bad channel EEG files (more than ' + str(flat_cutoff) + ' flat 1 second intervals) ')
    for i in range(0,len(bad)):
        thisid = str(bad.iloc[i,id_i])
        thisvisit = bad.iloc[i,vis_i]
        this_ch = bad.iloc[i,chan_i]
        this_subj_vis = fig_info[(fig_info.ID==thisid) & (fig_info.this_visit==thisvisit) & (fig_info.channels==this_ch)]
        if not this_subj_vis.empty:
            pth = this_subj_vis.dir.values[0]
            fn = this_subj_vis.fn.values[0]
            bad_source_path_fn = pth + fn
            this_alc_diag = bad.iloc[i,alc_diag_i]
            if this_alc_diag:
                diag_folder = 'alcoholic\\'
            else:
                diag_folder = 'nonalcoholic\\'
            
            
            try:
                os.rename(bad_source_path_fn, badf_target_path + diag_folder + fn)
            except Exception as e:
                missing_flat+=1
                print('Some problem with moving ' + fn + '\t error msg: ' + str(e) + '\n')
                with open(base_dir + 'errors_from_bad_channel_exclusion.txt', 'a') as bf:
                    bf.write(fn + '\t' + str(e) + ' flat channel,' + diag_folder[:-1]  + ' \n')
                continue    
            print('Moved ' + fn + ' (' + str(i+1) + ' of ' + str(len(bad)) + ' flat channels) ' + ' - ' + diag_folder[:-1]  )
        else:
            missing_flat+=1
    total_bad_flat = str(len(bad))
    print('/nChannels in pacdat without flat channel information: ' + str(missing_flat) + ' out of ' + total_bad_flat + '/n')


    bad = pacdat[pacdat.noise_score>noise_cutoff]    
    # bad = pacdat[pacdat.bad_noisy==1]
    print('There are ' + str(len(bad)) + ' bad channel EEG files (more than ' + str(noise_cutoff) + ' noisy 1 second intervals) /n')
    for i in range(0,len(bad)):
        thisid = str(bad.iloc[i,id_i])
        thisvisit = bad.iloc[i,vis_i]
        this_ch = bad.iloc[i,chan_i]
        this_subj_vis = fig_info[(fig_info.ID==thisid) & (fig_info.this_visit==thisvisit) & (fig_info.channels==this_ch)]
        if not this_subj_vis.empty:
            pth = this_subj_vis.dir.values[0]
            fn = this_subj_vis.fn.values[0]
            bad_source_path_fn = pth + fn
            this_alc_diag = bad.iloc[i,alc_diag_i]
            if this_alc_diag:
                diag_folder = 'alcoholic\\'
            else:
                diag_folder = 'nonalcoholic\\'
            
            try:
                os.rename(bad_source_path_fn, badn_target_path + diag_folder + fn)
            except Exception as e:
                missing_flat+=1
                print('Some problem with moving ' + fn + '\t error msg: ' + str(e) + '\n')
                with open(base_dir + 'errors_from_bad_channel_exclusion.txt', 'a') as bf:
                    bf.write(fn + '\t' + str(e) + ' noisy channel, ' + diag_folder[:-1]  + ' \n')
                continue                
            print('Moved ' + fn + ' (' + str(i+1) + ' of ' + str(len(bad)) + ' noisy channels) ' + ' - ' + diag_folder[:-1]  )
        else:
            missing_noisy+=1
    total_bad_noisy = str(len(bad))
    print('Channels in pacdat without noisy channel information: ' + str(missing_noisy) + ' out of ' + total_bad_noisy + '/n')
    
    
if do_bad_channel_check:
#  HIGHJACKED TO DO IMAGE SIZE CONVERSION FOR RESNET-50

    from PIL import Image


    whichEEGfileExtention = 'jpg'
    read_dir = 'C:\\Users\\lifep\\Documents\\COGA_eec\\pac_figures_segmented\\'
    # read_dir = 'D:\\COGA_eec\\pac_figures\\'  #  BIOWIZARD
    # base_dir = 'D:\\COGA_eec\\' #  BIOWIZARD
    # whichEEGfileExtention = 'png'
    # read_dir = 'D:\\COGA_eec\\eeg_figures\\'  #  BIOWIZARD
    # base_dir = 'C:\\Users\\crichard\\Documents\\COGA\\' # LAPTOP
    
    # badf_target_path = 'D:\\COGA_eec\\pac_figures\\flat_channels\\'
    # badn_target_path = 'D:\\COGA_eec\\pac_figures\\noisy_channels\\'
    # badf_target_path = 'D:\\COGA_eec\\eeg_figures_flat\\'
    # badn_target_path = 'D:\\COGA_eec\\eeg_figures_noise\\'
    # pacdat_target_path = 'D:\\COGA_eec\\eeg_figures_pacdat\\'
    
    chan_i = 0 
    visit_i = 3 
    id_i = 4 
    
    flat_cutoff = 200 # OUT OF 256
    noise_cutoff = 250 # OUT OF 256

    fl_alc = csd.get_file_list(read_dir + 'alcoholic\\', whichEEGfileExtention)
    fl_nonalc = csd.get_file_list(read_dir + 'nonalcoholic\\', whichEEGfileExtention)    
    figList = fl_alc + fl_nonalc
    
    fig_info = pd.DataFrame(figList, columns=['dir','fn'])
    
    c = [f.split('_')[chan_i] for f in  fig_info.fn]
    c = pd.DataFrame(c,columns=['channels'])
    fig_info.insert(0,'channels',c)

    visitCodeList = [f.split('_')[visit_i][0] for f in  fig_info.fn]
    visitCodeList = [csd.convert_visit_code(v) for v in visitCodeList]    
    v = pd.DataFrame(visitCodeList,columns=['this_visit'])
    fig_info.insert(0,'this_visit',v)
    
    v = [f.split('_')[id_i] for f in  fig_info.fn]
    v = pd.DataFrame(v,columns=['ID'])
    fig_info.insert(0,'ID',v)  
        
    # THIS BLOCK USED TO RESIZE IMAGES FOR RESNET-50
    # IN FUTURE VERSION PERHAPS AS A HELPER FUNCTION
    for i in range(0,len(fig_info)):
        thisfig_dir = fig_info.loc[i,'dir']
        thisfig_fn = fig_info.loc[i,'fn']
        
        img2 = Image.open(thisfig_dir + thisfig_fn)
        print('Resizing ' + thisfig_fn + ' (' + str(i+1) + ' of ' + str(len(fig_info)) + ')' )
        img2 = img2.resize((224, 224))
        img2.save(thisfig_dir + thisfig_fn)
        img2.close()
    


if do_filter_figures_by_subject:
# MOVE PAC IMAGE FILES FROM SAME SUBJECT

    import shutil

    base_dir = 'D:\\COGA_eec\\' #  BIOWIZARD
    source_folder, targ_folder = 'pac_figures_all','pac_by_subj'
    whichEEGfileExtention = 'jpg'
    which_pacdat = 'pacdat_cutoffs_flat_25_excessnoise_25.pkl'

    # CONSTANTS
    chan_i = 0 
    visit_i = 3 
    id_i = 4 

    # GET MASTER TABLE OUT 
    pacdat = pd.read_pickle(base_dir + which_pacdat)

    # GET FILE LIST WITH GIVEN EXTENSION FROM SOURCE FOLDER 
    figList = csd.get_file_list(base_dir + source_folder, whichEEGfileExtention)
    # MAKE A DATAFRAME TO ENRICH WITH ADDITIONAL COLUMNS DERIVED FROM FILENAME INFO
    fig_info = pd.DataFrame(figList, columns=['dir','fn'])
    
    c = [f.split('_')[chan_i] for f in  fig_info.fn]
    c = pd.DataFrame(c,columns=['channels'])
    fig_info.insert(0,'channels',c)

    visitCodeList = [f.split('_')[visit_i][0] for f in  fig_info.fn]
    visitCodeList = [csd.convert_visit_code(v) for v in visitCodeList]    
    v = pd.DataFrame(visitCodeList,columns=['this_visit'])
    fig_info.insert(0,'this_visit',v)
    
    v = [f.split('_')[id_i] for f in  fig_info.fn]
    v = pd.DataFrame(v,columns=['ID'])
    fig_info.insert(0,'ID',v)  
    
    # CYCLE THROUGH EVERY SUBJECT REPRESENTED IN FILES FROM SOURCE FOLDER
    all_subj_figs = pd.unique(fig_info.ID) 
    for i in range(0,len(all_subj_figs)):
        this_subj = all_subj_figs[i]
        print(this_subj)
        # FIGURES FOR ALL VISITS BY THIS SUBJECT
        svisits = fig_info[fig_info.ID==this_subj]
        svisits = svisits.sort_values(by=['this_visit'])
        # WE WANT TO INCLUDE AUD DIAGNOSES IN FOLDER NAME FOR QUICK REF
        vinfo = pacdat[(pacdat.ID==int(this_subj)) & (pacdat.channel=='FZ')]
        vinfo = vinfo.sort_values(by=['this_visit'])
        
        alc_diag = ['_'+str(int(i)) for i in vinfo.AUD_this_visit]
        folder_tag = '_' + str(len(vinfo)) + 'visits_AUD' + ''.join(alc_diag)
        subj_path = base_dir + targ_folder + '\\' + this_subj + folder_tag + '\\'
        
        if not os.path.exists(subj_path):
            os.makedirs(subj_path) 
        vinfo.to_csv(subj_path + 'details.csv')

        for v in svisits.index:
            src = svisits.loc[v,'dir'] + '\\' + svisits.loc[v,'fn']
            shutil.copy(src, subj_path + svisits.loc[v,'fn'])




if do_filter_figures_by_condition:
# MOVE PAC IMAGE FILES THAT MEET CONDITIONS
    import shutil

    source_folder = 'pac_figures_all' # pac_figures_all chan_hz
    data_str = 'PAC' # PAC chanxHz
    whichEEGfileExtention = 'jpg'
    min_age = 35 
    max_age = 45    
    sex = 'M' # F M or both 
    do_copy = True

    diag_dirs_exist = False # HAVE THE IMAGES ALREADY BEEN SORTED INTO ALCOHOLIC AND NONALCHOLIC?
    chan_in_fn = True # IS THERE CHANNEL INFORMATION IN THE FILENAME?
    
    
    which_pacdat = 'pacdat_cutoffs_flat_25_excessnoise_25.pkl'
    read_dir = 'D:\\COGA_eec\\' + source_folder + '\\'  #  BIOWIZARD
    base_dir = 'D:\\COGA_eec\\' #  BIOWIZARD
    
    title_str = 'Age ' + str(min_age) + '-' + str(max_age)  + ' ' + sex    
    targ_folder = data_str + '_' + str(min_age) + '_' + str(max_age) + '_' + sex

    # CONSTANTS 
    if chan_in_fn:
    # FOR PAC IMAGES
        chan_i = 0 
        visit_i = 3 
        id_i = 4 
    else:
    # FOR CHANNEL X FREQUENCY BAND SPECTAL POWER IMAGES
        # EXAMPLE: 'eec_1_a1_10003051_cnt_256.jpg'
        visit_i = 2 
        id_i = 3

    if diag_dirs_exist:
        fl_alc = csd.get_file_list(read_dir + 'alcoholic\\', whichEEGfileExtention)
        fl_nonalc = csd.get_file_list(read_dir + 'nonalcoholic\\', whichEEGfileExtention)    
        figList = fl_alc + fl_nonalc
    else:
        # GET FILE LIST WITH GIVEN EXTENSION FROM SOURCE FOLDER 
        figList = csd.get_file_list(read_dir, whichEEGfileExtention)
    
    # MAKE A DATAFRAME TO ENRICH WITH ADDITIONAL COLUMNS DERIVED FROM FILENAME INFO
    fig_info = pd.DataFrame(figList, columns=['dir','fn'])
    
    # TO PROCESS CHANNEL INFO IN FILE NAMES
    if chan_in_fn:
        c = [f.split('_')[chan_i] for f in  fig_info.fn]
        c = pd.DataFrame(c,columns=['channels'])
        fig_info.insert(0,'channels',c)

    visitCodeList = [f.split('_')[visit_i][0] for f in  fig_info.fn]
    visitCodeList = [csd.convert_visit_code(v) for v in visitCodeList]    
    v = pd.DataFrame(visitCodeList,columns=['this_visit'])
    fig_info.insert(0,'this_visit',v)
    
    v = [f.split('_')[id_i] for f in  fig_info.fn]
    v = pd.DataFrame(v,columns=['ID'])
    fig_info.insert(0,'ID',v)  
    
    pacdat = pd.read_pickle(base_dir + which_pacdat)
        
    # del pacdat

    # diags = [ int(i) for i in subset[['AUD_this_visit']].values[:,0] ]
    # alc = str(round((sum(diags)/len(diags))*100,1))
    
    if chan_in_fn:
        if not sex=='both':    
            subset = pacdat[(pacdat.age_this_visit>=min_age) & (pacdat.age_this_visit<=max_age) & (pacdat.sex==sex) & (pacdat.channel=='FZ')]
        else:
            subset = pacdat[(pacdat.age_this_visit>=min_age) & (pacdat.age_this_visit<=max_age) & (pacdat.channel=='FZ')]

        for i in range(0,len(fig_info)):
            this_id = int(fig_info.loc[i,'ID'])
            this_chan = fig_info.loc[i,'channels']
            this_visit = fig_info.loc[i,'this_visit']
            
            this_subj_vis = subset[(subset.ID==this_id) & (subset.channel==this_chan) & (subset.this_visit==this_visit)]
            if not this_subj_vis.empty:            
                this_dir = fig_info.loc[i,'dir']
                this_fn = fig_info.loc[i,'fn']
                
                this_alc_diag = this_subj_vis.AUD_this_visit.values[0]
                if this_subj_vis.AUD_this_visit.values[0]:
                    diag_folder = 'alcoholic'
                else:
                    diag_folder = 'nonalcoholic'
                    
                
                old_path_fn = this_dir + this_fn
                
                new_path_fn = this_dir
                new_path_fn = new_path_fn.replace(source_folder, targ_folder + '\\' + diag_folder)
                
                if not os.path.exists(new_path_fn):
                    os.makedirs(new_path_fn) 
                
                if do_copy:
                    shutil.copy(old_path_fn, new_path_fn + this_fn)
                    print('Copying file ' + this_fn)
        
        print('\n'.replace('\\n','\n'))
        print('There are ' + str(len(subset)) + ' with ' + str(min_age) + '-' + str(max_age) + ' age range of ' + sex)
        if 1:
            subset[['age_this_visit']].plot.hist(bins=30) 
            
            
            
    else:
        if not sex=='both':    
            subset = pacdat[(pacdat.age_this_visit>=min_age) & (pacdat.age_this_visit<=max_age) & (pacdat.sex==sex)]
        else:    
            subset = pacdat[(pacdat.age_this_visit>=min_age) & (pacdat.age_this_visit<=max_age)]

        for i in range(0,len(fig_info)):
            this_id = int(fig_info.loc[i,'ID'])
            this_visit = fig_info.loc[i,'this_visit']
            
            this_subj_vis = subset[(subset.ID==this_id) & (subset.this_visit==this_visit)]
            if not this_subj_vis.empty:            
                this_dir = fig_info.loc[i,'dir']
                this_fn = fig_info.loc[i,'fn']
                
                # old_path_fn = this_dir + this_fn
                # new_path_fn = this_dir
                # new_path_fn = new_path_fn.replace(source_folder, targ_folder)
                # print('Copying file ' + this_fn)
                # if not os.path.exists(new_path_fn):
                #     os.makedirs(new_path_fn) 
                # shutil.copy(old_path_fn, new_path_fn + this_fn)
                
                this_alc_diag = this_subj_vis.AUD_this_visit.values[0]
                if this_subj_vis.AUD_this_visit.values[0]:
                    diag_folder = 'alcoholic'
                else:
                    diag_folder = 'nonalcoholic'
                    
                
                old_path_fn = this_dir + this_fn
                
                new_path_fn = this_dir
                new_path_fn = new_path_fn.replace(source_folder, targ_folder + '\\' + diag_folder)
                
                if not os.path.exists(new_path_fn):
                    os.makedirs(new_path_fn) 
                if do_copy:
                    shutil.copy(old_path_fn, new_path_fn + this_fn)
                    print('Copying file ' + this_fn)

        print('\n'.replace('\\n','\n'))
        print('There are ' + str(len(subset)) + ' with ' + str(min_age) + '-' + str(max_age) + ' age range of ' + sex)
        if 1:
            subset[['age_this_visit']].plot.hist(bins=30)     
    
    
    
    # HOW MANY IN SAMPLE ARE FROM ALCOHOLICS?
    diags = [ int(i) for i in subset[['AUD_this_visit']].values[:,0] ]
    alc = str(round((sum(diags)/len(diags))*100,1))
    # alc_not = str(round(((len(diags) - sum(diags))/len(diags))*100,1))
    # print('\n'.replace('\\n','\n'))
    print(alc + '% of sample are AUD\n')
        
        
        
        
        
        
        
if do_resnet_image_conversion:
# THIS BLOCK USED TO RESIZE IMAGES FOR RESNET-50

    from PIL import Image

    whichEEGfileExtention = 'jpg'
    # read_dir = 'C:\\Users\\lifep\\OneDrive\\Desktop\\processed_new\\'
    read_dir = 'D:\\COGA_eec\\new_pac\\' 
    write_dir = 'D:\\COGA_eec\\new_pac\\' 

    # fl_alc = csd.get_file_list(base_dir + 'alcoholic\\', whichEEGfileExtention)
    # fl_nonalc = csd.get_file_list(base_dir + 'nonalcoholic\\', whichEEGfileExtention)    
    # figList = fl_alc + fl_nonalc
    figList = csd.get_file_list(read_dir, whichEEGfileExtention)
    
    fig_info = pd.DataFrame(figList, columns=['dir','fn'])
        
    # IN FUTURE VERSION PERHAPS AS A HELPER FUNCTION
    for i in range(0,len(fig_info)):
        thisfig_fn = fig_info.loc[i,'fn']
        
        img2 = Image.open(read_dir + thisfig_fn)
        print('Resizing ' + thisfig_fn + ' (' + str(i+1) + ' of ' + str(len(fig_info)) + ')' )
        img2 = img2.resize((224, 224))
        img2.save(write_dir + thisfig_fn)
        img2.close()
        
        
        
        
        
        
if do_filter_by_subject:
# MOVE PAC IMAGE FILES FROM SAME SUBJECT

    import shutil
    import random
    
    which_dx = 'AUD' # AUD ALAB ALD
    sex = '' # M F
    min_age = 0 
    max_age = 99
    race = ''
    flat_cut = 50 # MAXIMUM DURATION IN SECONDS OF FLAT INTERVAL IN EEG SIGNAL (<5uV)
    noise_cut = 100 # MAXIMUM DURATION IN SECONDS OF NOISE INTERVAL IN EEG SIGNAL (>100uV)
    
    # flat_cut = 256
    # noise_cut = 256
    channel = 'FZ'
    base_dir = 'D:\\COGA_eec\\' #  BIOWIZARD
    source_folder = 'new_pac' # eeg_figures new_pac
    targ_folder = 'resnet_by_subj_e_' + str(min_age) + '_' + str(max_age) + '_' + which_dx + '_%flat' + str(flat_cut) + '_%noise' + str(noise_cut) + '_' + sex
    whichEEGfileExtention = 'jpg' # png jpg
    which_pacdat = 'pacdat_MASTER.pkl'

    # CONSTANTS    
    chan_i = 0 
    visit_i = 3 
    id_i = 4  

    # GET FILE LIST WITH GIVEN EXTENSION FROM SOURCE FOLDER 
    FileList = csd.get_file_list(base_dir + source_folder, whichEEGfileExtention)
    # MAKE A DATAFRAME TO ENRICH WITH ADDITIONAL COLUMNS DERIVED FROM FILENAME INFO
    file_info = pd.DataFrame(FileList, columns=['dir','fn'])
    
    c = [f.split('_')[chan_i] for f in  file_info.fn]
    c = pd.DataFrame(c,columns=['channels'])
    file_info.insert(0,'channels',c)

    visitCodeList = [f.split('_')[visit_i][0] for f in  file_info.fn]
    visitCodeList = [csd.convert_visit_code(v) for v in visitCodeList]    
    v = pd.DataFrame(visitCodeList,columns=['this_visit'])
    file_info.insert(0,'this_visit',v)
    
    v = [f.split('_')[id_i] for f in  file_info.fn]
    v = pd.DataFrame(v,columns=['ID'])
    file_info.insert(0,'ID',v)  
    # EXCEPTIONS - REMOVING BECAUSE BELOW ID NOT CONVERTABLE INTO AN INTEGER
    file_info[file_info.ID=='p0000079'] = 0
    
    # GET MASTER TABLE OUT 
    pacdat = pd.read_pickle(base_dir + which_pacdat)
    if len(sex)==0: 
        # pd_filtered = pacdat[(pacdat.channel==channel) & (pacdat.age_this_visit>=min_age) & (pacdat.age_this_visit<=max_age) & (pacdat.flat_score<=flat_cut) & (pacdat.noise_score<=noise_cut)]
        # pd_filtered = pacdat[(pacdat.channel==channel) & (pacdat.age_this_visit>=min_age) & (pacdat.age_this_visit<=max_age) & ((pacdat.perc_flat_slip1<=flat_cut) & (pacdat.max_noise<=noise_cut))]
        pd_filtered = pacdat[(pacdat.channel==channel) & (pacdat.age_this_visit>=min_age) & (pacdat.age_this_visit<=max_age) & (pacdat.perc_flat_slip0<=flat_cut) & (pacdat.perc_noise_slip0<=noise_cut)]
        # pd_filtered = pacdat[(pacdat.channel==channel) & (pacdat.age_this_visit>=min_age) & (pacdat.age_this_visit<=max_age) & ((pacdat.max_flat<=flat_cut) | (pacdat.max_noise<=noise_cut)) & (pacdat.race==race)]
        sexlbl = 'both'

    else:             
        pd_filtered = pacdat[(pacdat.channel==channel) & (pacdat.age_this_visit>=min_age) & (pacdat.age_this_visit<=max_age) & (pacdat.sex==sex) & (pacdat.perc_flat_slip0<=flat_cut) & (pacdat.perc_noise_slip0<=noise_cut)]
        # pd_filtered = pacdat[(pacdat.channel==channel) & (pacdat.age_this_visit>=min_age) & (pacdat.age_this_visit<=max_age) & (pacdat.sex==sex) & ((pacdat.max_flat<=flat_cut) | (pacdat.max_noise<=noise_cut))]
        # pd_filtered = pacdat[(pacdat.channel==channel) & (pacdat.age_this_visit>=min_age) & (pacdat.age_this_visit<=max_age) & (pacdat.sex==sex) & ((pacdat.max_flat<=flat_cut) | (pacdat.max_noise<=noise_cut)) & (pacdat.race==race)]
        sexlbl = sex
        
    jpg_subj = set([int(i) for i in set(file_info.ID)])
    pd_subj =  set([int(i) for i in set(pd_filtered.ID)])
    overlap = jpg_subj.intersection(pd_subj)
    
    print(targ_folder)
    filtered_N = str(len(set(pd_filtered.ID)))
    qualifying_N = str(len(overlap))
    total_N = str(len(pd_filtered))

    print('There are N = ' + qualifying_N + ' subjects represented in PAC heatmap dataset')
    print('out of ' + filtered_N + ' total subjects in this dataset')
    print('total # files = ' + total_N)
    print('Excluding EEG signals with:')
    print(' - Ages ' + str(min_age) + '-' + str(max_age) + '')
    print(' - Sex: ' + sexlbl)
    print(' - flat intervals (<5uV) lasting longer than ' + str(flat_cut) + ' in duration (seconds)')
    print(' - noise intervals (>100uV) lasting longer than ' + str(noise_cut) + ' in duration (seconds)')

    # CYCLE THROUGH EVERY SUBJECT REPRESENTED IN FILES FROM SOURCE FOLDER
    all_subj_figs = pd.unique(file_info.ID) 
    # all_subj_figs = (np.array(list(overlap)))
    for i in range(0,len(all_subj_figs)):
        this_subj = all_subj_figs[i]
        # this_subj = '10071029'
        
        # FIND ALL VISITS FOR A SUBJECT THEN FILTER BY AGE
        svisits = file_info[(file_info.ID==this_subj)]
        if len(svisits)>0:
            # print(this_subj)

            # WE WANT TO INCLUDE AUD DIAGNOSES IN FOLDER NAME FOR QUICK REF
            vinfo = pd_filtered[(pd_filtered.ID==int(this_subj))]
            if len(vinfo)>0:
                
                sv = list(set(svisits.this_visit.values))
                vi = list(set(vinfo.this_visit.values))
                for vs in sv:
                    if vs not in vi:
                        svisits = svisits[svisits.this_visit!=vs]
                        
                    
                if len(svisits)>0:
                    rand_row = svisits.loc[random.choice(svisits.index)]
                    this_file =  rand_row.fn
                    if which_dx=='ALAB':
                        if vinfo[vinfo.this_visit==rand_row.this_visit].ALAB_this_visit.values[0]:
                            diag_folder = 'alcoholic'
                        else:
                            diag_folder = 'nonalcoholic'
                    elif which_dx=='AUD':
                        if vinfo[vinfo.this_visit==rand_row.this_visit].AUD_this_visit.values[0]:
                            diag_folder = 'alcoholic'
                        else:
                            diag_folder = 'nonalcoholic'
                    elif which_dx=='ALD':
                        if vinfo[vinfo.this_visit==rand_row.this_visit].ALD_this_visit.values[0]:
                            diag_folder = 'alcoholic'
                        else:
                            diag_folder = 'nonalcoholic'
                    subj_path = base_dir + targ_folder + '\\' + diag_folder + '\\' 
                    if not os.path.exists(subj_path):
                        os.makedirs(subj_path) 
                    src = base_dir + source_folder + '\\' + this_file
                    trg = subj_path + this_file
                    shutil.copy(src, trg)
                    
                    # for f in range(0,len(svisits)):
                    #     this_file =  svisits.iloc[f].fn
                    #     if which_dx=='ALAB':
                    #         if vinfo[vinfo.this_visit==svisits.iloc[f].this_visit].ALAB_this_visit.values[0]:
                    #             diag_folder = 'alcoholic'
                    #         else:
                    #             diag_folder = 'nonalcoholic'
                    #     elif which_dx=='AUD':
                    #         if vinfo[vinfo.this_visit==svisits.iloc[f].this_visit].AUD_this_visit.values[0]:
                    #             diag_folder = 'alcoholic'
                    #         else:
                    #             diag_folder = 'nonalcoholic'
                    #     elif which_dx=='ALD':
                    #         if vinfo[vinfo.this_visit==svisits.iloc[f].this_visit].ALD_this_visit.values[0]:
                    #             diag_folder = 'alcoholic'
                    #         else:
                    #             diag_folder = 'nonalcoholic'                            
                    #     subj_path = base_dir + targ_folder + '\\' + diag_folder + '\\' 
                    #     if not os.path.exists(subj_path):
                    #         os.makedirs(subj_path) 
                    #     src = base_dir + source_folder + '\\' + this_file
                    #     trg = subj_path + this_file
                    #     shutil.copy(src, trg)
            
    print('\n ~~~~~~ There are N = ' + str(len(all_subj_figs)) + ' subjects in this dataset \n')
            


if do_resnet_pac_regularization:
    
    import matplotlib.pyplot as plotter_lib
    import numpy as np
    from PIL import Image
    import tensorflow as tf
    from keras.layers.core import Dense
    from keras.models import Model

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.layers.experimental.preprocessing import Rescaling
    from tensorflow.keras import regularizers
    from sklearn.model_selection import KFold
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix

    from keras.applications.resnet import preprocess_input
    from tensorflow.keras.layers import BatchNormalization, Input, GlobalAveragePooling2D, Flatten, Dropout


    # which_dx = 'AUD' # AUD ALAB ALD
    # sex = '' # M F
    # min_age = 0
    # max_age = 99 
    # race = ''
    # flat_cut = 0 # MAXIMUM DURATION IN SECONDS OF FLAT INTERVAL IN EEG SIGNAL (<5uV)
    # noise_cut = 0 # MAXIMUM DURATION IN SECONDS OF NOISE INTERVAL IN EEG SIGNAL (>100uV)
        
        
    # DEEP LEARNING MODEL
    learning_rate = .0001
    pooling = 'avg'
    img_height,img_width=224,224
    batch_size=32
    epochs=50
    
    include_top = False
    save_resnet_model = False
    
    # REGULARIZATION
    # Define L2 regularization factor
    alpha = .001
    # USING KFold TO DO CROSS-VALIDATION
    # n_splits = 2 

    # PATHS AND DATA INFO
    base_dir = 'D:\\COGA_eec\\'
    # targ_folder = 'resnet_by_subj_e_0_99_AUD_flat0_noise0_' # 'resnet_by_subj_20_40_cAUD_flat20_noise5'
    whichEEGfileExtention = 'jpg'
    data_str = 'FZ' # PAC@FZ chanxHz
    
    
    # title_str = 'rn50 d-RSV L2 alpha=' + str(alpha) + ' ' + which_dx + ' ' + sex +  ' Age ' + str(min_age) + '-' + str(max_age) + ' f' + str(flat_cut) + 'n' + str(noise_cut) 
    title_str = 'rn50 d-RSV' + ' ' + which_dx + ' ' + sex +  ' Age ' + str(min_age) + '-' + str(max_age) + ' %f' + str(flat_cut) + 'n' + str(noise_cut) 
    pth = base_dir + targ_folder + '\\'
    fl = csd.get_file_list(pth, whichEEGfileExtention)    
    fl_alc = csd.get_file_list(pth + 'alcoholic\\', whichEEGfileExtention)
    alc = str(round((len(fl_alc)/len(fl))*100,1))
    N_str = str(len(fl))

    # LISTS TO HOLD ACCURACY AND LOSS FUNCTION VALUES FOR PLOTTING
    t_acc = []
    v_acc = []
    t_loss = []
    v_loss = []
    
    # INPUT DATA AND LABELS TO PASS THROUGH KFold FUNCTION
    images = []
    labels = []
    for dx in ['alcoholic', 'nonalcoholic']:
        file_list = csd.get_file_list(pth + dx + '\\', whichEEGfileExtention)
        for i in file_list:
            img = Image.open(i[0] + i[1])
            img_array = np.array(img)
            img_array = preprocess_input(img_array)
            images.append(img_array)
            labels.append(dx)
    labels = np.array(labels)            
    labels[labels=='alcoholic'] = 1
    labels[labels=='nonalcoholic'] = 0
    labels = labels.astype(int)
    
    images = np.array(images)      
    labels = np.array(labels)      
    
    # kf = KFold(n_splits=n_splits, shuffle=True)    
    # for train_index, val_index in kf.split(images):
    #     X_train, X_val = images[train_index], images[val_index]
    #     y_train, y_val = labels[train_index], labels[val_index]
    
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

  
    # rn = tf.keras.applications.ResNet152(
    rn = tf.keras.applications.ResNet50(
        include_top=include_top,
        input_shape=(img_height, img_width,3),
        pooling=pooling,
        classes=2,
        weights='imagenet') # imagenet or None

    rn.trainable = False

    # Add L2 regularization to each convolutional and dense layer
    for layer in rn.layers:
        if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
        # if isinstance(layer, tf.keras.layers.Dense):
            layer.kernel_regularizer = regularizers.l2(alpha)
            if layer.use_bias:
                layer.bias_regularizer = regularizers.l2(alpha)


    coga_model = Sequential()
    coga_model.add(rn)
    # coga_model.add(GlobalAveragePooling2D())
    coga_model.add(Flatten())
    coga_model.add(Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(alpha)))
    coga_model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(alpha)))
    # coga_model.add(Dense(1024, activation='relu'))
    # coga_model.add(Dense(1, activation='sigmoid'))


    coga_model.compile(optimizer=Adam(learning_rate=learning_rate),
                       loss=tf.keras.losses.BinaryCrossentropy(),
                       metrics=['accuracy'])
    
    
    history = coga_model.fit(X_train, 
                             y_train,
                             validation_data=(X_val, y_val), 
                             epochs=epochs,
                             batch_size=batch_size,
                             verbose=1)
    t_acc = history.history['accuracy']
    v_acc = history.history['val_accuracy']
    t_loss = history.history['loss']
    v_loss = history.history['val_loss']
        
    
    # SAVE THIS MODEL
    if save_resnet_model:
        coga_model.save(base_dir + 'MODEL_' + targ_folder + '.keras')
        
        
    title_str+= ' N=' + N_str + ' alc=' + alc + '% lr=' + str(learning_rate) + ' alpha=' + str(alpha)
    fn = title_str.replace('(','')
    fn = fn.replace(')','')
    fn = fn.replace('=','_')
    fn = fn.strip() + '.jpg'

    epochs_range= range(epochs)
    
    plotter_lib.figure(figsize=(8, 8))
    plotter_lib.plot(epochs_range, t_acc, label="Training Accuracy")
    plotter_lib.plot(epochs_range, v_acc, label="Validation Accuracy")
    plotter_lib.axis(ymin=0.4,ymax=1.09)
    plotter_lib.grid()
    plotter_lib.title(data_str + ' ' + title_str)
    plotter_lib.ylabel('Accuracy')
    plotter_lib.xlabel('Epochs')
    plotter_lib.legend(['train', 'validation'])    
    
    plotter_lib.figure(figsize=(8, 8))
    plotter_lib.plot(epochs_range, t_loss, label="Training Loss")
    plotter_lib.plot(epochs_range, v_loss, label="Validation Loss")
    plotter_lib.axis(ymin=0,ymax=max(v_loss))
    plotter_lib.grid()
    plotter_lib.title(data_str + ' ' + title_str)
    plotter_lib.ylabel('Loss')
    plotter_lib.xlabel('Epochs')
    plotter_lib.legend(['train', 'validation'])  

    yhat_probs = coga_model.predict(X_val, verbose=0)
    yhat_probs = (yhat_probs > 0.5)
    cm = confusion_matrix(y_val, yhat_probs)
    TP = cm[0][0]
    FN = cm[1][0]
    FP = cm[0][1]
    TN = cm[1][1]
    
    prec = TP/(TP+FP)
    sens = TP/(TP+FN)
    spec = TN/(TN+FP)
    F1 = (2*prec*sens)/(prec + sens)
    
    print('precision: ' + str(prec))
    print('sensitivity: ' + str(sens) )
    print('specificity: ' + str(spec) )
    print('F1: ' + str(F1))
    









if do_resnet_pac:
    # after unimpressive training using ImageNet,
    # tried setting weights to None,  
    # then tried making each_layer trainable 
    # can also change from categorical to binary label mode and
    # the loss function fom categorical_crossentropy to binary_crossentropy
    # can also check that the images are being read in RBG per input_shape 
    # 224 x 224 x 3 prerequisite using applications.resnet50.preprocess_input
    
    # TO DO
    # try other resnet models
    # add batch normalization to input layer to center/rescale data
    # consider other options to best fit 'expectations' of pretrained resnet model
    # 
    
    import matplotlib.pyplot as plotter_lib
    import numpy as np
    # import PIL as image_lib
    import tensorflow as tf
    from tensorflow.keras.layers import BatchNormalization, Input, GlobalAveragePooling2D, Flatten, Dropout
    from keras.layers.core import Dense
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam
    from keras.models import Model
    from tensorflow.keras.applications.resnet import preprocess_input
    from tensorflow.keras.layers.experimental.preprocessing import Rescaling
    # import cv2

    # which_dx = 'ALD' # AUD ALAB ALD
    # sex = ''
    # min_age = 0
    # max_age = 99 
    # flat_cut = 0 # MAXIMUM DURATION IN SECONDS OF FLAT INTERVAL IN EEG SIGNAL (<5uV)
    # noise_cut = 0 # MAXIMUM DURATION IN SECONDS OF NOISE INTERVAL IN EEG SIGNAL (>100uV)

    # base_dir = 'C:\\Users\\crichard\\Documents\\COGA\\' # LAPTOP    
    base_dir = 'D:\\COGA_eec\\' #  BIOWIZARD
    learning_rate = 0.0001
    pooling = 'avg'
    img_height,img_width=224,224
    batch_size=32
    epochs=100

    include_top = False
    resnet_layers_trainable = False
    coga_layers_trainable = True
    save_resnet_model = False
    
    base_dir = 'D:\\COGA_eec\\'
    targ_folder = 'resnet_alldat_0_99_d_AUD_flat0_noise0' # 'resnet_by_subj_20_40_cAUD_flat20_noise5'
    whichEEGfileExtention = 'jpg'
    
    data_str = 'FZ' # PAC@FZ chanxHz
    title_str = 'rn50 d-RSV ALL ' + which_dx + ' Age ' + str(min_age) + '-' + str(max_age) + ' f' + str(flat_cut) + 'n' + str(noise_cut) 


    
    pth = base_dir + targ_folder + '\\'
    fl = csd.get_file_list(pth, whichEEGfileExtention)    
    fl_alc = csd.get_file_list(pth + 'alcoholic\\', whichEEGfileExtention)
    alc = str(round((len(fl_alc)/len(fl))*100,1))
    N_str = str(len(fl))
    
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
      pth,
      validation_split=0.2,
      subset="training",
      seed=999,
      label_mode='binary',
      image_size=(img_height, img_width),
      batch_size=batch_size)
    # train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y))
    # train_ds = train_ds.map(lambda x, y: (Rescaling(scale=1.0 / 255.0)(x), y))

    validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
        pth,
        validation_split=0.2,
        subset="validation",
        seed=999,
        label_mode='binary',
        image_size=(img_height, img_width),
        batch_size=batch_size)
    # validation_ds = validation_ds.map(lambda x, y: (preprocess_input(x), y))
    # validation_ds = validation_ds.map(lambda x, y: (Rescaling(scale=1.0 / 255.0)(x), y))
    
    # rn = tf.keras.applications.ResNet152(
    rn = tf.keras.applications.ResNet50(
        include_top=include_top,
        input_shape=(img_height, img_width,3),
        pooling=pooling,
        classes=2,
        weights='imagenet') # imagenet or None

    rn.trainable = False
    # for each_layer in rn.layers[-4:]:
    for each_layer in rn.layers:
        each_layer.trainable = resnet_layers_trainable
                
    # # PREPROCESSING BLOCK TO PREP PAC IMAGE DATA FOR RESNET ??
    # preprocess_input = tf.keras.applications.resnet50.preprocess_input
    # train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y))
    # validation_ds = validation_ds.map(lambda x, y: (preprocess_input(x), y))
    
    # coga_model = Sequential()
    # coga_model.add(rn)
    # # coga_model.layers[0].trainable=False
    # coga_model.add(Dense(1024, activation='relu'))
    # coga_model.add(Dense(10, activation='relu'))
    # coga_model.add(Dense(1, activation='sigmoid'))

    x = rn.output
    # x = GlobalAveragePooling2D()(x)  # Add a global spatial average pooling layer
    # x = Flatten()(x)  # Flatten the output to feed into a Dense layer
    x = Dense(2048, activation='relu')(x)  # Add a fully connected layer with 1024 units and ReLU activation
    # x = Dropout(0.25)(x)
    # x = Dense(512, activation='relu')(x)  # Add a fully connected layer with 1024 units and ReLU activation
    # x = Dense(256, activation='relu')(x)  # Add a fully connected layer with 1024 units and ReLU activation
    # x = Dense(128, activation='relu')(x)  # Add a fully connected layer with 1024 units and ReLU activation
    # x = Dense(64, activation='relu')(x)  # Add a fully connected layer with 1024 units and ReLU activation
    # x = Dense(32, activation='relu')(x)  # Add a fully connected layer with 1024 units and ReLU activation
    # x = Dense(16, activation='relu')(x)  # Add a fully connected layer with 1024 units and ReLU activation
    predictions = Dense(1, activation='sigmoid')(x)  # Add the final output layer with one neuron and sigmoid activation for binary classification
    # predictions = Dense(k-1, activation='softmax')(x)  # Add the final output layer with one neuron and sigmoid activation for binary classification
    
    # This is the model we will train
    # coga_model = Model(inputs=rn.input, outputs=x)
    # input_layer = Input(shape=(224,224,3))
    # x = BatchNormalization()(input_layer)
    # x = rn(x)
    # predictions = Dense(1, activation='sigmoid')(x)
    coga_model = Model(inputs=rn.input, outputs=predictions)
    
                
    # coga_model.add(rn50)
    # coga_model.layers[0].trainable=False
    # coga_model.add(Flatten())
    # coga_model.add(Dense(512, activation='relu'))
    # coga_model.add(Dense(1, activation='sigmoid'))

    # coga_model.add(K.layers.Flatten())
    # coga_model.add(K.layers.BatchNormalization())
    # coga_model.add(K.layers.Dense(256, activation='relu'))
    # coga_model.add(K.layers.Dropout(0.5))
    # coga_model.add(K.layers.BatchNormalization())
    # coga_model.add(K.layers.Dense(128, activation='relu'))
    # coga_model.add(K.layers.Dropout(0.5))
    # coga_model.add(K.layers.BatchNormalization())
    # coga_model.add(K.layers.Dense(64, activation='relu'))
    # coga_model.add(K.layers.Dropout(0.5))
    # coga_model.add(K.layers.BatchNormalization())
    # # coga_model.add(K.layers.Dense(10, activation='softmax'))
    # coga_model.add(Dense(1, activation='sigmoid'))
   
    # CHECK LAYERS
    # for i, layer in enumerate(rn.layers): print(i, layer.name, "-", layer.trainable)
    # for i, layer in enumerate(coga_model.layers): print(i, layer.name, "-", layer.trainable)
    # coga_model.summary()

    # INSERT SOME OTHER CLASSICAL MACHINE LEARNNG TECHNIQUE HERE 
    # sklearn
    # GET LINK FROM JT; 
    # ASK CHATGPT: 
    # I HAVE A KERAS DATALOADER THAT TRAINS A MODEL - HOW CAN I TRAIN AN SKLEARN 
    # LOGISTIC REGRESSION MODEL USING RESNET50 EMBEDDING OF SHAPE 2048?

    coga_model.compile(optimizer=Adam(learning_rate=learning_rate),
                       loss=tf.keras.losses.BinaryCrossentropy(),
                       metrics=['accuracy'])
    # loss=tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    
    history = coga_model.fit(train_ds, 
                             validation_data=validation_ds, 
                             epochs=epochs,
                             batch_size=batch_size,
                             verbose=1)

    # SAVE THIS MODEL
    if save_resnet_model:
        coga_model.save(base_dir + 'MODEL_' + targ_folder + '.keras')
    
    
    title_str+= ' N=' + N_str + ' alc=' + alc + '% lr=' + str(learning_rate) + ' pool=' + pooling + ' dropout=F' 
    fn = title_str.replace('(','')
    fn = fn.replace(')','')
    fn = fn.replace('=','_')
    fn = fn.strip() + '.jpg'
    
    plotter_lib.figure(figsize=(8, 8))
    epochs_range= range(epochs)
    plotter_lib.plot( epochs_range, history.history['accuracy'], label="Training Accuracy")
    plotter_lib.plot(epochs_range, history.history['val_accuracy'], label="Validation Accuracy")
    plotter_lib.axis(ymin=0.4,ymax=1.09)
    plotter_lib.grid()
    plotter_lib.title(data_str + ' ' + title_str)
    plotter_lib.ylabel('Accuracy')
    plotter_lib.xlabel('Epochs')
    plotter_lib.legend(['train', 'validation'])    
    
    plotter_lib.figure(figsize=(8, 8))
    epochs_range= range(epochs)
    plotter_lib.plot( epochs_range, history.history['loss'], label="Training Loss")
    plotter_lib.plot(epochs_range, history.history['val_loss'], label="Validation Loss")
    plotter_lib.axis(ymin=0,ymax=3)
    plotter_lib.grid()
    plotter_lib.title(data_str + ' ' + title_str)
    plotter_lib.ylabel('Loss')
    plotter_lib.xlabel('Epochs')
    plotter_lib.legend(['train', 'validation'])  








if resnet_to_logistic:
    
    # import matplotlib.pyplot as plotter_lib
    import numpy as np
    from PIL import Image
    import tensorflow as tf
    from tensorflow.keras.layers import BatchNormalization, Input, GlobalAveragePooling2D, Flatten, Dropout
    from keras.layers.core import Dense
    from keras.applications.resnet import preprocess_input
    from keras.models import Model

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.layers.experimental.preprocessing import Rescaling
    from tensorflow.keras import regularizers
    from sklearn.model_selection import KFold
    from sklearn.model_selection import train_test_split
    from sklearn import metrics, svm
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import classification_report
    
    # Assuming clf is your trained classifier
    # test_data and test_labels are your testing data and labels

    which_dx = 'AUD' # AUD ALAB ALD
    sex = ''
    min_age = 20
    max_age = 50 
    race = ''
    flat_cut = 0 # MAXIMUM DURATION IN SECONDS OF FLAT INTERVAL IN EEG SIGNAL (<5uV)
    noise_cut = 0 # MAXIMUM DURATION IN SECONDS OF NOISE INTERVAL IN EEG SIGNAL (>100uV)

    
    # DEEP LEARNING MODEL
    learning_rate = .0001
    pooling = 'avg'
    img_height,img_width=224,224
    batch_size=32
    epochs=10
    
    include_top = False
    save_resnet_model = False
    
    # REGULARIZATION
    # Define L2 regularization factor
    alpha = .0001
    # USING KFold TO DO CROSS-VALIDATION
    # n_splits = 2 

    # PATHS AND DATA INFO
    base_dir = 'D:\\COGA_eec\\'
    targ_folder = 'resnet_by_subj_d_20_50_AUD_flat0_noise0_' # 'resnet_by_subj_20_40_cAUD_flat20_noise5'
    whichEEGfileExtention = 'jpg'
    data_str = 'FZ' # PAC@FZ chanxHz
    
    
    title_str = 'rn50-SVM d-RSV  ' + which_dx + ' ' + sex +  ' Age ' + str(min_age) + '-' + str(max_age) + ' f' + str(flat_cut) + 'n' + str(noise_cut) 
    pth = base_dir + targ_folder + '\\'
    fl = csd.get_file_list(pth, whichEEGfileExtention)    
    fl_alc = csd.get_file_list(pth + 'alcoholic\\', whichEEGfileExtention)
    alc = str(round((len(fl_alc)/len(fl))*100,1))
    N_str = str(len(fl))

    # LISTS TO HOLD ACCURACY AND LOSS FUNCTION VALUES FOR PLOTTING
    t_acc = []
    v_acc = []
    t_loss = []
    v_loss = []
    
    # INPUT DATA AND LABELS TO PASS THROUGH KFold FUNCTION
    images = []
    labels = []
    for dx in ['alcoholic', 'nonalcoholic']:
        file_list = csd.get_file_list(pth + dx + '\\', whichEEGfileExtention)
        for i in file_list:
            img = Image.open(i[0] + i[1])
            img_array = np.array(img)
            img_array = preprocess_input(img_array)
            images.append(img_array)
            labels.append(dx)
    labels = np.array(labels)            
    labels[labels=='alcoholic'] = 1
    labels[labels=='nonalcoholic'] = 0
    labels = labels.astype(int)
    
    images = np.array(images)      
    labels = np.array(labels)      
    
    # kf = KFold(n_splits=n_splits, shuffle=True)    
    # for train_index, val_index in kf.split(images):
    #     X_train, X_val = images[train_index], images[val_index]
    #     y_train, y_val = labels[train_index], labels[val_index]
    
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

  
    rn = tf.keras.applications.ResNet152(
    # rn = tf.keras.applications.ResNet50(
        include_top=include_top,
        input_shape=(img_height, img_width,3),
        pooling=pooling,
        classes=2,
        weights='imagenet') # imagenet or None

    rn.trainable = False

    # Add L2 regularization to each convolutional and dense layer
    for layer in rn.layers:
        # if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
        if isinstance(layer, tf.keras.layers.Dense):
            layer.kernel_regularizer = regularizers.l2(alpha)
            if layer.use_bias:
                layer.bias_regularizer = regularizers.l2(alpha)


    coga_model = Sequential()
    coga_model.add(rn)
    # coga_model.add(GlobalAveragePooling2D())
    # coga_model.add(Flatten())
    # coga_model.add(Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(alpha)))
    # coga_model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(alpha)))

    train_features = rn.predict(X_train)
    val_features = rn.predict(X_val)
    
    # Train an SVM classifier on the features
    clf = svm.SVC()
    clf.fit(train_features, np.ravel(y_train))
    # Use the classifier to make predictions
    predictions = clf.predict(val_features)
    # Create a confusion matrix
    cm = metrics.confusion_matrix(y_val, predictions)
    
    # Plot the confusion matrix
    plt.figure(figsize=(10,10))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    
    # Show the plot
    plt.show()
    
    # Print a classification report
    print(classification_report(y_val, predictions))    
    
    
    








if do_cnn_pac:
    
    import matplotlib.pyplot as plotter_lib
    import numpy as np
    from PIL import Image
    import tensorflow as tf
    from tensorflow.keras.layers import BatchNormalization, Input, GlobalAveragePooling2D, Flatten, Dropout, Conv2D, MaxPooling2D
    from keras.layers.core import Dense
    from keras.applications.resnet import preprocess_input
    from keras.models import Model

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.layers.experimental.preprocessing import Rescaling
    from tensorflow.keras import regularizers
    from sklearn.model_selection import KFold
    from sklearn.model_selection import train_test_split


    which_dx = 'AUD' # AUD ALAB ALD
    sex = ''
    min_age = 0
    max_age = 99 
    race = ''
    flat_cut = 0 # MAXIMUM DURATION IN SECONDS OF FLAT INTERVAL IN EEG SIGNAL (<5uV)
    noise_cut = 0 # MAXIMUM DURATION IN SECONDS OF NOISE INTERVAL IN EEG SIGNAL (>100uV)

    
    # DEEP LEARNING MODEL
    learning_rate = .00001
    img_height,img_width=224,224
    batch_size=32
    epochs=100
    
    include_top = False
    save_resnet_model = False
    
    # REGULARIZATION
    # Define L2 regularization factor
    alpha = .0001
    # USING KFold TO DO CROSS-VALIDATION
    # n_splits = 2 

    # PATHS AND DATA INFO
    base_dir = 'D:\\COGA_eec\\'
    targ_folder = 'resnet_by_subj_d_20_50_AUD_flat0_noise0_' # 'resnet_by_subj_20_40_cAUD_flat20_noise5'
    whichEEGfileExtention = 'jpg'
    data_str = 'FZ' # PAC@FZ chanxHz
    
    
    title_str = 'rn50 d-RSV L2 alpha=' + str(alpha) + ' ' + which_dx + ' ' + sex +  ' Age ' + str(min_age) + '-' + str(max_age) + ' f' + str(flat_cut) + 'n' + str(noise_cut) 
    pth = base_dir + targ_folder + '\\'
    fl = csd.get_file_list(pth, whichEEGfileExtention)    
    fl_alc = csd.get_file_list(pth + 'alcoholic\\', whichEEGfileExtention)
    alc = str(round((len(fl_alc)/len(fl))*100,1))
    N_str = str(len(fl))

    # LISTS TO HOLD ACCURACY AND LOSS FUNCTION VALUES FOR PLOTTING
    t_acc = []
    v_acc = []
    t_loss = []
    v_loss = []
    
    # INPUT DATA AND LABELS TO PASS THROUGH KFold FUNCTION
    images = []
    labels = []
    for dx in ['alcoholic', 'nonalcoholic']:
        file_list = csd.get_file_list(pth + dx + '\\', whichEEGfileExtention)
        for i in file_list:
            img = Image.open(i[0] + i[1])
            img_array = np.array(img)
            img_array = preprocess_input(img_array)
            images.append(img_array)
            labels.append(dx)
    labels = np.array(labels)            
    labels[labels=='alcoholic'] = 1
    labels[labels=='nonalcoholic'] = 0
    labels = labels.astype(int)
    
    images = np.array(images)      
    labels = np.array(labels)      
    
    # kf = KFold(n_splits=n_splits, shuffle=True)    
    # for train_index, val_index in kf.split(images):
    #     X_train, X_val = images[train_index], images[val_index]
    #     y_train, y_val = labels[train_index], labels[val_index]
    
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

  
    # # rn = tf.keras.applications.ResNet152(
    # rn = tf.keras.applications.ResNet50(
    #     include_top=include_top,
    #     input_shape=(img_height, img_width,3),
    #     pooling=pooling,
    #     classes=2,
    #     weights='imagenet') # imagenet or None

    # rn.trainable = False

    # # Add L2 regularization to each convolutional and dense layer
    # for layer in rn.layers:
    #     # if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
    #     if isinstance(layer, tf.keras.layers.Dense):
    #         layer.kernel_regularizer = regularizers.l2(alpha)
    #         if layer.use_bias:
    #             layer.bias_regularizer = regularizers.l2(alpha)

    # rn_partial = tf.keras.Model(inputs = rn.input, outputs = rn.layers[18].output)
    # output = rn_partial.output
    # predictions = Dense(1, activation='sigmoid')(output)
    # coga_model = tf.keras.Model(inputs = rn_partial.input, outputs = predictions)
    
    # coga_model = Sequential()
    # # coga_model.add(rn)
    # # coga_model.add(GlobalAveragePooling2D())
    # coga_model.add(Flatten())
    # coga_model.add(Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(alpha)))
    # coga_model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(alpha)))


    coga_model = Sequential()
    coga_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_height,img_width, 3)))
    coga_model.add(MaxPooling2D((2, 2)))
    coga_model.add(Conv2D(64, (3, 3), activation='relu'))
    coga_model.add(MaxPooling2D((2, 2)))
    coga_model.add(Conv2D(64, (3, 3), activation='relu'))
    coga_model.add(BatchNormalization())
    # coga_model.add(Flatten())
    
    coga_model.add(Dense(64, activation='relu'))
    coga_model.add(Dense(64, activation='relu'))
    coga_model.add(Dense(1, activation='sigmoid'))



    coga_model = Sequential()
    coga_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_height,img_width, 3)))
    coga_model.add(MaxPooling2D((2, 2)))
    coga_model.add(Conv2D(64, (3, 3), activation='relu'))
    coga_model.add(MaxPooling2D((2, 2)))
    coga_model.add(Conv2D(64, (3, 3), activation='relu'))
    coga_model.add(BatchNormalization())   
    coga_model.add(Dense(64, activation='relu'))
    coga_model.add(Dense(64, activation='relu'))
    coga_model.add(Dense(1, activation='sigmoid'))



    coga_model.compile(optimizer=Adam(learning_rate=learning_rate),
                       loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                       metrics=['accuracy'])
    
    
    history = coga_model.fit(X_train, 
                             y_train,
                             validation_data=(X_val, y_val), 
                             epochs=epochs,
                             batch_size=batch_size,
                             verbose=1)
    t_acc = history.history['accuracy']
    v_acc = history.history['val_accuracy']
    t_loss = history.history['loss']
    v_loss = history.history['val_loss']
        
    
    # SAVE THIS MODEL
    if save_resnet_model:
        coga_model.save(base_dir + 'MODEL_' + targ_folder + '.keras')
        
        
    title_str+= ' N=' + N_str + ' alc=' + alc + '% lr=' + str(learning_rate) 
    fn = title_str.replace('(','')
    fn = fn.replace(')','')
    fn = fn.replace('=','_')
    fn = fn.strip() + '.jpg'

    epochs_range= range(epochs)
    
    plotter_lib.figure(figsize=(8, 8))
    plotter_lib.plot(epochs_range, t_acc, label="Training Accuracy")
    plotter_lib.plot(epochs_range, v_acc, label="Validation Accuracy")
    plotter_lib.axis(ymin=0.4,ymax=1.09)
    plotter_lib.grid()
    plotter_lib.title(data_str + ' ' + title_str)
    plotter_lib.ylabel('Accuracy')
    plotter_lib.xlabel('Epochs')
    plotter_lib.legend(['train', 'validation'])    
    
    plotter_lib.figure(figsize=(8, 8))
    plotter_lib.plot(epochs_range, t_loss, label="Training Loss")
    plotter_lib.plot(epochs_range, v_loss, label="Validation Loss")
    plotter_lib.axis(ymin=0,ymax=max(v_loss))
    plotter_lib.grid()
    plotter_lib.title(data_str + ' ' + title_str)
    plotter_lib.ylabel('Loss')
    plotter_lib.xlabel('Epochs')
    plotter_lib.legend(['train', 'validation'])  
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    # coga_model.compile(optimizer=Adam(learning_rate=learning_rate),
    #                    loss=tf.keras.losses.BinaryCrossentropy(),
    #                    metrics=['accuracy'])
    
    
    # history = coga_model.fit(X_train, 
    #                          y_train,
    #                          validation_data=(X_val, y_val), 
    #                          epochs=epochs,
    #                          batch_size=batch_size,
    #                          verbose=1)
    # t_acc = history.history['accuracy']
    # v_acc = history.history['val_accuracy']
    # t_loss = history.history['loss']
    # v_loss = history.history['val_loss']
        
    
    # # SAVE THIS MODEL
    # if save_resnet_model:
    #     coga_model.save(base_dir + 'MODEL_' + targ_folder + '.keras')
        
        
    # title_str+= ' N=' + N_str + ' alc=' + alc + '% lr=' + str(learning_rate) + ' pool=' + pooling
    # fn = title_str.replace('(','')
    # fn = fn.replace(')','')
    # fn = fn.replace('=','_')
    # fn = fn.strip() + '.jpg'

    # epochs_range= range(epochs)
    
    # plotter_lib.figure(figsize=(8, 8))
    # plotter_lib.plot(epochs_range, t_acc, label="Training Accuracy")
    # plotter_lib.plot(epochs_range, v_acc, label="Validation Accuracy")
    # plotter_lib.axis(ymin=0.4,ymax=1.09)
    # plotter_lib.grid()
    # plotter_lib.title(data_str + ' ' + title_str)
    # plotter_lib.ylabel('Accuracy')
    # plotter_lib.xlabel('Epochs')
    # plotter_lib.legend(['train', 'validation'])    
    
    # plotter_lib.figure(figsize=(8, 8))
    # plotter_lib.plot(epochs_range, t_loss, label="Training Loss")
    # plotter_lib.plot(epochs_range, v_loss, label="Validation Loss")
    # plotter_lib.axis(ymin=0,ymax=max(v_loss))
    # plotter_lib.grid()
    # plotter_lib.title(data_str + ' ' + title_str)
    # plotter_lib.ylabel('Loss')
    # plotter_lib.xlabel('Epochs')
    # plotter_lib.legend(['train', 'validation'])  





    
    

if 0:
    # import scipy.stats as ss
    import yaml
    
    # fa = pacdat[(pacdat.AUD_this_visit==True) & (pacdat.sex=='F')].age_this_visit
    # fna = pacdat[(pacdat.AUD_this_visit==False) & (pacdat.sex=='F')].age_this_visit
        
    # ss.ttest_ind(fa,fna, equal_var=False)
    
    yml_p = 'C:\\Users\\lifep\\'
    yml_f = 'pac3.yml'
    with open(yml_p + yml_f, 'r') as file:
        p = yaml.safe_load(file)
        # Loader=yaml.FullLoader
        print(p)
    
    
    
    
    
    
