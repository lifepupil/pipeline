# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 13:02:08 2025
EEG preprocessing


@author: lifep
"""


import numpy as np
import pandas as pd 
import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components
import matplotlib.pyplot as plt
import coga_support_defs as csd
import pybv
from pyprep.prep_pipeline import PrepPipeline



# The PREP pipeline is freely available under the GNU General Public License. 
# Please cite the following publication if using:
# Bigdely-Shamlo N, Mullen T, Kothe C, Su K-M and Robbins KA (2015)
# The PREP pipeline: standardized preprocessing for large-scale EEG analysis
# Front. Neuroinform. 9:16. doi: 10.3389/fninf.2015.00016


# PARAMETERS FOR do_filter_eeg_signal_cnt 
notch_freq = 60.0       # FREQUENCY (Hz) TO REMOVE LINE NOISE FROM SIGNAL 
lowfrq = 1              # LOW PASS FREQUENCY, RECOMMENDED SETTING TO 1 HZ IF USING mne-icalabel
hifrq = 100             # HIGH PASS FREQUENCY
maxZeroPerc = 0.5       # PERCENTAGE OF ZEROS IN SIGNAL ABOVE WHICH CHANNEL IS LABELED 'BADS'
do_plot_channels = True # TO GENERATE PLOTS OF THE CLEANED EEG SIGNAL
# mpl.rcParams['figure.dpi'] = 300 # DETERMINES THE RESOLUTION OF THE EEG PLOTS
eye_blink_chans = ['X', 'Y'] # NAMES OF CHANNELS CONTAINING EOG
write_to_csv = False

# base_dir = "E:\\Documents\\COGA_eec\\data\\"
base_dir = 'D:\\COGA_eec\\0_gonogo\\test_data\\'
write_dir = "D:\\COGA_eec\\0_gonogo\\cleaned_data\\"

# TO FILTER SIGNALS
# GET ALL THE .CNT FILE NAMES AND PATHS AND PUT INTO LIST 
cntList = csd.get_file_list(base_dir, 'cnt')
# NOW WE REMOVE COMPLETED FILES FROM THE MAIN LIST OF FILES TO PROCESS, I.E., FROM cntList
# cntList = csd.remove_completed_files_from_list(cntList, write_dir + 'cleaned_data\\', institutionDir)
totalFileCount = str(len(cntList))
#  WE GO THROUGH EACH OF THE FILES IN cntList 
for f in range(len(cntList)):
    fname = cntList[f][1]
    print('\n\n\nWORKING ON ' + str(f+1) + ' OF THE ' + totalFileCount + ' CNT FILES\n' + fname + '\n\n')
    path_and_file = cntList[f][0] + fname
    try:
        data = mne.io.read_raw_cnt(path_and_file, data_format='int32' ,preload=True, verbose=False)
        # print(data.info["bads"])
        # data.plot()
        # a=1
    except Exception as e:
        with open(base_dir + 'errors_from_core_pheno.txt', 'a') as bf:
            bf.write(str(fname) + '\t ' + str(e) + '\n')
        continue                
    
    data.drop_channels(['BLANK'], on_missing='warn')
       
    channels = data.ch_names
    info = data.info
    # WE EXCLUDE THE BLANK CHANNEL AND RELABEL CHANNEL TYPES OF THE TWO EYE CHANNELS TO eog
    # ASSUMES THAT ALL CHANNELS ARE LABELED AS EEG WHETHER THEY ARE OR NOT
    # for ch in eye_blink_chans:
    #     if ch in channels:
    #         data.set_channel_types({ch: 'eog'})
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
    
    # ATTEMPT AT USING pyprep BUT IT WAS REJECTING CHANNELS THAT VISIBLY WERE NOT BAD
    # SHOULD BE TRIED AGAIN BY PLAYING WITH THE ARGUMENTS FOR THE pyprep.PrepPipeline
    # FUNCTION CALL, OR BY USING THE pyprep.NoisyChannels CLASS
    
    # WE EXCLUDE THE BLANK CHANNEL AND RELABEL CHANNEL TYPES OF THE TWO EYE CHANNELS TO eog
    data.set_channel_types({'X': 'eog', 'Y': 'eog'})
    filtered_data = data.copy()
    # filtered_data.notch_filter(60, filter_length='auto', phase='zero')
    prep_params = {
        "ref_chs": "eeg",
        "reref_chs": "eeg",
        "line_freqs": np.arange(60, int(data.info['sfreq']) / 2, 60)
    }
    prep = PrepPipeline(filtered_data, prep_params, filtered_data.get_montage())
    prep.fit()
    print("Bad channels: {}".format(prep.interpolated_channels))
    print("Bad channels original: {}".format(prep.noisy_channels_original["bad_all"]))
    print("Bad channels after interpolation: {}".format(prep.still_noisy_channels))
    filtered_data.plot()
    
    montage = mne.channels.make_standard_montage('standard_1005')

    
    
    
    # NOW WE PERFORM PREPROCESSING STEPS ON THE (COMPLETELY) RAW DATA FROM THE .CNT FILES
    # LOW AND HIGH PASS FILTERING THAT SATISFIES ZERO-PHASE DESIGN
    filtered_data = data.copy().filter(lowfrq, hifrq)
    # REMOVE 60 HZ LINE NOISE FROM SIGNAL WITH NOTCH FILTER
    filtered_data.notch_filter(notch_freq, filter_length='auto', phase='zero', verbose=False)
    # WE NEED TO APPLY A COMMON AVERAGE REFERENCE TO USE MNE-ICALabel
    # UNCLEAR WHETHER AVERAGE SHOULD INCLUDE OR EXCLUDE THE EYE CHANNELS 
    # ALSO, WE WANT TO EXCLUDE BAD CHANNELS BEFORE THIS STEP SO WE MUST 
    # HAVE A PRELIMINARY CHECK OF CONSPICUOUSLY BAD CHANNELS            
    
    # SETS AVERAGE REFERENCE AND MONTAGE PROJECTION
    filtered_data = filtered_data.set_eeg_reference(ref_channels='average',projection=True)
    filtered_data.apply_proj()
    # TO PLOT THE AVERAGE PSD FOR THIS EEG RECORDING
    # data.compute_psd(fmax=100).plot(average=True, amplitude=False, picks='data')
    # filtered_data.compute_psd(fmax=100).plot(average=True, amplitude=False, picks='data')
    
    # NOW WE DO ICA FOR ARTIFACT REMOVAL
    # EYE CHANNELS SHOULD BE INCLUDED 
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
    if 'eog' not in filtered_data.get_channel_types():
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
    # data.plot(show_scrollbars=False, title='raw')
    # reconst_data.plot(show_scrollbars=False, title='reconstr')
    
    filtered_data.save(write_dir + fname[:-4] + '_eeg.fif', overwrite=True)
    # mne.export.export_raw(write_dir + fname, filtered_data,fmt='brainvision', overwrite=True)
    
    if write_to_csv:
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
                figFN = ch + '_' + fname[:-4] + '_' + str(samp_freq) + '_eeg' + '.png'
                plt.plot(this_chan)
                # plt.ylim((-50/1000000),(50/1000000))
                plt.title(ch + ', ' + ' -- ' + fname[:-4])
                # plt.show()
                plt.savefig(write_dir + 'eeg_figures\\' + figFN)
                plt.clf()