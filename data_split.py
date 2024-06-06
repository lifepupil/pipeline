# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 12:56:17 2024

@author: lifep
"""

# THIS FUNCTION WRITTEN TO AUTOMATE DIVISION OF ALL FILES TO BE PROCESSED INTO 
# THEIR OWN FOLDERS SO THAT THEY CAN BE RUN ON HPC WITH MULTIPLE JOB SCRIPTS

import coga_support_defs as csd
import pandas as pd
import os
import shutil

base_dir = 'fc2o2_'
whichEEGfileExtention = 'csv'
num_nodes = 4 # HOW MANY HPC NODES (AND THEREFORE SEPARATE FOLDERS) TO PUT FILES INTO 
# read_dir = 'D:\\COGA_eec\\FC2O2\\'
# write_dir = 'D:\\COGA_eec\\'
read_dir = "/ddn/crichard/eeg_csv/FC2O2/"
write_dir = "/ddn/crichard/eeg_csv/"

# MAKE DATAFRAME WITH ALL PATHS AND FILENAMES AT read_dir
fileList = csd.get_file_list(read_dir, whichEEGfileExtention)
file_info = pd.DataFrame(fileList, columns=['dir','fn'])

# NOW WE WANT TO EXTRACT FILE NAME INFO EXCLUDING CHANNEL SO THAT WE CAN SORT ON THAT
subvis = [f.split('_')[1:] for f in file_info.fn]
subvis = ['_'.join(f) for f in subvis]
file_info.insert(2,'subvis',subvis)
# NEXT WE SORT ON THE SUBJECT AND VISIT INFO FROM THE FILENAMES
file_info = file_info.sort_values(by='subvis')

# NOW WE COMPUTE THE INTERVALS TO USE FOR COPYING OVER FILES FOR EACH NODE
fileNum = len(file_info)
# WE WANT TO MOVE PAIRS OF FILES FROM DIFFERENT CHANNELS BUT SAME SUBJECT AND VISIT
# SO WE NEED TO MAKE SURE THAT THE step VALUE IS EVEN
step = int((fileNum/num_nodes))
if step%2!=0:
    step +=1
    
index_intervals = list(range(0,fileNum,step))
# SINCE THE FINAL INTERVAL INDEX MAY NOT GO UP TO THE LAST FILE WE NEED TO 
# CHECK WHETHER index_intervals ACCOUNTS FOR ALL FILES FOR EACH NODE
# FIRST, THE LENGTH OF index_intervals NEEDS TO BE num_nodes MINUS 1
if len(index_intervals)-1==num_nodes:
    index_intervals[-1] = fileNum
else:
    index_intervals.append(fileNum)
 
# NOW WE MOVE FILES USING VALUES IN index_intervals
for ii in range(0,len(index_intervals)-1): 
    start = index_intervals[ii]
    end = index_intervals[ii+1]
    # CHECK TO SEE IF FOLDER EXISTS, IF NOT MAKE IT
    trg_path = write_dir + base_dir + str(ii+1) + '/'
    if not os.path.exists(trg_path):
        os.makedirs(trg_path)
    
    # file_batch = file_info.apply(lambda row: row['dir']+row['fn'], axis=1)
    file_batch = file_info['fn'].tolist()
    # FINALLY WE MOVE FILE PAIRS THAT GO TOGETHER FOR EACH BATCH OF FILES
    for f in file_batch:
        src = read_dir + f
        trg = trg_path + f
        shutil.copy(src, trg)

