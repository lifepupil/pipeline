# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 18:42:16 2024

@author: lifep
"""

# do_remove_done_from_remaining_files

import coga_support_defs as csd
import os

base_dir = 'D:\\COGA_eec\\' #  BIOWIZARD
read_folder, write_folder = 'cleaned_data_test','resnet_by_subj_5'
SourceEEGfileExtention = 'csv'
TargetEEGfileExtention = 'jpg'

# # GET MASTER TABLE OUT 
# pacdat = pd.read_pickle(base_dir + which_pacdat)

# # GET FILE LIST WITH GIVEN EXTENSION FROM SOURCE FOLDER 
# existingFiles = csd.get_file_list(base_dir + read_folder, SourceEEGfileExtention)
# cf = [f[1] for f in existingFiles]

completeFiles = csd.get_file_list(base_dir + write_folder, TargetEEGfileExtention)
cf = [f[1] for f in completeFiles]
cff = [str.split(fn,'_') for fn in cf]
cfff = set(['_'.join(fn[0:len(fn)-1])+'.'+SourceEEGfileExtention for fn in cff])
i = 0 

rl = [base_dir + read_folder + '\\' + f for f in cfff]

for f in rl:
    if os.path.isfile(f):
        i+=1
        print(f)
        os.remove(f)
        
print('\n tranfered single EEG channel .CSV file count = ' + str(i))