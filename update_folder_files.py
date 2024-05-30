# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 18:42:16 2024

@author: lifep
"""

# do_remove_done_from_remaining_files

import coga_support_defs as csd
import os

# base_dir = 'D:\\COGA_eec\\update_folder_testing\\' #  BIOWIZARD
base_dir = "/ddn/crichard/pipeline/"
#read_dir = os.environ['TMPDIR'] + '/input/'

read_folder, write_folder = 'reference','processed'
SourceEEGfileExtention = 'csv'
TargetEEGfileExtention = 'jpg'

# # GET MASTER TABLE OUT 
# pacdat = pd.read_pickle(base_dir + which_pacdat)

# # GET FILE LIST WITH GIVEN EXTENSION FROM SOURCE FOLDER 
# existingFiles = csd.get_file_list(base_dir + read_folder, SourceEEGfileExtention)
# cf = [f[1] for f in existingFiles]

for i in range(1,11):
    
    completeFiles = csd.get_file_list(base_dir + write_folder + str(i), TargetEEGfileExtention)
    cf = [f[1] for f in completeFiles]
    cff = [str.split(fn,'_') for fn in cf]
    cfff = set(['_'.join(fn[0:len(fn)-1])+'.'+SourceEEGfileExtention for fn in cff])
    fcount = 0 
    
    rl = [base_dir + read_folder + str(i) + '/' + f for f in cfff]
    wl = [base_dir + read_folder + 's/' + f for f in cfff]
    
    for f in range(0,len(rl)):
        if os.path.isfile(rl[f]):
            fcount+=1
            print(rl[f] + '  ' + wl[f])
            os.rename(rl[f], wl[f])
            
    # os.rename(base_dir + write_folder + str(i) + '/*', base_dir + write_folder + 's/')
    print('folder ' + read_folder + str(i) + ' tranfered single EEG channel .CSV file count = ' + str(fcount))
