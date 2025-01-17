# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 18:18:59 2024

@author: lifep
"""

# MOVE PAC IMAGE FILES FROM SAME SUBJECT

import shutil
import random
from PIL import Image
import coga_support_defs as csd
import pandas as pd
import numpy as np
import os 

use_pickle = False
which_dx = 'AUD' # AUD ALAB ALD
sex = '' # M F
min_age = 0 
max_age = 99
race = ''
channel = 'FZ'
channelstr = 'fz'
start_seg = 0
end_seg = 3

base_dir = 'D:\\COGA_eec\\' #  BIOWIZARD
source_folder = 'new_pac_' + channelstr # eeg_figures | new_pac | new_pac_fz
write_folder = 'new_pac_fz_AVG' + '_' + str(start_seg) + '_' + str(end_seg)
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
pd_filtered = pacdat[(pacdat.channel==channel)]
sexlbl = 'both'

if use_pickle:
    pd_filtered =  pd.read_pickle('D:\\COGA_eec\\TEMP\\pd_fn__16_25_AUD.pkl')

jpg_subj = set([int(i) for i in set(file_info.ID)])
pd_subj =  set([int(i) for i in set(pd_filtered.ID)])
overlap = jpg_subj.intersection(pd_subj)

filtered_N = str(len(set(pd_filtered.ID)))
qualifying_N = str(len(overlap))
total_N = str(len(pd_filtered))

print('There are N = ' + qualifying_N + ' subjects represented in PAC heatmap dataset')
print('out of ' + filtered_N + ' total subjects in this dataset')
print('total # files = ' + total_N)
print('Excluding EEG signals with:')
print(' - Ages ' + str(min_age) + '-' + str(max_age) + '')
print(' - Sex: ' + sexlbl)

# CYCLE THROUGH EVERY SUBJECT REPRESENTED IN FILES FROM SOURCE FOLDER
all_subj_figs = pd.unique(file_info.ID) 
seg_fn = np.arange(start_seg, end_seg)

for i in range(0,len(all_subj_figs)):
    this_subj = all_subj_figs[i]
    # this_subj = '10071029'
    
    # FIND ALL VISITS FOR A SUBJECT THEN FILTER BY AGE
    svisits = file_info[(file_info.ID==this_subj)]
    if len(svisits)>0:
        # print(this_subj)        
        for v in set(svisits.this_visit):
        
            svisit = svisits[svisits.this_visit==v]
            if len(svisit)>=len(seg_fn):
                base_fn = ('_').join(svisit.iloc[0].fn.split('_')[:-1])
    
                
                # ssss = [s.split('_')[-1].split('.')[0] for s in svisits.fn]
                lbl_224 = [str(i) for i in np.arange(0,224)]
                pac_avg = pd.DataFrame((np.zeros((224,224))), columns=lbl_224)
                
                for f in seg_fn:
                    this_file = base_fn + '_t' + str(f) + '.jpg'
                    src = base_dir + source_folder + '\\' + this_file
    
                    if not os.path.exists(src):
                        print(src)
                        continue
                    
                    img = Image.open(src) 
                    grayImage = img.convert('L')
                    # grayImage.show()
                    array = np.array(grayImage) 
                    this_seg = pd.DataFrame(array,columns=lbl_224)
                    pac_avg = pac_avg.add(this_seg, fill_value=0)
                    
                pa = pac_avg.div(len(seg_fn))
                # pa.stack().hist(grid=False)
                pa = pa.round()
                # CONVERT AVERAGE PAC MATRIX TO JPEG
                pa = Image.fromarray(np.array(pa))
                pa = pa.convert('L')
                trg = 'D:\\COGA_eec\\' + write_folder + '\\' 
                if not os.path.exists(trg):
                    os.makedirs(trg)
                pa.save(trg + base_fn + '.jpg')