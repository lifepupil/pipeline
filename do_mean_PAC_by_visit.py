# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 21:22:50 2025

@author: lifep
"""


import os
import numpy as np
import coga_support_defs as csd
import pandas as pd
import shutil
import random
from PIL import Image
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, mannwhitneyu, sem, linregress


use_pickle = False
sex = '' # M F
min_age = 1
max_age = 99
start_seg = 0
end_seg = 3
channel = 'FZ'
channelstr = 'fz'
pac_len = 224


base_dir = 'D:\\COGA_eec\\' #  BIOWIZARD
source_folder = 'new_pac_fz' 
targ_folder = 'new_pac_' + channelstr + '_AVG_' + str(start_seg) + '_' + str(end_seg) + '_NOBORDER_test'
whichEEGfileExtention = 'jpg' # png jpg
which_pacdat = 'pacdat_MASTER.pkl'

# CONSTANTS    
chan_i = 0 
visit_i = 3 
id_i = 4  

targ_folder = base_dir + targ_folder + '\\' 
if not os.path.exists(targ_folder):
    os.makedirs(targ_folder) 

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
    
    
    # alc = pacdat[(pacdat.AUD_this_visit==True) & (pacdat.ald5sx_cnt>=6) & (pacdat.channel==channel) & (pacdat.age_this_visit>=min_age) & (pacdat.age_this_visit<=max_age)]
    # ctl = pacdat[(pacdat.AUD_this_visit==False) & (pacdat.ald5sx_cnt.isnull()) & (pacdat.channel==channel) & (pacdat.age_this_visit>=min_age) & (pacdat.age_this_visit<=max_age)]
    # pd_filtered = pd.concat([alc,ctl])
    
    pd_filtered = pacdat[(pacdat.channel==channel) & (pacdat.age_this_visit>=min_age) & (pacdat.age_this_visit<=max_age)]
    # pd_filtered = pacdat[(pacdat.channel==channel) & (pacdat.age_this_visit>=min_age) & (pacdat.age_this_visit<=max_age) & (pacdat.flat_score<=flat_cut) & (pacdat.noise_score<=noise_cut)]
    # pd_filtered = pacdat[(pacdat.channel==channel) & (pacdat.age_this_visit>=min_age) & (pacdat.age_this_visit<=max_age) & ((pacdat.perc_flat_slip1<=flat_cut) & (pacdat.max_noise<=noise_cut))]
    # pd_filtered = pacdat[(pacdat.channel==channel) & (pacdat.age_this_visit>=min_age) & (pacdat.age_this_visit<=max_age) & (pacdat.perc_flat_slip0<=flat_cut) & (pacdat.perc_noise_slip0<=noise_cut)]
    # pd_filtered = pacdat[(pacdat.channel==channel) & (pacdat.age_this_visit>=min_age) & (pacdat.age_this_visit<=max_age) & (pacdat.max_flat_slip0<=flat_cut) & (pacdat.max_noise_slip0<=noise_cut)]
    # pd_filtered = pacdat[(pacdat.channel==channel) & (pacdat.age_this_visit>=min_age) & (pacdat.age_this_visit<=max_age) & ((pacdat.max_flat<=flat_cut) | (pacdat.max_noise<=noise_cut)) & (pacdat.race==race)]
    
    sexlbl = 'both'

else:            
    alc = pacdat[(pacdat.AUD_this_visit==True) & (pacdat.ald5sx_cnt>=6) & (pacdat.channel==channel) & (pacdat.age_this_visit>=min_age) & (pacdat.age_this_visit<=max_age) & (pacdat.sex_x==sex)]
    ctl = pacdat[(pacdat.AUD_this_visit==False) & (pacdat.ald5sx_cnt.isnull()) & (pacdat.channel==channel) & (pacdat.age_this_visit>=min_age) & (pacdat.age_this_visit<=max_age) & (pacdat.sex_x==sex)]
    pd_filtered = pd.concat([alc,ctl])
    # pd_filtered = pacdat[(pacdat.channel==channel) & (pacdat.age_this_visit>=min_age) & (pacdat.age_this_visit<=max_age) & (pacdat.sex_x==sex)  & (pacdat.max_flat_slip0<=flat_cut) & (pacdat.max_noise_slip0<=noise_cut)]
    # pd_filtered = pacdat[(pacdat.channel==channel) & (pacdat.age_this_visit>=min_age) & (pacdat.age_this_visit<=max_age) & (pacdat.sex==sex) & ((pacdat.max_flat<=flat_cut) | (pacdat.max_noise<=noise_cut))]
    # pd_filtered = pacdat[(pacdat.channel==channel) & (pacdat.age_this_visit>=min_age) & (pacdat.age_this_visit<=max_age) & (pacdat.sex==sex) & ((pacdat.max_flat<=flat_cut) | (pacdat.max_noise<=noise_cut)) & (pacdat.race==race)]
    sexlbl = sex

if use_pickle:
    pd_filtered =  pd.read_pickle('D:\\COGA_eec\\TEMP\\pd_fn__16_25_AUD.pkl')

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

# WE NEED A RECORD OF WHICH IMAGES HAD BORDERS OR NOT AND 
top_record = []
bottom_record = []
left_record = []
right_record = []
jpg_record = []
missing_jpgs = []

seg_fn = np.arange(start_seg,end_seg+1)                    
lbl_224 = [str(i) for i in np.arange(0,224)]
lbl_214 = [str(i) for i in np.arange(0,214)]
lbl_212 = [str(i) for i in np.arange(0,212)]

# EXPECTED BORDER EDGES 
# box=(left, upper, right, lower)
pac_image = (5, 6, pac_len-5, pac_len-6)

# CYCLE THROUGH EVERY SUBJECT REPRESENTED IN FILES FROM SOURCE FOLDER
all_subj_figs = pd.unique(file_info.ID) 
# all_subj_figs = (np.array(list(overlap)))
for i in range(0,len(all_subj_figs)):
    # this_subj = all_subj_figs[i]
    this_subj = '61174006'
    
    # FIND ALL VISITS FOR A SUBJECT 
    svisits = file_info[(file_info.ID==this_subj)]
    if len(svisits)>0:
        # print(this_subj)

        # WE WANT TO INCLUDE AUD DIAGNOSES FOR QUICK REF
        vinfo = pd_filtered[(pd_filtered.ID==int(this_subj))]
        if len(vinfo)>0:
            
            # sv CONTAINS ORDINALS FOR ALL VISITS FOR THIS SUBJECT FROM svisits
            sv = list(set(svisits.this_visit.values))
            # vi CONTAINS ORDINALS FOR VISITS THAT SATISFY AGE CRITERIA
            vi = list(set(vinfo.this_visit.values))
            # REMOVE VISITS NOT WITHIN AGE REQUIREMENTS FROM svisits
            for vs in sv:
                if vs not in vi:
                    svisits = svisits[svisits.this_visit!=vs]
                    
                
            if len(svisits)>0:

                visit_order = list(set(svisits.this_visit))
                for vo in visit_order:
                    
                    svisit = svisits[svisits.this_visit==vo]
                    
                    if len(svisit)>=end_seg+1:

                        base_fn = ('_').join(svisit.iloc[0].fn.split('_')[:-1])
                        pac_avg = pd.DataFrame((np.zeros((212,214))), columns=lbl_214, index=lbl_212)
                            
                        for f in seg_fn:
                            this_file = base_fn + '_t' + str(f) + '.jpg'                                
                            src = base_dir + source_folder + '\\' + this_file
                            # if not(os.path.isfile(src)):
                            #     missing_jpgs.append(src)
                            #     break
                            img = Image.open(src) 
                            grayImage = img.convert('L')
                            # grayImage.show()
                            array = np.array(grayImage) 
                            this_seg = pd.DataFrame(array, columns=lbl_224, index=lbl_224)
                            # EXPECTED BORDER WIDTHS AT EACH SODE
                            # TOP - FIVE ELEMENTS ADJACENT TO EDGE, 219 TO 224
                            
                            if 0:
                                borw = 3
                                wbord = 'right'
                                border_cutoff = 240
                                if wbord=='bottom':
                                    ti = (this_seg.iloc[borw+1:borw+borw+1,30:200]).to_numpy().reshape(1,borw*170)[0]
                                    to = (this_seg.iloc[0:borw,30:200]).to_numpy().reshape(1,borw*170)[0]
                                elif wbord=='top':
                                    ti = (this_seg.iloc[pac_len-borw-borw-1:pac_len-borw-1,30:200]).to_numpy().reshape(1,borw*170)[0]
                                    to = (this_seg.iloc[pac_len-borw:pac_len,30:200]).to_numpy().reshape(1,borw*170)[0]
                                elif wbord=='right':
                                    ti = (this_seg.iloc[30:200, pac_len-borw-borw-1:pac_len-borw-1]).to_numpy().reshape(1,borw*170)[0]
                                    to = (this_seg.iloc[30:200, pac_len-borw:pac_len]).to_numpy().reshape(1,borw*170)[0]                          
                                elif wbord=='left':
                                    ti = (this_seg.iloc[30:200, borw+1:borw+borw+1]).to_numpy().reshape(1,borw*170)[0]
                                    to = (this_seg.iloc[30:200, 0:borw]).to_numpy().reshape(1,borw*170)[0]
                                perc = (len(to[to>border_cutoff])/len(to))*100
                                bord_ratio =((len(to[to>=border_cutoff])/len(to))) / ((len(ti[ti<border_cutoff])/len(ti)))
                                # in_out_rat =( (len(to[to>=240])/len(to))*100 ) / ( (len(ti[ti>=240])/len(ti)) + 1 )
                                in_out_rat =( (len(to[to>=border_cutoff])) + 1) / ( (len(to[to<border_cutoff])) + 1 )
    
                                tidf = pd.DataFrame(ti, columns=['ti'])
                                todf = pd.DataFrame(to, columns=['to'])                            
                                bins = np.linspace(100, 255, 30)
                                lgd = [ 'inner pixels', 'outer pixels']
                                ax2 = tidf.ti.hist(bins=bins,legend=False, histtype='step', color='blue') #, alpha=1, edgecolor='blue', linewidth=1)
                                todf.to.hist(bins=bins, ax=ax2, alpha=0.4, color='red') #, edgecolor='red', linewidth=1)
                                plt.legend(lgd)
                                plt.title(wbord + ' border width=' + str(borw) + ', percent ' + str((round(perc,2))) + '% pixels are '   + str(border_cutoff) + '-256, brat='  + str(round(bord_ratio,2)) + ', iorat=' + str(round(in_out_rat,2)) + '\n ' + this_file )
                                plt.xlabel('Pixel values 100-256')
                                plt.ylabel('Counts')
                                plt.show()
                            
                                # plt.plot( abs( (to.var(axis=0))) )
                                # plt.plot( abs(np.diff(to.var(axis=0))))
                                # plt.plot( abs(np.diff(ti.var(axis=1))))
                            
                            
                            top = (this_seg.iloc[pac_len-6:pac_len,30:200]).to_numpy().reshape(1,6*170)[0]
                            top_record.append( (len(top[top>240])/len(top))*100 )
                            
                            bottom = (this_seg.iloc[0:6,30:200]).to_numpy().reshape(1,6*170)[0]
                            bottom_record.append( (len(bottom[bottom>240])/len(bottom))*100 )
    
                            left = (this_seg.iloc[30:200,0:5]).to_numpy().reshape(1,5*170)[0]
                            left_record.append( (len(left[left>240])/len(left))*100 )
                            
                            right = (this_seg.iloc[30:200,224-5:224]).to_numpy().reshape(1,5*170)[0]
                            right_record.append( (len(right[right>240])/len(right))*100 )
                            
                            jpg_record.append( this_file )
                            
                            trimmed_image = img.crop(pac_image)
                            # trimmed_image.show()
                            grayImage = trimmed_image.convert('L')
                            # grayImage.show()
                            array = np.array(grayImage) 
                            this_seg = pd.DataFrame(array, columns=lbl_214, index=lbl_212)
                            pac_avg = pac_avg.add(this_seg, fill_value=0)
                        
                        # if (all(pac_avg==0)) | (f!=seg_fn[-1]) | (any(pac_avg>255)):
                        #     continue
                        # else:
                        pa = pac_avg.div(len(seg_fn))
                        pa = pa.round()
                        # CONVERT AVERAGE PAC MATRIX TO JPEG
                        pa = Image.fromarray(np.array(pa))
                        pa = pa.convert('L')
                        trg = targ_folder + base_fn + '.jpg'
                        pa.save(trg)
                        # print('Subj ' + this_subj + ' visit ' + str(vo))
                        
                    else:
                        print(' not enough EEG segments for ' + this_file)
                        missing_jpgs.append(this_file)

border_metrics = pd.DataFrame({'top':top_record, 'bottom':bottom_record, 'left':left_record, 'right':right_record, 'file':jpg_record})
    