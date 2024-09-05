# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 14:18:20 2024

@author: lifep
"""



import numpy as np
from PIL import Image
import coga_support_defs as csd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, mannwhitneyu

# MOVE PAC IMAGE FILES FROM SAME SUBJECT

import shutil
import random
import os

import datetime

print('START: ' + str( datetime.datetime.now()))

base_dir = 'D:\\COGA_eec\\' #  BIOWIZARD
do_random_segment = False
which_segment='t7'
which_dx = 'AUD' # AUD ALAB ALD
sex = '' # M F
min_age = 11
max_age = 17
race = ''
vmin = -7.5
vmax = 7.5
# flat_cut = 99999 # FLAT INTERVAL IN EEG SIGNAL (<1uV)
# noise_cut = 99999 # NOISE INTERVAL IN EEG SIGNAL (>100uV)
channel = 'FZ'
# flat_cut = 256
# noise_cut = 256
whichEEGfileExtention = 'jpg' # png jpg
which_pacdat = 'pacdat_MASTER.pkl'

# ~~~~~~~~~~~~~~~~

channelstr = channel.lower()
source_folder = 'new_pac_' + channelstr # eeg_figures | new_pac | new_pac_fz
targ_folder = 'pacstats_by_subj_' + channelstr + '_' + str(min_age) + '_' + str(max_age) + '_' + which_dx + '_' + sex


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
    # pd_filtered = pacdat[(pacdat.channel==channel) & (pacdat.age_this_visit>=min_age) & (pacdat.age_this_visit<=max_age) & (pacdat.perc_flat_slip0<=flat_cut) & (pacdat.perc_noise_slip0<=noise_cut)]
    pd_filtered = pacdat[(pacdat.channel==channel) & (pacdat.age_this_visit>=min_age) & (pacdat.age_this_visit<=max_age)]
    # pd_filtered = pacdat[(pacdat.channel==channel) & (pacdat.age_this_visit>=min_age) & (pacdat.age_this_visit<=max_age) & ((pacdat.max_flat<=flat_cut) | (pacdat.max_noise<=noise_cut)) & (pacdat.race==race)]
    sexlbl = 'both'

else:             
    pd_filtered = pacdat[(pacdat.channel==channel) & (pacdat.age_this_visit>=min_age) & (pacdat.age_this_visit<=max_age) & (pacdat.sex==sex)]
    # pd_filtered = pacdat[(pacdat.channel==channel) & (pacdat.age_this_visit>=min_age) & (pacdat.age_this_visit<=max_age) & (pacdat.sex==sex) & ((pacdat.max_flat<=flat_cut) | (pacdat.max_noise<=noise_cut))]
    # pd_filtered = pacdat[(pacdat.channel==channel) & (pacdat.age_this_visit>=min_age) & (pacdat.age_this_visit<=max_age) & (pacdat.sex==sex) & ((pacdat.max_flat<=flat_cut) | (pacdat.max_noise<=noise_cut)) & (pacdat.race==race)]
    sexlbl = sex
    
    

alc = pd_filtered[pd_filtered.AUD_this_visit==True]
ctl = pd_filtered[pd_filtered.AUD_this_visit==False]    
[unaf_df, alc_df] = csd.match_age1(ctl,alc)

# TEST OF csd.match_age1 THAT AGE DIFFS ARE NOT SIGNIFICANT
aa = alc_df.age_this_visit.values
bb = unaf_df.age_this_visit.values
pv = ttest_ind(aa,bb).pvalue
# pv = mannwhitneyu(aa,bb).pvalue


print('age differences, pval = ' + str(pv) + '\n')
# pd_filtered_age_balanced = pd.concat([alc.iloc[alc_df], ctl.iloc[unaf_df]]).reset_index(drop=True)
pd_filtered_age_balanced = pd.concat([alc_df, unaf_df]).reset_index(drop=True)

ages_alc = []
ages_ctl = []
jpg_subj = set([int(i) for i in set(file_info.ID)])
pd_subj =  set([int(i) for i in set(pd_filtered_age_balanced.ID)] )
overlap = jpg_subj.intersection(pd_subj)

print(targ_folder)
filtered_N = str(len(set(pd_filtered_age_balanced.ID)))
qualifying_N = str(len(overlap))
total_N = str(len(pd_filtered_age_balanced))

print('There are N = ' + qualifying_N + ' subjects represented in PAC heatmap dataset')
print('out of ' + filtered_N + ' total subjects in this dataset')
print('total # files = ' + total_N)
print('Excluding EEG signals with:')
print(' - Ages ' + str(min_age) + '-' + str(max_age) + '')
print(' - Sex: ' + sexlbl)

# CYCLE THROUGH EVERY SUBJECT REPRESENTED IN FILES FROM SOURCE FOLDER
all_subj_figs = pd.unique(file_info.ID) 
# all_subj_figs = (np.array(list(overlap)))
for i in range(0,len(all_subj_figs)):
    # i = 17
    this_subj = all_subj_figs[i]
    # this_subj = '10071029'
    
    # FIND ALL VISITS FOR A SUBJECT THEN FILTER BY AGE
    svisits = file_info[(file_info.ID==this_subj)]
    if len(svisits)>0:
        # print(this_subj)

        # WE WANT TO INCLUDE AUD DIAGNOSES IN FOLDER NAME FOR QUICK REF
        vinfo = pd_filtered_age_balanced[(pd_filtered_age_balanced.ID==int(this_subj))]
        if len(vinfo)>0:
            
            sv = list(set(svisits.this_visit.values))
            vi = list(set(vinfo.this_visit.values))
            for vs in sv:
                if vs not in vi:
                    svisits = svisits[svisits.this_visit!=vs]
                    
            if len(svisits)>0:
                svisits.reset_index(inplace=True, drop=True)


                if do_random_segment:
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
                            ages_alc += [vinfo[vinfo.this_visit==rand_row.this_visit].age_this_visit.values[0]]
                        else:
                            diag_folder = 'nonalcoholic'
                            ages_ctl += [vinfo[vinfo.this_visit==rand_row.this_visit].age_this_visit.values[0]]

                    elif which_dx=='ALD':
                        if vinfo[vinfo.this_visit==rand_row.this_visit].ALD_this_visit.values[0]:
                            diag_folder = 'alcoholic'
                        else:
                            diag_folder = 'nonalcoholic'

                else:
                    segnum = pd.DataFrame([x[-6:-4] for x in svisits.fn],columns=['segnum'])
                    svisits.insert(0,'segnum',segnum)
                    visit_row = svisits[svisits.segnum==which_segment]
                    if len(visit_row)>0:
                        this_file =  visit_row.fn.values[0]
                        if which_dx=='ALAB':
                            if vinfo[vinfo.this_visit==visit_row.this_visit].ALAB_this_visit.values[0]:
                                diag_folder = 'alcoholic'
                            else:
                                diag_folder = 'nonalcoholic'
                                
                        elif which_dx=='AUD':
                            if vinfo[vinfo.this_visit==visit_row.this_visit.values[0]].AUD_this_visit.values[0]:
                                diag_folder = 'alcoholic'
                                ages_alc += [vinfo[vinfo.this_visit==visit_row.this_visit.values[0]].age_this_visit.values[0]]
                            else:
                                diag_folder = 'nonalcoholic'
                                ages_ctl += [vinfo[vinfo.this_visit==visit_row.this_visit.values[0]].age_this_visit.values[0]]
                                
                        elif which_dx=='ALD':
                            if vinfo[vinfo.this_visit==visit_row.this_visit].ALD_this_visit.values[0]:
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

age_pval = (mannwhitneyu(ages_alc,ages_ctl).pvalue)
ap = f'{age_pval:.10f}'
print('differences in ages, p = ' + ap)
print('ages_alc, ' + str(np.mean(ages_alc)) + ' +/- ' + str(np.std(ages_alc)) + ' N=' + str(len(ages_alc)))
print('ages_ctl, ' + str(np.mean(ages_ctl)) + ' +/- ' + str(np.std(ages_ctl)) + ' N=' + str(len(ages_ctl)))

plt.hist(ages_alc)
plt.title('ages_alc ' + sexlbl + '_' + str(min_age) + '-' + str(max_age) + ' ' + which_segment)
plt.ylim([0,200])
plt.show()

plt.hist(ages_ctl)
plt.title('ages_ctl ' + sexlbl + '_' + str(min_age) + '-' + str(max_age) + ' ' + which_segment)
plt.ylim([0,200])
plt.show()

datasets = [targ_folder]


# channel = 'FZ' # PAC@FZ chanxHz
# datasets = [
#             'resnet_by_subj_f_0_20_AUD_%flat50_%noise1_',
#             'resnet_by_subj_f_20_30_AUD_%flat50_%noise1_',
#             'resnet_by_subj_f_30_40_AUD_%flat50_%noise1_',
#             'resnet_by_subj_f_40_75_AUD_%flat50_%noise1_',
#             'resnet_by_subj_f_0_99_AUD_%flat50_%noise1_',
#             'resnet_by_subj_f_0_99_AUD_%flat999_%noise999_'
#             ]
            
# channel = 'FC2' # PAC@FZ chanxHz FC2 O2 FC2O2 O2FC2
# datasets = [
#             'resnet_by_subj_a_25_40_AUD_%flat99999_%noise99999_',
#             'resnet_by_subj_a_25_40_AUD_%flat99999_%noise99999_M',
#             'resnet_by_subj_a_25_40_AUD_%flat99999_%noise99999_F',
#             'resnet_by_subj_a_FC2_25_40_AUD_%flat99999_%noise99999_F',
#             'resnet_by_subj_a_FC2_25_40_AUD_%flat99999_%noise99999_M'
#             ]


# channel = 'FC2O2' # PAC@FZ chanxHz
# datasets = [
#             'resnet_by_subj_a_fc2o2_25_40_AUD_%flat99999_%noise99999_F'
#             ]

whichEEGfileExtention = 'jpg'
# PATHS AND DATA INFO
base_dir = 'D:\\COGA_eec\\'
write_dir = 'C:\\Users\\lifep\\OneDrive - Downstate Medical Center\\PAC stats paper\\'


for targ_folder in datasets:

    pth = base_dir + targ_folder + '\\'
    fileListAll = csd.get_file_list(pth, whichEEGfileExtention)    
    fileList_alc = csd.get_file_list(pth + 'alcoholic\\', whichEEGfileExtention)
    alc = str(round((len(fileList_alc)/len(fileListAll))*100,1))
    
    xax = np.arange(0,13,(13/224))
    yax = np.arange(4,50,(46/224))
    
    freq_pha = [str(round(x,1)) for x in xax]
    freq_amp = [str(round(x,1)) for x in yax]
    freq_amp.reverse()

    pval_mx = np.zeros((224,224))
    effect_mx = np.zeros((224,224))
    
    # INPUT DATA AND LABELS 
    images = np.zeros((1,224,224))
    labels_dx = []
    print('collecting PAC into 3-D matrix')
    for dx in ['alcoholic', 'nonalcoholic']:
        file_list = csd.get_file_list(pth + dx + '\\', whichEEGfileExtention)
        for i in file_list:
            img = Image.open(i[0] + i[1]) 
            grayImage = img.convert('L')
            # grayImage.show()
            array = np.array(grayImage)        
            images = np.vstack((images,array[None]))
            # img_array = np.array(img)
            # img_array = preprocess_input(img_array)
            # images.append(array)
            labels_dx.append(dx)
    labels_dx = np.array(labels_dx)            
    labels_dx[labels_dx=='alcoholic'] = 1
    labels_dx[labels_dx=='nonalcoholic'] = 0
    labels_dx = labels_dx.astype(int)
    
    print('do statistics on PAC frequency pairs')
    for x in range(224):
        for y in range(224):
            # print(str(x) + ' ' + str(y))
            alc_i = np.where(labels_dx==1)
            unaff_i = np.where(labels_dx==0)
            alc_pac = images[alc_i,x,y]
            unaff_pac = images[unaff_i,x,y]
            stats = ttest_ind(alc_pac[0], unaff_pac[0], equal_var=False)
            pval_mx[x,y] = stats.pvalue
            effect_mx[x,y] = np.mean(alc_pac[0]) -  np.mean(unaff_pac[0])

    ttl = channel + ' ' + targ_folder + ' alc_' + str(len(ages_alc)) + ' unaff_' + str(len(ages_ctl)) + '_' + which_segment
    print('make and save figures \n')
    pv = pd.DataFrame(pval_mx,  columns=freq_pha, index=freq_amp)
    es = pd.DataFrame(effect_mx,  columns=freq_pha, index=freq_amp)

    # pv[pv>0.05] = 0.05
    hmmin = str(round(np.min(pv),9))
    hm = sns.heatmap(pv, vmax=0.05,cmap="rocket_r", cbar_kws={'label': 'p-value (min=' + hmmin + ')'})
    plt.title(ttl, fontsize = 9)
    plt.xlabel('Phase Frequency (Hz)', fontsize = 9) 
    plt.ylabel('Amplitude Frequency (Hz)', fontsize = 9)    
    output = plt.Axes.get_figure(hm)
    output.savefig(write_dir + ttl + '-PVALUES.jpg', bbox_inches='tight')
    plt.show()
    plt.close(output)
  
    hm = sns.heatmap(es, cmap="icefire", cbar_kws={'label': 'diff mean PAC strength (AUD - unaff)'}, vmin=vmin, vmax=vmax)
    plt.title(ttl, fontsize = 9)
    plt.xlabel('Phase Frequency (Hz)', fontsize = 9) 
    plt.ylabel('Amplitude Frequency (Hz)', fontsize = 9)
    output = plt.Axes.get_figure(hm)
    output.savefig(write_dir + ttl + '-EFFECTSIZE', bbox_inches='tight')
    plt.show()
    plt.close(output)
    
    pv[pv>0.05] = 0
    hm = sns.heatmap(es, cmap="icefire", cbar_kws={'label': 'diff mean PAC strength (AUD - unaff)'}, mask=(pv==0), vmin=vmin, vmax=vmax)
    plt.title(ttl, fontsize = 9)
    plt.xlabel('Phase Frequency (Hz)', fontsize = 9) 
    plt.ylabel('Amplitude Frequency (Hz)', fontsize = 9)
    output = plt.Axes.get_figure(hm)
    output.savefig(write_dir + ttl + '-EFFECTSIZExPVAL', bbox_inches='tight')
    plt.show()
    plt.close(output)


