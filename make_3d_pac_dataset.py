# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 11:16:06 2025

@author: lifep
"""
 

import numpy as np
from PIL import Image
import coga_support_defs as csd
import pandas as pd
import os
import datetime


multvis = True
SEX = [ '' ]
sexlbl = 'both'
age_groups = [[1,99]]
severity_scores = [[0,11,'ALL']]
img_rows = 212
img_cols = 214
do_pd_fn_df = False

source_folder = 'new_pac_fz_AVG_0_3_NOBORDER'
imgInfo = '_0_3_NOBORDER'
base_dir = 'D:\\COGA_eec\\' 
temp_dir = 'TEMP\\'
which_dx = 'AUD' # AUD ALAB ALD

channel = 'FZ'
whichEEGfileExtention = 'jpg' # png jpg
which_pacdat = 'pacdat_MASTER.pkl'
        

# CONSTANTS    
chan_i = 0 
visit_i = 3 
id_i = 4  
            
            
for sev in severity_scores:

    
    for age_rng in age_groups:
        
        
        for sex in SEX: 
   
            start_dt = datetime.datetime.now()
            min_age = age_rng[0]
            max_age = age_rng[1]

            channelstr = channel.lower()
            targ_folder = '' + channelstr + '_' + str(min_age) + '_' + str(max_age) + '_' + which_dx + '_' + sex + '_' + sev[2] + '_' + str(sev[0]) + '_' + str(sev[1])
            
            print('START: ' + str(start_dt))
            print(sev[2])
            print(' - Ages ' + str(min_age) + '-' + str(max_age) + '')
            

            # BEFOER WE DO ANYTHING WE NEED TO GET ALL THE INFORMATION FOR EACH JPEG DATA FILE 
            # WE HAVE FROM pacdat AS IT MAKES DOWNSTREAM PROCESSES MUCH EASIER. 
            # OUTPUT OF THIS BLOCK IS A PANDAS DATAFRAME DERIVED FROM pacdat ENTRIES OF ALL EXISTING JPEG FILES 
            if do_pd_fn_df:
                
                
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
                file_info.sort_values(by=['ID'],inplace=True) 
                file_info.reset_index(drop=True)
                
                # GET MASTER TABLE OUT 
                pacdat = pd.read_pickle(base_dir + which_pacdat)            
                # DO THIS TO GET EVERYTHING OUT THAT IS IN pacdat E.G. FOR LINEAR MIXED EFFECT MODELING USING ENTIRE DATASET
                pd_filtered = pacdat[(pacdat.channel==channel)].copy()

                    
                print(' - Sex: ' + sexlbl + '\n')

                N_all = str(len(set(pacdat.ID)))
                print('Total N = ' + N_all)
                print('Men, N = ' + str(len(set(pacdat[pacdat.sex_x=='M'].ID))))
                print('Women, N = ' + str(len(set(pacdat[pacdat.sex_x=='F'].ID))) + '\n')
                
                aip = pacdat[(pacdat.AUD_this_visit==True) & (pacdat.sex_x==sexlbl)].ID
                uip = pacdat[(pacdat.AUD_this_visit==False) & (pacdat.sex_x==sexlbl)].ID            
                print(sexlbl + ' ' + which_dx +  ' in pacdat, N = ' + str(len(set(aip))) + '')
                print(sexlbl + ' unaffected in pacdat, N = ' + str(len(set(uip))))
                print('(longitudinal data)' + '\n')

                
                pd_filtered.reset_index(drop=True, inplace=True)   
                

                aipf = pd_filtered[(pd_filtered.AUD_this_visit==True) & (pd_filtered.sex_x==sexlbl)].ID
                uipf = pd_filtered[(pd_filtered.AUD_this_visit==False) & (pd_filtered.sex_x==sexlbl)].ID
                print(sexlbl + ' ' + which_dx +  ' in pd_filtered, N = ' + str(len(set(aipf))) + '')
                print(sexlbl + ' unaffected in pd_filtered, N = ' + str(len(set(uipf))) + '\n')

                
                print('building dataframe, matching JPG files to pacdat entries')
                pd_fn = []
                
                for i in range(0,len(file_info)):
                    fi = file_info.iloc[i].fn
                    this_fn = fi.split('.')[0]
                    this_entry = pd_filtered[(pd_filtered.eeg_file_name==this_fn)]
                    
                    if len(this_entry)==0:
                        print('missing ' + this_fn)
                        # pacdat[(pacdat.ID==int(file_info.iloc[i].ID)) & (pacdat.this_visit==(file_info.iloc[i].this_visit))]
                        continue
                    elif len(this_entry)==1:
                        pd_fn.append(this_entry.copy())
                    elif (len(this_entry)>1) & (multvis==True):
                        for j in range(0,len(this_entry)): 
                            pd_fn.append(this_entry[this_entry.index==this_entry.index[j]].copy())
                    else:
                        pd_fn.append(this_entry[this_entry.eeg_file_name==this_fn].copy())    
                pd_fn_df = pd.DataFrame(pd_fn[0], columns=pd_fn[0].columns)
                
                
                for r in range(1,len(pd_fn)):
                    pd_fn_df = pd.concat([pd_fn_df, pd_fn[r]])
                    
                pd_fn_df.to_pickle(base_dir + temp_dir + 'pd_fn' + '_' + sex + '_' + str(min_age) + '_' + str(max_age) + '_sev' + str(sev[0])+ '_' + str(sev[1]) + '_' + which_dx + '.pkl')
                
            elif not(do_pd_fn_df):
                pd_fn_df =  pd.read_pickle(base_dir + temp_dir + 'pd_fn' + '_' + sex + '_' + str(min_age) + '_' + str(max_age) + '_sev' + str(sev[0])+ '_' + str(sev[1]) + '_' + which_dx + '.pkl')
                print('opening ' + 'pd_fn' + '_' + sex + '_' + str(min_age) + '_' + str(max_age) + '_' + which_dx + '.pkl')
                
                
            # WE REMOVE DUPLICATE FILE NAMES TO MAKE SURE THAT ANY GIVEN FILE CAN ONLY 
            pd_fn_df = pd_fn_df.drop_duplicates(subset=['eeg_file_name'])

            aipf = pd_fn_df[(pd_fn_df.AUD_this_visit==True) & (pd_fn_df.sex_x==sexlbl)].ID
            uipf = pd_fn_df[(pd_fn_df.AUD_this_visit==False) & (pd_fn_df.sex_x==sexlbl)].ID
            print(sexlbl + ' AUD in pd_fn_df, N = ' + str(len(set(aipf))) + '')
            print(sexlbl + ' unaffected in pd_fn_df, N = ' + str(len(set(uipf))) + '\n')                


            pd_filtered_age_balanced = pd_fn_df
            pd_filtered_age_balanced.sort_values(by=['ID','age_this_visit'],inplace=True) 
            pd_filtered_age_balanced.reset_index(drop=True,inplace=True) 
            
            datasets = [targ_folder]

            
            whichEEGfileExtention = 'jpg'
            # PATHS AND DATA INFO
            base_dir = 'D:\\COGA_eec\\'
            
            
            for targ_folder in datasets:
            
                pth = base_dir + source_folder + '\\'

                
                # INPUT DATA AND LABELS 
                images = np.zeros((1,img_rows,img_cols))
                labels_dx = []
                missing = 0 
                print('\ncollecting PAC into 3-D matrix')
                
                
                for s in range(0,len(pd_filtered_age_balanced)-1): 
                    fname = pd_filtered_age_balanced.iloc[s].eeg_file_name + '.jpg'
                    if os.path.isfile(pth + fname):
                        age = pd_filtered_age_balanced.iloc[s].age_this_visit
                        this_id = pd_filtered_age_balanced.iloc[s].ID
                        family = str(this_id)[0:4]
                        yalc = pd_filtered_age_balanced.iloc[s].years_alc
                        evd = pd_filtered_age_balanced.iloc[s].ever_drink
                        afd = pd_filtered_age_balanced.iloc[s].age_first_got_drunk_x
                        vst = pd_filtered_age_balanced.iloc[s].this_visit
                        sex = pd_filtered_age_balanced.iloc[s].sex_x
                        ald5cnt = pd_filtered_age_balanced.iloc[s].ald5sx_cnt
                        if np.isnan(ald5cnt): ald5cnt = 0
                        
                        
                        # print(fname)
                        if pd_filtered_age_balanced.iloc[s].AUD_this_visit==True:
                            dx = 'alcoholic'
                            labels_dx.append({'ID' : this_id, 'age': age, 'yalc' : yalc, 'evd' : evd, 'afd' : afd, 'visit' : vst, 'audcnt' : ald5cnt, 'sex' : sex, 'AUD' : 1, 'family': family})
                        else:
                            dx = 'nonalcoholic'
                            labels_dx.append({'ID' : this_id, 'age': age, 'yalc' : yalc, 'evd' : evd, 'afd' : afd, 'visit' : vst, 'audcnt' : ald5cnt, 'sex' : sex, 'AUD' : 0, 'family': family})

                        img = Image.open(pth + fname)
                        # grayImage = img.convert('L')
                        # grayImage.show()
                        array = np.array(img)        
                        images = np.vstack((images,array[None]))
                    else:
                        missing += 1
                        print(fname + ' missing, ' + str(missing))

                # FINALLY WE REMOVE THE zeros STARTING IMAGE SO THAT 
                # WE'RE NOT INCLUDING THE zeros 2D SLICE IN DOWNSTREAM STATISTICAL ANALYSES
                images = np.delete(images,0,axis=0)
                np.save(base_dir + 'pac_3d_' + targ_folder + imgInfo + '.npy', images)
                
                # NOW WE SAVE THE INFORMATION DATAFRAME 
                pac_age = pd.DataFrame(labels_dx)
                sbjs = list(set(pac_age.ID))
                pac_age['visit_cnt'] = np.ones(len(pac_age))
                for s in sbjs:
                    idx = pac_age[pac_age.ID==s].index
                    vnum = len(pac_age[pac_age.ID==s])
                    pac_age.loc[idx,'visit_cnt'] = np.ones(len(idx))*vnum
                pac_age.to_pickle(base_dir + 'pac_age_' + targ_folder + imgInfo + '.pkl')
                
                
                # START: 2025-01-21 15:12:30.738204
                # ALL
                #  - Ages 1-99
                # opening pd_fn__1_99_AUD.pkl
                # both AUD in pd_fn_df, N = 0
                # both unaffected in pd_fn_df, N = 0


                # collecting PAC into 3-D matrix
                # FZ_eec_3_c1_20047079_32_cnt_500.jpg missing, 1
                # FZ_eec_4_b1_20130061_32_cnt_500.jpg missing, 2
                # FZ_eec_4_c1_30095021_32_cnt_500.jpg missing, 3
                # FZ_eec_3_c1_30164028_32_cnt_500.jpg missing, 4
                # FZ_eec_2_a1_40001205_cnt_512.jpg missing, 5
                # FZ_eec_3_c1_40150003_32_cnt_500.jpg missing, 6
                # FZ_eec_4_c1_49383003_32_cnt_500.jpg missing, 7
                # FZ_eec_4_e1_50138134_32_cnt_500.jpg missing, 8
                # FZ_eec_1_a1_61174009_cnt_256.jpg missing, 9
                # FZ_eec_4_g1_62219003_32_cnt_500.jpg missing, 10
                
                
                
                
                
                
                
                
                
    
    
        
    
