# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 12:33:29 2024

@author: lifep
"""


import numpy as np
from PIL import Image
import coga_support_defs as csd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, mannwhitneyu, sem
import os
import datetime
# import shutil
# import random

# MOVE PAC IMAGE FILES FROM SAME SUBJECT
# SEX = [ '', 'M', 'F' ]
# age_groups = [[12,17],[18,23],[24,30],[31,40],[41,75]]
# severity_scores = [[6,11,'SEVERE'],[4,5,'MODERATE'],[2,3,'MILD'],[0,0,'NONE']]
# severity_scores = [[3,4,'MILD'],[5,6,'MODERATE'],[7,11,'SEVERE']]
# eeg_segments = ['t7', 't6', 't5', 't4', 't3', 't2', 't1', 't0']
# eeg_segments = ['t0', 't1', 't2', 't3', 't4', 't5', 't6', 't7']
# eeg_segments.reverse()

SEX = ['' , 'M' , 'F']
# age_groups = [[17,29],[30,39],[40,75]]
# age_groups = [[25,35],[35 ,45],[45,75]]
# age_groups = [[25,37],[38,50]]
age_groups = [[25,50]]

grps = [3,3,3]
shuf_seeds = [42,42]

# severity_scores = [[6,11,'MOD_SEV']]
severity_scores = [[6,11,'SEVERE']]
# severity_scores = [[6,11,'SEVERE_wthdrl_no_tol']]
yalc = [20,100]

# eeg_segments.reverse()
alpha = 0.05

do_pd_fn_df = True
do_pd_fab = True
# do_pd_fn_df = False
# do_pd_fab = False

base_dir = 'D:\\COGA_eec\\' #  BIOWIZARD
temp_dir = 'TEMP\\'
do_random_segment = False
which_dx = 'AUD' # AUD ALAB ALD
race = ''  
# vmin = -6
# vmax = 6

# low-theta-gamma
fpl = 3.02
fph = 3.37
fal = 37.68
fah = 48.36


# alpha-gamma2
# fpl = 8.98
# fph = 9.92
# fal = 40.76
# fah = 45.28

# high-theta-gamma
# fpl = 4.7
# fph = 5.75
# fal = 33.98
# fah = 40.76

# alpha-beta
# fpl = 7.83
# fph = 11.49
# fal = 18.17
# fah = 22.69


#  FREQUENCY VALUES FOR PHASE AND AMPLITUDE 
xax = np.arange(0,13,(13/224))
yax = np.arange(4,50,(46/224))

freq_pha = [str(round(x,2)) for x in xax]
freq_amp = [str(round(x,2)) for x in yax]


# GET INDICES FOR PHASE AND AMPLITUDE FREQUENCIES TO DO PAC REGION STATISTICS
fp = pd.DataFrame([float(f) for f in freq_pha], columns=['freq'])
fp_lo = fp[(fp.freq==fpl)].index[0]
fp_hi = fp[(fp.freq==fph)].index[0]
# WE REVERSE THIS SO THAT Y-AXIS IS PLOTTED CORRECTLY
freq_amp.reverse()
fa = pd.DataFrame([float(f) for f in freq_amp], columns=['freq'])
fa_lo = fa[(fa.freq==fal)].index[0]
fa_hi = fa[(fa.freq==fah)].index[0]

# # HELPS TO GET AVAILABLE FREQUENCIES
fa[(fa.freq>=49) & (fa.freq<=50)]
fp[(fp.freq>=3.1) & (fp.freq<=4)]


# es_region = es.iloc[fa_hi:fa_lo, fp_lo:fp_hi]
# es_region = es.iloc[fa_lo:fa_hi, fp_lo:fp_hi]
# vmin = -3
# vmax = 3
# hm = sns.heatmap(es_region, cmap="icefire", cbar_kws={'label': 'diff mean PAC strength (AUD - unaff)'}, vmin=vmin, vmax=vmax)
# plt.title(ttl, fontsize = 9)
# plt.xlabel('Phase Frequency (Hz)', fontsize = 9) 
# plt.ylabel('Amplitude Frequency (Hz)', fontsize = 9)
# output = plt.Axes.get_figure(hm)
# plt.show()




        
for sev in severity_scores:
# for sev in ["b'SEVERE'", "b'NONE'", "b'MODERAT'", "b'MILD'"]:
#     severlbl = sev.split("'")[1] 
    
    for age_rng in age_groups:
        
        if (grps[0]==2) | (grps[1]==2) | (grps[2]==2):
            vmin = -2
            vmax = 2
        elif (grps[0]==3) | (grps[1]==3) | (grps[2]==3):
            vmin = -3
            vmax = 3
        elif (grps[0]==4) | (grps[1]==4) | (grps[2]==4):
            vmin = -4
            vmax = 4
        elif (grps[0]==5) | (grps[1]==5) | (grps[2]==5):
            vmin = -5
            vmax = 5
        elif (grps[0]==6) | (grps[1]==6) | (grps[2]==6):
            vmin = -6
            vmax = 6
        
        for sex in SEX: 

                
            start_dt = datetime.datetime.now()
            fldrname = sex + ' by AVG segment\\' + str(age_rng[0]) + '-' + str(age_rng[1])
            min_age = age_rng[0]
            max_age = age_rng[1]

            # flat_cut = 99999 # FLAT INTERVAL IN EEG SIGNAL (<1uV)
            # noise_cut = 99999 # NOISE INTERVAL IN EEG SIGNAL (>100uV)
            channel = 'FZ'
            # flat_cut = 256
            # noise_cut = 256
            whichEEGfileExtention = 'jpg' # png jpg
            which_pacdat = 'pacdat_MASTER.pkl'
            
            # ~~~~~~~~~~~~~~~~
            
            write_dir = 'C:\\Users\\lifep\\OneDrive - Downstate Medical Center\\PAC stats paper\\' + fldrname + '\\'
            if not os.path.exists(write_dir):
                os.makedirs(write_dir) 
      
            channelstr = channel.lower()
            # source_folder = 'new_pac_' + channelstr + '_AVG' # eeg_figures | new_pac | new_pac_fz
            source_folder = 'new_pac_fz_AVG_0_3' 
            targ_folder = 'PAC_stats_' + channelstr + '_' + str(min_age) + '_' + str(max_age) + '_' + which_dx + '_' + sex + '_' + sev[2] + '_' + str(sev[0]) + '_' + str(sev[1])
            
            print('START: ' + str(start_dt))
            print(sev[2])
            print(' - Ages ' + str(min_age) + '-' + str(max_age) + '')
            
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
                        
            # segnum = pd.DataFrame([x[-6:-4] for x in file_info.fn],columns=['segnum'])
            # file_info.insert(0,'segnum',segnum)
            
            # GET MASTER TABLE OUT 
            pacdat = pd.read_pickle(base_dir + which_pacdat)            
            if len(sex)==0: 
                # pd_filtered = pacdat[(pacdat.channel==channel) & (pacdat.age_this_visit>=min_age) & (pacdat.age_this_visit<=max_age) & (pacdat.flat_score<=flat_cut) & (pacdat.noise_score<=noise_cut)]
                # pd_filtered = pacdat[(pacdat.channel==channel) & (pacdat.age_this_visit>=min_age) & (pacdat.age_this_visit<=max_age) & ((pacdat.perc_flat_slip1<=flat_cut) & (pacdat.max_noise<=noise_cut))]
                # pd_filtered = pacdat[(pacdat.channel==channel) & (pacdat.age_this_visit>=min_age) & (pacdat.age_this_visit<=max_age) & (pacdat.perc_flat_slip0<=flat_cut) & (pacdat.perc_noise_slip0<=noise_cut)]
                pd_filtered = pacdat[(pacdat.channel==channel) & (pacdat.age_this_visit>=min_age) & (pacdat.age_this_visit<=max_age)].copy()
                # pd_filtered = pacdat[(pacdat.channel==channel) & (pacdat.age_this_visit>=min_age) & (pacdat.age_this_visit<=max_age) & ((pacdat.max_flat<=flat_cut) | (pacdat.max_noise<=noise_cut)) & (pacdat.race==race)]
                sexlbl = 'both'
            
            else:             
                pd_filtered = pacdat[(pacdat.channel==channel) & (pacdat.age_this_visit>=min_age) & (pacdat.age_this_visit<=max_age) & (pacdat.sex_x==sex)].copy()
                # pd_filtered = pacdat[(pacdat.channel==channel) & (pacdat.age_this_visit>=min_age) & (pacdat.age_this_visit<=max_age) & (pacdat.sex==sex) & ((pacdat.max_flat<=flat_cut) | (pacdat.max_noise<=noise_cut))]
                # pd_filtered = pacdat[(pacdat.channel==channel) & (pacdat.age_this_visit>=min_age) & (pacdat.age_this_visit<=max_age) & (pacdat.sex==sex) & ((pacdat.max_flat<=flat_cut) | (pacdat.max_noise<=noise_cut)) & (pacdat.race==race)]
                sexlbl = sex
                
            print(' - Sex: ' + sexlbl + '\n')

            N_all = str(len(set(pacdat.ID)))
            print('Total N = ' + N_all)
            print('Men, N = ' + str(len(set(pacdat[pacdat.sex_x=='M'].ID))))
            print('Women, N = ' + str(len(set(pacdat[pacdat.sex_x=='F'].ID))) + '\n')
            
            if sexlbl!='both':
                aip = pacdat[(pacdat.AUD_this_visit==True) & (pacdat.sex_x==sexlbl)].ID
                uip = pacdat[(pacdat.AUD_this_visit==False) & (pacdat.sex_x==sexlbl)].ID            
                print(sexlbl + ' ' + which_dx +  ' in pacdat, N = ' + str(len(set(aip))) + '')
                print(sexlbl + ' unaffected in pacdat, N = ' + str(len(set(uip))))
                print('(longitudinal data)' + '\n')
            else:
                aip = pacdat[(pacdat.AUD_this_visit==True)].ID 
                uip = pacdat[(pacdat.AUD_this_visit==False)].ID            
                print(sexlbl + ' ' + which_dx +  ' in pacdat, N = ' + str(len(set(aip))) + '')
                print(sexlbl + ' unaffected in pacdat, N = ' + str(len(set(uip))))
                print('(longitudinal data)' + '\n')
            
            pd_filtered.reset_index(drop=True, inplace=True)   
            
            aa = pd_filtered[(pd_filtered.AUD_this_visit==True)].age_this_visit.values
            bb = pd_filtered[(pd_filtered.AUD_this_visit==False)].age_this_visit.values
            # pv = ttest_ind(aa,bb).pvalue
            pv = mannwhitneyu(aa,bb).pvalue
            print('age differences in pd_filtered, pval = ' + str(pv) + '\n')

            if sexlbl!='both':
                aipf = pd_filtered[(pd_filtered.AUD_this_visit==True) & (pd_filtered.sex_x==sexlbl)].ID
                uipf = pd_filtered[(pd_filtered.AUD_this_visit==False) & (pd_filtered.sex_x==sexlbl)].ID
                print(sexlbl + ' ' + which_dx +  ' in pd_filtered, N = ' + str(len(set(aipf))) + '')
                print(sexlbl + ' unaffected in pd_filtered, N = ' + str(len(set(uipf))) + '\n')
            else:
                aipf = pd_filtered[(pd_filtered.AUD_this_visit==True)].ID
                uipf = pd_filtered[(pd_filtered.AUD_this_visit==False)].ID
                print(sexlbl + ' ' + which_dx +  ' in pd_filtered, N = ' + str(len(set(aipf))) + '')
                print(sexlbl + ' unaffected in pd_filtered, N = ' + str(len(set(uipf))) + '\n')

            # BEFOER WE DO ANYTHING WE NEED TO GET ALL THE INFORMATION FOR EACH JPEG DATA FILE 
            # WE HAVE FROM pacdat AS IT MAKES DOWNSTREAM PROCESSES MUCH EASIER. 
            # OUTPUT OF THIS BLOCK IS A PANDAS DATAFRAME DERIVED FROM pacdat ENTRIES OF ALL EXISTING JPEG FILES 
            if do_pd_fn_df:
                print('building dataframe, matching JPG files to pacdat entries')
                pd_fn = []
                
                for i in range(0,len(file_info)):
                    fi = file_info.iloc[i].fn
                    this_fn = fi.split('.')[0]
                    this_entry = pd_filtered[(pd_filtered.ID==int(file_info.iloc[i].ID)) & (pd_filtered.this_visit==(file_info.iloc[i].this_visit))]
                    
                    # print('does not satisfy age, sex, etc. criteria in pd_filtered')
                    if len(this_entry)==0:
                        # pacdat[(pacdat.ID==int(file_info.iloc[i].ID)) & (pacdat.this_visit==(file_info.iloc[i].this_visit))]
                        continue
                    elif len(this_entry)==1:
                        pd_fn.append(this_entry.copy())
                    else:
                        pd_fn.append(this_entry[this_entry.eeg_file_name==this_fn].copy())    
                pd_fn_df = pd.DataFrame(pd_fn[0], columns=pd_fn[0].columns)
                for r in range(0,len(pd_fn)):
                    pd_fn_df = pd.concat([pd_fn_df, pd_fn[r]])
                pd_fn_df.to_pickle(base_dir + temp_dir + 'pd_fn' + '_' + sex + '_' + str(min_age) + '_' + str(max_age) + '_sev' + str(sev[0])+ '_' + str(sev[1]) + '_' + which_dx + '.pkl')
                # pd_fn_df.to_pickle(base_dir + temp_dir + 'pd_fn' + '_' + sex + '_' + str(min_age) + '_' + str(max_age) + '_' + which_dx + '.pkl')
                
            elif not(do_pd_fn_df):
                pd_fn_df =  pd.read_pickle(base_dir + temp_dir + 'pd_fn' + '_' + sex + '_' + str(min_age) + '_' + str(max_age) + '_sev' + str(sev[0])+ '_' + str(sev[1]) + '_' + which_dx + '.pkl')
                print('opening ' + 'pd_fn' + '_' + sex + '_' + str(min_age) + '_' + str(max_age) + '_' + which_dx + '.pkl')
                
                
            # WE REMOVE DUPLICATE FILE NAMES TO MAKE SURE THAT ANY GIVEN FILE CAN ONLY 
            pd_fn_df = pd_fn_df.drop_duplicates(subset=['eeg_file_name'])

            if sexlbl!='both':
                aipf = pd_fn_df[(pd_fn_df.AUD_this_visit==True) & (pd_fn_df.sex_x==sexlbl)].ID
                uipf = pd_fn_df[(pd_fn_df.AUD_this_visit==False) & (pd_fn_df.sex_x==sexlbl)].ID
                print(sexlbl + ' AUD in pd_fn_df, N = ' + str(len(set(aipf))) + '')
                print(sexlbl + ' unaffected in pd_fn_df, N = ' + str(len(set(uipf))) + '\n')                
            else:
                aipf = pd_fn_df[(pd_fn_df.AUD_this_visit==True)].ID
                uipf = pd_fn_df[(pd_fn_df.AUD_this_visit==False)].ID
                print(sexlbl + ' AUD in pd_fn_df, N = ' + str(len(set(aipf))) + '')
                print(sexlbl + ' unaffected in pd_fn_df, N = ' + str(len(set(uipf))) + '\n')


                    
            if do_pd_fab:

                criteria_lbls = ['Tolerance','Withdrawal','Increasing Use', 'Irresistable Craving','Alcohol seeking','Social neglect','Persists despite loss','fails roles','hazardous use','legal problems','persists despite pblems']
                # 1 Tolerance
                # 2 Withdrawal
                # 3 Alcohol is often used in larger amounts or over a longer period than was intended
                # 4 There is a persistent desire or unsuccessful efforts to cut down or control alcohol use
                # 5 A great deal of time is spent in activities necessary to obtain alcohol, use alcohol, or recover from its effects
                # 6 Important social, occupational, or recreational activities are given up or reduced because of alcohol use
                # 7 Alcohol use is continued despite knowledge of having a persistent or recurrent physical or psychological problem that is likely to have been caused or exacerbated by alcohol (e.g. continued drinking despite recognition that an ulcer was made worse by alcohol consumption)
                # (1) recurrent alcohol use resulting in a failure to fulfill major role obligations at work, school, or home (e.g., repeated absences or poor work performance related to alcohol use; alcohol-related absences, suspensions, or expulsions from school; neglect of children or household)
                # (2) recurrent alcohol use in situations in which it is physically hazardous (e.g., driving an automobile or operating a machine when impaired by alcohol use)
                # (3) recurrent alcohol-related legal problems (e.g., arrests for alcohol-related disorderly conduct)
                # (4) continued alcohol use despite having persistent or recurrent social or interpersonal problems caused or exacerbated by the effects of the alcohol (e.g., arguments with spouse about consequences of Intoxication, physical fights)
                
                lowage = str(int(min(pd_fn_df[pd_fn_df.AUD_this_visit==True].age_this_visit)))
                hiage = str(int(max(pd_fn_df[pd_fn_df.AUD_this_visit==True].age_this_visit)))
                print('age matching from ' + lowage + ' to ' + hiage + ' years old')                
                pd_fn_df = pd_fn_df.drop_duplicates(subset=['eeg_file_name'])
                pd_fn_df.reset_index(drop=True, inplace=True)   


                # MAKING THIS MORE EXPLICIT SO THAT THERE ARE NO DUPLICATES OF 
                # SUBJECTS AND NUMBER OF AUD IS MAXIMIZED
                # aaa = set(pd_fn_df[(pd_fn_df.AUD_this_visit==True) & (pd_fn_df.ald5sx_cnt>=sev[0]) & (pd_fn_df.ald5sx_cnt<=sev[1]) & (pd_fn_df.alc_dep_sx4==5)].ID)
                aaa = set(pd_fn_df[(pd_fn_df.AUD_this_visit==True) & (pd_fn_df.ald5sx_cnt>=sev[0]) & (pd_fn_df.ald5sx_cnt<=sev[1])].ID)
                # aaa = set(pd_fn_df[(pd_fn_df.AUD_this_visit==True) & (pd_fn_df.alc_dep_sx1!=5) & (pd_fn_df.alc_dep_sx2==5)].ID)
                # aaa = set(pd_fn_df[(pd_fn_df.AUD_this_visit==True) & (pd_fn_df.alc_dep_sx1==5) & (pd_fn_df.alc_dep_sx2=!=5)].ID)
                # aaa = set(pd_fn_df[(pd_fn_df.AUD_this_visit==True) & (pd_fn_df.alc_dep_sx1==5) & (pd_fn_df.alc_dep_sx2==5)].ID)
                # aaa = set(pd_fn_df[(pd_fn_df.AUD_this_visit==True) & (pd_fn_df.alc_dep_sx1!=5) & (pd_fn_df.alc_dep_sx2!=5)].ID)
                # aaa = set(pd_fn_df[(pd_fn_df.AUD_this_visit==True) & (pd_fn_df.alc_dep_sx4==5) ].ID)
                # aaa = set(pd_fn_df[(pd_fn_df.AUD_this_visit==True) & (pd_fn_df.alc_dep_sx2==5) ].ID)
                # aaa = set(pd_fn_df[(pd_fn_df.AUD_this_visit==True) & (pd_fn_df.years_alc>=yalc[0]) & (pd_fn_df.years_alc<=yalc[1]) ].ID)
                
                aaa_lbl = ''
                

                uuu = set(pd_fn_df[(pd_fn_df.AUD_this_visit==False)].ID)
                
                alcs = list( aaa.difference(uuu) )
                boths = list( aaa.intersection(uuu) )
                uuu = set(pd_fn_df[(pd_fn_df.AUD_this_visit==False) & (pd_fn_df.ald5sx_cnt.isnull())].ID)
                unafs = list( uuu.difference(aaa) )


                both_l = []
                for b in boths: 
                    ss = pd_fn_df[(pd_fn_df.ID==b) & (pd_fn_df.AUD_this_visit==True)]
                    for i in range(0,len(ss)):
                        both_l.append( ss.iloc[i].copy() )   
                
                alc_l = []
                for a in alcs: 
                    ss = pd_fn_df[(pd_fn_df.ID==a) & (pd_fn_df.AUD_this_visit==True)]
                    for i in range(0,len(ss)):
                        alc_l.append( ss.iloc[i].copy() ) 
                        
                alc = pd.concat([pd.DataFrame(alc_l), pd.DataFrame(both_l)]).reset_index(drop=True)  
                
                unaf_l = []
                for u in unafs: 
                    ss = pd_fn_df[(pd_fn_df.ID==u)]
                    for i in range(0,len(ss)):
                        unaf_l.append( ss.iloc[i].copy() ) 
                ctl = pd.DataFrame(unaf_l)
                
                # IMPORTANT ~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # ADD A SAMPLING TO THE NEXT LINE FOR WHICH 
                # DUPLICATES ARE SAVED WHICH COULD INCLUDE BY FAMILY
                alc.drop_duplicates(subset=['ID'],inplace=True, keep='last')
                
                [unaf_df, alc_df] = csd.match_age2(ctl,alc,shuf_seeds, targ_folder)
                # TEST OF csd.match_age1 THAT AGE DIFFS ARE NOT SIGNIFICANT
                aa = alc_df.age_this_visit.values
                bb = unaf_df.age_this_visit.values
                # pv = ttest_ind(aa,bb).pvalue
                pv = mannwhitneyu(aa,bb).pvalue
                print(which_dx +  ' , N = ' + str(len(alc_df)) + '')
                print('unaffected, N = ' + str(len(unaf_df)) + '')
                print('age differences, pval = ' + str(pv) + '\n')
                
                # pd_filtered_age_balanced = pd.concat([alc.iloc[alc_df], ctl.iloc[unaf_df]]).reset_index(drop=True)
                pd_filtered_age_balanced = pd.concat([alc_df, unaf_df]).reset_index(drop=True)
                pd_filtered_age_balanced.to_pickle(base_dir + temp_dir + 'pd_filtered_age_balanced' + '_' + sex + '_' + str(min_age) + '_' + str(max_age) + '_sev' + str(sev[0])+ '_' + str(sev[1]) + '_' + which_dx + '.pkl')

                s1 = pd_filtered_age_balanced.columns.get_loc('alc_dep_sx1')
                e1 = pd_filtered_age_balanced.columns.get_loc('alc_dep_sx7')
                s2 = pd_filtered_age_balanced.columns.get_loc('alc_abuse_sx1')
                e2 = pd_filtered_age_balanced.columns.get_loc('alc_abuse_sx4')
                # FOR SOME WEIRD REASON, I NEED TO ADD 1 TO THE INDEX FOR e1 AND e2 BUT NOT FOR s1 OR s2
                df1 = pd_filtered_age_balanced.iloc[:,s1:e1+1]
                df2 = pd_filtered_age_balanced.iloc[:,s2:e2+1]
                df3 = pd.merge(df1,df2,left_index=True, right_index=True)
                ss = []
                for i in range(0,11): ss.append( ((np.sum(df3.iloc[:,i]==5))/(len(df3)/2))*100 )
                
                fig, ax = plt.subplots()
                ax.bar(criteria_lbls,ss)
                plt.xticks(rotation=80, fontsize=14)
                plt.title(targ_folder + aaa_lbl)
                plt.ylabel('% subjects with criterion')
                plt.ylim([0,100])
                plt.show()
                


            elif not(do_pd_fab):
                pd_filtered_age_balanced =  pd.read_pickle(base_dir + temp_dir + 'pd_filtered_age_balanced' + '_' + sex + '_' + str(min_age) + '_' + str(max_age) + '_sev' + str(sev[0])+ '_' + str(sev[1]) + '_' + which_dx + '.pkl')
                print('opening ' + 'pd_filtered_age_balanced' + '_' + sex + '_' + str(min_age) + '_' + str(max_age) + '_' + which_dx + '.pkl')

                
            if sexlbl!='both':
                aipfab = pd_filtered_age_balanced[(pd_filtered_age_balanced.AUD_this_visit==True) & (pd_filtered_age_balanced.sex_x==sexlbl)].ID
                uipfab = pd_filtered_age_balanced[(pd_filtered_age_balanced.AUD_this_visit==False) & (pd_filtered_age_balanced.sex_x==sexlbl)].ID
                print(which_dx + ' in pd_filtered_age_balanced, N = ' + str(len(set(aipfab))) + '')
                print(sexlbl + ' unaffected in pd_filtered_age_balanced, N = ' + str(len(set(uipfab))) + '\n')
            else:
                aipfab = pd_filtered_age_balanced[(pd_filtered_age_balanced.AUD_this_visit==True)].ID
                uipfab = pd_filtered_age_balanced[(pd_filtered_age_balanced.AUD_this_visit==False)].ID
                print(which_dx + ' in pd_filtered_age_balanced, N = ' + str(len(set(aipfab))) + '')
                print(sexlbl + ' unaffected in pd_filtered_age_balanced, N = ' + str(len(set(uipfab))) + '\n')

            
            # RUN ORIGINAL CODE FOR AGE MATCHING
            if 0:
                alc = pd_filtered_age_balanced[pd_filtered_age_balanced.AUD_this_visit==True]
                ctl = pd_filtered[pd_filtered.AUD_this_visit==False]    
                # [unaf_df, alc_df] = csd.match_age1(ctl,alc)

                # TEST OF csd.match_age1 THAT AGE DIFFS ARE NOT SIGNIFICANT
                aa = alc_df.age_this_visit.values
                bb = unaf_df.age_this_visit.values
                pv = ttest_ind(aa,bb).pvalue
                pv = mannwhitneyu(aa,bb).pvalue


                print('age differences, pval = ' + str(pv) + '\n')
            #     # pd_filtered_age_balanced = pd.concat([alc.iloc[alc_df], ctl.iloc[unaf_df]]).reset_index(drop=True)
            #     pd_filtered_age_balanced = pd.concat([alc_df, unaf_df]).reset_index(drop=True)
                
            # alc = pd_filtered_age_balanced[(pd_filtered_age_balanced.AUD_this_visit==True) ]
            # ctl = pd_filtered_age_balanced[pd_filtered_age_balanced.AUD_this_visit==False]    
            # [unaf_df, alc_df] = csd.match_age1(ctl,alc)
            # pd_filtered_age_balanced = pd.concat([alc_df, unaf_df])
            # pd_filtered_age_balanced.sort_values(by=['AUD_this_visit'], ascending=False, inplace=True)
            pd_filtered_age_balanced.reset_index(drop=True)
            
            ages_alc = []
            ages_ctl = []
            

            ages_alc = pd_filtered_age_balanced[pd_filtered_age_balanced.AUD_this_visit==True].age_this_visit.values
            ages_ctl = pd_filtered_age_balanced[pd_filtered_age_balanced.AUD_this_visit==False].age_this_visit.values

            age_pval = (mannwhitneyu(ages_alc,ages_ctl).pvalue)
            ap = f'{age_pval:.3f}'
            print('differences in ages after removing replicate subjects, pval = ' + ap)
            print('ages_alc, ' + str(np.mean(ages_alc)) + ' +/- ' + str(np.std(ages_alc)) + ' N=' + str(len(ages_alc)))
            print('ages_ctl, ' + str(np.mean(ages_ctl)) + ' +/- ' + str(np.std(ages_ctl)) + ' N=' + str(len(ages_ctl)))
            
            # plt.hist(ages_alc)
            # plt.title('ages_alc ' + sexlbl + '_' + str(min_age) + '-' + str(max_age) + ' N=' + str(len(ages_alc)))
            # plt.ylim([0,250])
            # plt.show()
            
            # plt.hist(ages_ctl)
            # plt.title('ages_ctl ' + sexlbl + '_' + str(min_age) + '-' + str(max_age) + ' N=' + str(len(ages_ctl)))
            # plt.ylim([0,250])
            # plt.show()
            
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
            
            
            for targ_folder in datasets:
            
                pth = base_dir + 'new_pac_fz_AVG' + '\\'
                
                # fileListAll = csd.get_file_list(pth, whichEEGfileExtention)    
                # fileList_alc = csd.get_file_list(pth + 'alcoholic\\', whichEEGfileExtention)
                # alc = str(round((len(fileList_alc)/len(fileListAll))*100,1))
                
            
                pval_mx = np.zeros((224,224))
                effect_mx = np.zeros((224,224))
                
                # INPUT DATA AND LABELS 
                images = np.zeros((1,224,224))
                labels_dx = []
                missing = 0 
                print('\ncollecting PAC into 3-D matrix')
                for s in range(0,len(pd_filtered_age_balanced)-1): 
                    fname = pd_filtered_age_balanced.iloc[s].eeg_file_name + '.jpg'
                    if os.path.isfile(pth + fname):

                        # print(fname)
                        if pd_filtered_age_balanced.iloc[s].AUD_this_visit==True:
                            dx = 'alcoholic'
                            labels_dx.append(1)
                        else:
                            dx = 'nonalcoholic'
                            labels_dx.append(0)

                        img = Image.open(pth + fname)
                        grayImage = img.convert('L')
                        # grayImage.show()
                        array = np.array(grayImage)        
                        images = np.vstack((images,array[None]))
                    else:
                        missing += 1
                        print(fname + ' missing, ' + str(missing))

                labels_dx = np.array(labels_dx)
                labels_dx = labels_dx.astype(int)
                # FINALLY WE REMOVE THE zeros STARTING IMAGE SO THAT 
                # WE'RE NOT INCLUDING THE zeros 2D SLICE IN DOWNSTREAM STATISTICAL ANALYSES
                images = np.delete(images,0,axis=0)
                
                
                if 1:
                    print('doing statistics on all PAC frequency pairs')
                    for x in range(224):
                        for y in range(224):
                            # print(str(x) + ' ' + str(y))
                            alc_i = np.where(labels_dx==1)[0]
                            unaff_i = np.where(labels_dx==0)[0]
                            alc_pac = images[alc_i,x,y]
                            unaff_pac = images[unaff_i,x,y]
                            # stats = ttest_ind(alc_pac[0], unaff_pac[0], equal_var=False)
                            stats = mannwhitneyu(alc_pac, unaff_pac)
                            pval_mx[x,y] = stats.pvalue
                            effect_mx[x,y] = np.mean(alc_pac) -  np.mean(unaff_pac)
                            
                    if int(float(ap))==1: 
                        ap='001'
                        
                    ttl = targ_folder + '_alc_' + str(len(ages_alc)) + '_unaff_' + str(len(ages_ctl))  + '_agep' + ap[2:] + '_a' + str(alpha)[2:]
                    # ttl = channel + ' ' + targ_folder + ' alc_' + str(len(ages_alc)) + ' unaff_' + str(len(ages_ctl)) + '_' + which_segment
                    print('make and save figures \n')
                    pv = pd.DataFrame(pval_mx,  columns=freq_pha, index=freq_amp)
                    es = pd.DataFrame(effect_mx,  columns=freq_pha, index=freq_amp)
                    # es = es.iloc[fp_lo:fp_hi,fa_lo:fa_hi]
                    # vmin = -3
                    # vmax = 3
                    # hm = sns.heatmap(es, cmap="icefire", cbar_kws={'label': 'diff mean PAC strength (AUD - unaff)'}, vmin=vmin, vmax=vmax)
                    # plt.title(ttl, fontsize = 9)
                    # plt.xlabel('Phase Frequency (Hz)', fontsize = 9) 
                    # plt.ylabel('Amplitude Frequency (Hz)', fontsize = 9)
                    # output = plt.Axes.get_figure(hm)
                    # plt.show()
                    
                    
                    pv[pv>0.05] = 0.05
                    hmmin = str(round(np.min(pv),6))
                    hm = sns.heatmap(pv, vmax=0.05,cmap="rocket_r", cbar_kws={'label': 'p-value (min=' + hmmin + ')'})
                    plt.title(ttl, fontsize = 9)
                    plt.xlabel('Phase Frequency (Hz)', fontsize = 9) 
                    plt.ylabel('Amplitude Frequency (Hz)', fontsize = 9)    
                    output = plt.Axes.get_figure(hm)
                    
                    if not os.path.exists(write_dir):
                        os.makedirs(write_dir) 
                        
                    output.savefig(write_dir + ttl + '-PVALUES.jpg', bbox_inches='tight')
                    plt.show()
                    plt.close(output)

                    hm = sns.heatmap(es, cmap="icefire", cbar_kws={'label': 'diff mean PAC strength (AUD - unaff)'}, vmin=vmin, vmax=vmax)
                    plt.title(ttl, fontsize = 9)
                    plt.xlabel('Phase Frequency (Hz)', fontsize = 9) 
                    plt.ylabel('Amplitude Frequency (Hz)', fontsize = 9)
                    output = plt.Axes.get_figure(hm)
                    plt.show()
                    output.savefig(write_dir + ttl + '_EFFECTSIZE', bbox_inches='tight')
                    plt.close(output)
                    
                    pv[pv>=alpha] = 0
                    hm = sns.heatmap(es, cmap="icefire", cbar_kws={'label': 'diff mean PAC strength (AUD - unaff)'}, mask=(pv==0), vmin=vmin, vmax=vmax)
                    plt.title(ttl, fontsize = 9)
                    plt.xlabel('Phase Frequency (Hz)', fontsize = 9) 
                    plt.ylabel('Amplitude Frequency (Hz)', fontsize = 9)
                    output = plt.Axes.get_figure(hm)
                    output.savefig(write_dir + ttl + '-EFFECTSIZExPVAL', bbox_inches='tight')
                    plt.show()
                    plt.close(output)
                
                
                if 1:
                    print('averaging PAC regions for statistical analysis')

                            
                    # for x in range(224):
                    #     for y in range(224):
                    #         # print(str(x) + ' ' + str(y))
                    #         alc_i = np.where(labels_dx==1)
                    #         unaff_i = np.where(labels_dx==0)
                    #         alc_pac = images[alc_i,x,y]
                    #         unaff_pac = images[unaff_i,x,y]
                    #         stats = ttest_ind(alc_pac[0], unaff_pac[0], equal_var=False)
                    #         pval_mx[x,y] = stats.pvalue
                    #         effect_mx[x,y] = np.mean(alc_pac[0]) -  np.mean(unaff_pac[0])
                
                    # ttl = channel + ' ' + targ_folder + ' alc_' + str(len(ages_alc)) + ' unaff_' + str(len(ages_ctl)) + '_' + which_segment
                    ttl = targ_folder + ' alc_' + str(len(ages_alc)) + ' unaff_' + str(len(ages_ctl))  + '_agep' + ap[2:]

                    print('make and save figures \n')
                    # plt.bar(['alc','unaff'],[np.mean(alc_pac),np.mean(unaff_pac)])


                    # effect_mx = np.mean(alc_pac[0]) -  np.mean(unaff_pac[0])
                    # es = pd.DataFrame(effect_mx,  columns=freq_pha, index=freq_amp)
                    es_region = es.iloc[fa_hi:fa_lo, fp_lo:fp_hi]
                    reglbl = '_fa_' + str(fa_hi) + '_' + str(fa_lo) + '_fp_' + str(fp_lo) + '_' + str(fp_hi)                   
                    
                    regions = [0]*len(images)
                    for thispac in range(len(images)):
                        regions[thispac] = np.mean(images[thispac,fa_hi:fa_lo, fp_lo:fp_hi])

                    regions = np.array(regions)
                    print('doing statistics on PAC frequency pair region')

                    alc_i = np.where(labels_dx==1)
                    unaff_i = np.where(labels_dx==0)
                    alc_pac = regions[alc_i]
                    unaff_pac = regions[unaff_i]
                    
                                        
                    # stats = ttest_ind(alc_pac, unaff_pac, equal_var=False)
                    stats = mannwhitneyu(alc_pac, unaff_pac)
                    
                    # if stats.pvalue<=0.05:
                    if stats.pvalue<=1:
                        hm = sns.heatmap(es_region, cmap="icefire", cbar_kws={'label': 'diff mean PAC strength (AUD - unaff)'}, vmin=vmin, vmax=vmax)
                        plt.title(ttl, fontsize = 9)
                        plt.xlabel('Phase Frequency (Hz)', fontsize = 9) 
                        plt.ylabel('Amplitude Frequency (Hz)', fontsize = 9)
                        output = plt.Axes.get_figure(hm)
                        plt.show()
                        output.savefig(write_dir + ttl + '-EFFECTSIZE_reg' + reglbl, bbox_inches='tight')
                        plt.close(output)
                    
                        plt.title(ttl + '\np = ' + str(stats.pvalue), fontsize = 9)
    
                        plt.errorbar(['alc','unaff'],[np.mean(alc_pac),np.mean(unaff_pac)],yerr=[sem(alc_pac),sem(unaff_pac)])
                        plt.show()
                        plt.savefig(write_dir + ttl + '-EFFECTSIZE_reg_bar' + reglbl, bbox_inches='tight')
                        plt.close()


                    

                    
                    print('p = ' + str(stats.pvalue))
                    
                    elapsed_dt = datetime.datetime.now() - start_dt
                    print('END: ' + str(datetime.datetime.now()))
                    print('elapsed: ' + str(elapsed_dt) + '\n')
    
    
        
    
