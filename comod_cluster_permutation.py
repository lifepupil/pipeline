# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 15:04:19 2025

@author: lifep
"""


import numpy as np
from PIL import Image
import coga_support_defs as csd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
from scipy.stats import ttest_ind, mannwhitneyu, sem, linregress
import os
from age_matcher import AgeMatcher
import matplotlib.patches as patches
from mne.stats import permutation_cluster_test
import mne
import numpy.ma as ma


SEX = ' F '.upper().strip()
channel = 'FZ'
age_groups = [25,50]
RACE = 'WHITE'.upper() # 'ASIAN', 'BLACK', 'OTHER', 'UNK.', 'WHITE'
HISPANIC = False
vmin = -3
vmax = 3
shuf_seeds = 424242
use_sev_scrs, severity_scores = True, [6,11,'SEVERE']
use_symptm, symptom =           False, 'ad2'
alpha = 0.05
aud_symp = 'ald5sx_max_cnt' # audcnt ald5sx_max_cnt

load_age_match = False 
do_age_match = False
do_all = True 
# load_age_match = False 
# do_age_match = True
# do_all = False 

# img_rows = 212
# img_cols = 214
img_rows = 224
img_cols = 224
data_info = '_new_pac_fz_AVG_0_3' # _0_3_NOBORDER _new_pac_fz_AVG_0_3
image_format = 'png'
pac_len = 224
# TO REMOVE BORDER SET BELOW TO NON-ZERO INTEGER VALUES CORRESPONDING 
# TO NUMBER OF ROWS OR COLUMNS TO REMOVE FROM MATRIX
border_tb = 6
border_rl = 5

load_new_dat = False # False True
info_fn = 'pac_age_fz_1_99_AUD__ALL_0_11' + data_info + '.pkl'
pac_fn = 'pac_3d_fz_1_99_AUD__ALL_0_11' + data_info + '.npy'
# base_dir = 'D:\\COGA_eec\\' 
base_dir = 'C:\\Users\\lifep\\OneDrive\\Documents\\'

fldrname = 'COMODULOGRAMS'


criteria_lbls = ['Tolerance',
                 'Withdrawal',
                 'Increasing Use', 
                 'Irresistable Craving',
                 'Alcohol seeking',
                 'Social neglect',
                 'Persists despite loss',
                 'fails roles','hazardous use',
                 'legal problems',
                 'persists despite pblems']

sympt_lbls = {'ad1' : 'Tolerance\t\t\t\t',
            'ad2' : 'Withdrawal\t\t\t\t',
            'ad3' : 'Increasing Use\t\t\t',
            'ad4' : 'Irresistable Craving\t',
            'ad5' : 'Alcohol seeking\t\t\t',
            'ad6' : 'Social neglect\t\t\t',
            'ad7' : 'Persists despite loss\t',
            'aa1' : 'fails roles\t\t\t\t',
            'aa2' : 'hazardous use\t\t\t',
            'aa3' : 'legal problems\t\t\t',
            'aa4' : 'persists despite pblems\t'
            }
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


# ~~~~~~~~~~~ BODY ~~~~~~~~~~~~~~

write_dir = 'C:\\Users\\lifep\\OneDrive - Downstate Medical Center\\PAC stats paper\\' + fldrname + '\\'
if not os.path.exists(write_dir):
    os.makedirs(write_dir) 
                
#  FREQUENCY VALUES FOR PHASE AND AMPLITUDE 
xax = np.arange(0,13,(13/img_cols))
yax = np.arange(4,50,(46/img_rows))

freq_pha = [str(round(x,2)) for x in xax]
# freq_amp = [str(round(x,2)) for x in yax]
freq_amp = [str(round(x,2)) for x in np.sort(yax)[::-1]]

min_age = age_groups[0]
max_age = age_groups[1]

channelstr = channel.upper()
if HISPANIC:
    hlbl = '+HISP'
else:
    hlbl = ''
targ_folder = '' + channelstr + '_' + str(min_age) + '_' + str(max_age) + '_' + RACE + hlbl + '_' + SEX + '_' + severity_scores[2] + '_' + str(severity_scores[0]) + '_' + str(severity_scores[1])

# MAKE SURE THAT THE INDEXING IS IDENTICAL BETWEEN pac_all AND images_all 
# THEY MUST ALSO HAVE THE SAME LENGTH, E.G., 8781
if ( not(['images_all' in locals()][0]) ) | load_new_dat:
    images_all = np.load(base_dir + pac_fn)
    pac_all = pd.read_pickle(base_dir + info_fn)
    
    
    
# TO ADD COLUMNS TO pac_all
if 0: 
    newcollbl = 'ald5sx_max_cnt'
    pacdatcol = 'ald5sx_max_cnt'
    which_pacdat = 'D:\\COGA_eec\\pacdat_MASTER.pkl' 
    pacdat = pd.read_pickle(which_pacdat)
    newcol = []
    pacdat = pacdat[(pacdat.channel=='FZ')]
    for i in range(0,len(pac_all)):
        tmp = pacdat[(pacdat.ID==pac_all.iloc[i].ID) & (pacdat.age_this_visit==pac_all.iloc[i].age)]
        if len(tmp)>0:
            newcol.append(tmp[pacdatcol].values[0])
        else: 
            print('~~~~~~~~~~~~~~~~~~~~  missing ' + str(pac_all.iloc[i].ID) + ' age ' + + str(pac_all.iloc[i].age))
    pac_all[newcollbl] = newcol
    pac_all.to_pickle(base_dir + info_fn)





# INITIAL FILTERING BY DEMOGRAPHICS
pac_all = pac_all[(pac_all.age>=min_age) & (pac_all.age<=max_age)]      

if not(RACE==''):
    pac_all = pac_all[(pac_all.race==RACE)]

if not(HISPANIC):
    pac_all = pac_all[(pac_all.hisp==False)]    

if not(SEX==''):
    pac_all = pac_all[(pac_all.sex==SEX)]  


# AGE MATCHING
if (load_age_match):
    matched_cases = pd.read_pickle('matched_cases_' + SEX + '.pkl')
    matched_controls = pd.read_pickle('matched_controls_' + SEX + '.pkl')
    pac_age = pd.concat([matched_cases,matched_controls])
    ttl = targ_folder + ' alc_' + str(len(matched_cases)) + ' unaff_' + str(len(matched_controls)) 

elif (do_age_match) & (not('matched_cases' in locals())):
    aaa = set(pac_all[(pac_all.AUD==True) & (pac_all[aud_symp]>=severity_scores[0]) & (pac_all[aud_symp]<=severity_scores[1])].ID)
    uuu = set(pac_all[(pac_all.AUD==False)].ID)
    # SINCE THERE ARE MULIPLE VISITS THE SAME SUBJECT CAN SHOW UP AS AUD OR NOT DEPENDING ON HOW YOUNG THEY WERE ON THEIR FIRST VISIT TO MAXIMIZE AUD SAMPLE SIZE 
    alcs = list( aaa.difference(uuu) )
    boths = list( aaa.intersection(uuu) )
    # NOW WE MAKE SURE THAT CONTROLS HAVE NEVER BEEN DIAGNOSED WITH AUD TO EXCLUDE SUBJECTS THAT GO FROM UNAFFECTED TO AUD 
    uuu = set(pac_all[(pac_all.AUD==False) & (pac_all[aud_symp]==0)].ID)
    unafs = list( uuu.difference(aaa) )
    
    both_l = []
    for b in boths: 
        ss = pac_all[(pac_all.ID==b) & (pac_all.AUD==True)]
        ss = ss[(ss.age==max(ss.age))]
        for i in range(0,len(ss)):
            both_l.append( ss.iloc[i].copy() )   
    alc_l = []
    for a in alcs: 
        ss = pac_all[(pac_all.ID==a) & (pac_all.AUD==True)]
        ss = ss[(ss.age==max(ss.age))]
        for i in range(0,len(ss)):
            alc_l.append( ss.iloc[i].copy() ) 
    alc = pd.concat([pd.DataFrame(alc_l), pd.DataFrame(both_l)])
    
    unaf_l = []
    for u in unafs: 
        ss = pac_all[(pac_all.ID==u)]
        ss = ss[(ss.age==max(ss.age))]
        for i in range(0,len(ss)):
            unaf_l.append( ss.iloc[i].copy() ) 
    ctl = pd.DataFrame(unaf_l)
    
    # matcher = AgeMatcher(age_tol=5, age_col='age', sex_col='sex', strategy='sricter', shuffle_df=True, random_state=42)
    if not(['matcher' in locals()][0]):
        print('\nmatch instance being made \n')
        matcher = AgeMatcher(age_tol=5, age_col='age', sex_col='sex', strategy='greedy', shuffle_df=True, random_state=shuf_seeds)
    matched_cases, matched_controls = matcher(alc, ctl)
    matched_cases.to_pickle('matched_cases_' + SEX + '.pkl')
    matched_controls.to_pickle('matched_controls_' + SEX + '.pkl')
    pac_age = pd.concat([matched_cases,matched_controls])
    # pac_age.to_pickle('PAC' + SEX + '.pkl')
    # pm = pd.read_pickle('PACM.pkl')
    # pf = pd.read_pickle('PACF.pkl')
    # pacmf = pd.concat([pm,pf])
    # del pacmf['PAC']
    # del pacmf['id']
    # pacmf.to_pickle('PAC_participant_info.pkl')
    ttl = targ_folder + ' alc_' + str(len(matched_cases)) + ' unaff_' + str(len(matched_controls)) 

elif do_all:
    
    if use_sev_scrs==True:
        pac_age = pac_all[(pac_all.ald5sx_max_cnt>=severity_scores[0]) | (pac_all.ald5sx_max_cnt==0)]
        
    if use_symptm==True:    
        pac_age = pac_all[(pac_all[symptom]==5) | (pac_all.ald5sx_max_cnt==0)]
        
        
    # alcid = list(set(pac_age[pac_age.AUD==True].ID))
    # for aid in alcid:
    #     tmpdf = pac_age[pac_age.ID==aid]
    #     if len(tmpdf)>1:
    #         pac_age[pac_age.ID==aid]
        


#  TO DO - TAKE ADVANTAGE OF THE MULTIPLE VISITS FOR REPEATED MEASURES TO ANSWER:
#   - HOW STABLE IS WITHIN-VISIT PAC? 
#   - HOW STABLE IS PAC BETWEEN VISITS (CHECK FOR AUD DIAGNOSIS X INTERACTIONS )
# pac_all.visit_cnt.hist(bins=len(set(pac_all.visit_cnt)))
# plt.title('Histogram of # visits for COGA participants')
# plt.xlabel('Number of visits in dataset')
# plt.ylabel('N = ')
# plt.show()
# plt.close()



    ttl = targ_folder + ' alc_' + str(len(pac_age[pac_age.AUD==True])) + ' unaff_' + str(len(pac_age[pac_age.AUD==False])) 


match_i = pac_age.index

if 1:
    which_pacdat = 'D:\\COGA_eec\\pacdat_MASTER.pkl' 
    pacdat = pd.read_pickle(which_pacdat)
    df_list = []
    if SEX=='':
        pacdat = pacdat[(pacdat.channel=='FZ')]
    else:
        pacdat = pacdat[(pacdat.channel=='FZ') & (pacdat.sex_x==SEX)]
    for i in range(0,len(pac_age)):
        tmp = pacdat[(pacdat.ID==pac_age.iloc[i].ID) & (pacdat.age_this_visit==pac_age.iloc[i].age)].copy()
        if len(tmp)>0:
            df_list.append(tmp)
        else: 
            print('~~~~~~~~~~~~~~~~~~~~  missing ' + str(pac_age.iloc[i].ID) + ' age ' + + str(pac_age.iloc[i].age))
    pac_df = pd.concat(df_list)
    pac_df.rename(columns={'sex_x' : 'sex'}, inplace=True)
    pac_df.rename(columns={'race_x' : 'race'}, inplace=True)
    csd.print_demo_vals(pac_df)
    
images = images_all[match_i]
alc_i = np.where(pac_age.AUD==1)
unaff_i = np.where(pac_age.AUD==0)
acmd = images[alc_i]
ccmd = images[unaff_i]


# AGE AND AUD SEVERITY HISTOGRAMS
pac_age['ald5sx_max_cnt'].hist()
plt.gca().set_yscale('log')
plt.title(ttl)
plt.xlabel('AUD symptom count')
plt.ylabel('counts')
plt.show()
plt.close()

pac_age['age'].hist()
plt.title(ttl)
plt.xlabel('age')
plt.ylabel('counts')
plt.show()
plt.close()

excl = np.ones((224,224), dtype=bool)
excl[border_tb:pac_len-border_tb,border_rl:pac_len-border_rl] = False
n_perm = 1000
tail = 1
t_obs, clusters, cluster_p_values, h0 = permutation_cluster_test(
    [acmd, ccmd], 
    n_permutations=n_perm, 
    exclude=excl,
    tail=tail,
    verbose=True)
tobs = ma.masked_array(t_obs,mask=excl)
im = plt.imshow(tobs, cmap='jet', interpolation='nearest')
plt.title(ttl)
plt.colorbar(im, label='PAC difference')
plt.show()

t_obs_plot = np.nan * np.ones_like(t_obs)
for c, p_val in zip(clusters, cluster_p_values):
    if p_val <= cluster_p_values.min():
        t_obs_plot[c] = t_obs[c]
        # t_obs[c] = np.nan
im = plt.imshow(t_obs_plot, cmap='jet', interpolation='nearest')
plt.title(ttl + '\n' + 'min p-value = ' + str(cluster_p_values.min()))
plt.colorbar(im, label='PAC difference')
plt.show()




# EXTRACTING SPECTRAL POWER INFO FROM pacdat
if 0:
    which_pacdat = 'D:\\COGA_eec\\pacdat_MASTER.pkl' 
    pacdat = pd.read_pickle(which_pacdat)
    pacdat = pacdat[(pacdat.channel==channel) & (pacdat.age_this_visit>=min_age) & (pacdat.age_this_visit<=max_age) & (pacdat.sex_x==SEX)]
    df_list = []
    print('\n ')
    for s in range(len(pac_age)):
        sbj = pac_age.iloc[s]
        df_list.append(pacdat[(pacdat.ID==sbj.ID) & (pacdat.age_this_visit==sbj.age)].copy())
    pac_df = pd.concat(df_list)
    freq_bands = ['delta', 'theta','alpha','low_beta','high_beta','gamma']
    for fd in freq_bands:
        aa = pac_df[pac_df.ald5dx==1][[fd]][fd].to_list()
        uu = pac_df[pac_df.ald5dx==5][[fd]][fd].to_list()
        mw = mannwhitneyu(aa,uu)
        diff = np.mean(aa) - np.mean(uu)
        print(fd + ' ' + str(mw.pvalue)[:6] + '      AUD - unaff = ' + str(np.mean(diff)))


if 0:
    pval_mx = np.zeros((img_rows,img_cols))
    effect_mx = np.zeros((img_rows,img_cols))
    aud_mx = np.zeros((224,224))
    # aud_sem_mx = np.zeros((224,224))
    ctl_mx = np.zeros((224,224))
    # ctl_sem_mx = np.zeros((224,224))
    print('\ndoing statistics on all PAC frequency pairs')
    # images = np.load('C:\\Users\\lifep\\OneDrive\\Documents\\pac_3d_' + targ_folder + '.npy')
    for x in range(img_rows):
        for y in range(img_cols):
            # print(str(x) + ' ' + str(y))
            alc_i = np.where(pac_age.AUD==1)
            unaff_i = np.where(pac_age.AUD==0)
            alc_pac = images[alc_i,x,y][0]
            unaff_pac = images[unaff_i,x,y][0]
            # stats = ttest_ind(alc_pac[0], unaff_pac[0], equal_var=False)
            stats = mannwhitneyu(alc_pac, unaff_pac)
            pval_mx[x,y] = stats.pvalue
            effect_mx[x,y] = np.mean(alc_pac) -  np.mean(unaff_pac)
            aud_mx[x,y] = np.mean(alc_pac)
            # aud_sem_mx[x,y] = np.std(alc_pac)/np.sqrt(len(alc_pac))
            ctl_mx[x,y] = np.mean(unaff_pac)
            # ctl_sem_mx[x,y] = np.std(unaff_pac)/np.sqrt(len(unaff_pac))
          
    # TRIM BORDER
    # RESIZED X AND Y AXIS LABELS
    xax = np.arange(0,13,(13/(pac_len-border_rl*2)))
    yax = np.arange(4,50,(46/(pac_len-border_tb*2)))
    freq_pha = [str(round(x,2)) for x in xax]
    freq_pha.pop(0)
    freq_pha.append('13')
    freq_amp = [str(round(x,2)) for x in np.sort(yax)[::-1]]
    p2 = pval_mx[border_tb:pac_len-border_tb,border_rl:pac_len-border_rl].copy()
    pv = pd.DataFrame(p2,  columns=freq_pha, index=freq_amp)
    e2 = effect_mx[border_tb:pac_len-border_tb,border_rl:pac_len-border_rl].copy()
    es = pd.DataFrame(e2,  columns=freq_pha, index=freq_amp)
    aud2 = aud_mx[border_tb:pac_len-border_tb,border_rl:pac_len-border_rl].copy()
    audcomod = pd.DataFrame(aud2,  columns=freq_pha, index=freq_amp)
    unaf2 = ctl_mx[border_tb:pac_len-border_tb,border_rl:pac_len-border_rl].copy()
    unafcomod = pd.DataFrame(unaf2,  columns=freq_pha, index=freq_amp)
    
    # HERE WE ARE BUILDING THE AXIS TICK VALUES FOR COMODULOGRAMS
    interval_numy = 7
    interval_numx = 7
    nn = np.array( es)
    lenx = np.shape(nn)[1]
    leny = np.shape(nn)[0]
    yy = np.arange(4,50,((50-4)/leny))
    ytks = np.array([str(np.round(y,1)) for y in yy])
    intvly = round((leny)/interval_numy)
    iya = np.arange(0, leny,intvly)
    ytv = ytks[iya]  
    iy = np.arange(leny,0,-intvly) # NEED TO USE A REVERSED INDEX LIST BECAUSE OF HJOW sms HEATMAP STORES VALUES 
    # iy = np.array(iya.tolist()[::-1]) # NEED TO USE A REVERSED INDEX LIST BECAUSE OF HJOW sms HEATMAP STORES VALUES 
    print('lengths: ytv= ' + str(len(ytv)) + ', iy= ' + str(len(iy)))
    xx = np.arange(0,13,((13-0)/lenx))
    xx = np.append(xx,13) 
    xtks = np.array([str(np.round(x,1)) for x in xx])
    intvlx = round((lenx)/interval_numx)
    ix = np.arange(0,lenx,intvlx)
    xtv = xtks[ix]  
    ix = np.append(ix,lenx)
    xtv = np.append(xtv,13)
    print('lengths: xtv= ' + str(len(xtv)) + ', ix= ' + str(len(ix)))
    
    hm = sns.heatmap(es, 
                     cmap="jet", 
                     cbar_kws={'label': '\nPAC strength change\n(AUD - Unaff)'}, 
                     vmin=vmin, vmax=vmax,
                     xticklabels=False,
                     yticklabels=False
                     )
    hm.set_xlabel('Phase Frequency (Hz)', fontsize=18)
    hm.set_ylabel('Amplitude Frequency (Hz)\n', fontsize=18)
    hm.set_xticks(ix, xtv)
    hm.set_yticks(iy, ytv)
    hm.tick_params(direction='out', labelsize=16, length=4, width=1, colors='k', bottom=True, left=True)
    hm.collections[0].colorbar.ax.tick_params(labelsize=16)
    hm.figure.axes[-1].yaxis.label.set_size(18)
    ax = plt.gca()
    for _,spine in ax.spines.items():
        spine.set_visible(True)
    plt.show()
    output = plt.Axes.get_figure(hm)
    output.savefig(write_dir + ttl + '_EFFECTSIZE.png', bbox_inches='tight', format=image_format)
    plt.close(output)
    
    
    pv_fig = pv.copy()
    pv_fig[pv_fig>alpha] = alpha
    hmmin = str(round(np.min(pv_fig),6))
    pvf = sns.heatmap(pv_fig, 
                     vmax=0.05,
                     cmap="rocket_r", 
                     cbar_kws={'label': 'p-value\n(min=' + hmmin + ')'},
                     xticklabels=False,
                     yticklabels=False
                     )
    pvf.set_xlabel('Phase Frequency (Hz)', fontsize=18)
    pvf.set_ylabel('Amplitude Frequency (Hz)\n', fontsize=18)
    pvf.set_xticks(ix, xtv)
    pvf.set_yticks(iy, ytv)
    pvf.tick_params(direction='out', labelsize=16, length=4, width=1, colors='k', bottom=True, left=True)
    pvf.collections[0].colorbar.ax.tick_params(labelsize=16)
    pvf.figure.axes[-1].yaxis.label.set_size(18)
    ax = plt.gca()
    for _,spine in ax.spines.items():
        spine.set_visible(True)
    output = plt.Axes.get_figure(pvf)
    plt.show()
    output.savefig(write_dir + ttl + '-PVALUES.png', bbox_inches='tight', format=image_format)
    plt.close(output)
    
    # vmin = -5
    # vmax = 5
    pv_fig = pv.copy()
    pv_fig[pv_fig>=alpha] = 0
    hmm = sns.heatmap(es,
                      cmap="jet", 
                      cbar_kws={'label': 'PAC strength change\n(AUD - unaff)'}, 
                      mask=(pv_fig==0), 
                      vmin=vmin, vmax=vmax,
                      xticklabels=False,
                      yticklabels=False
                      )
    hmm.set_xlabel('Phase Frequency (Hz)', fontsize=18)
    hmm.set_ylabel('Amplitude Frequency (Hz)\n', fontsize=18)
    hmm.set_xticks(ix, xtv)
    hmm.set_yticks(iy, ytv)
    hmm.tick_params(direction='out', labelsize=16, length=4, width=1, colors='k', bottom=True, left=True)
    hmm.collections[0].colorbar.ax.tick_params(labelsize=16)
    hmm.figure.axes[-1].yaxis.label.set_size(18)
    ax = plt.gca()
    for _,spine in ax.spines.items():
        spine.set_visible(True)
    plt.show()
    output = plt.Axes.get_figure(hmm)
    output.savefig(write_dir + ttl + '-EFFECTSIZExPVAL.png', bbox_inches='tight', format=image_format)
    plt.close(output)
    
    vmn = 186
    vmx = 194
    vs = str(vmn) + '-' + str(vmx)
    audmap = sns.heatmap(audcomod, 
                     cmap="jet", 
                     cbar_kws={'label': '\nAverage PAC strength'}, 
                     # vmin=180, vmax=195,
                     vmin=vmn, vmax=vmx,
                     xticklabels=False,
                     yticklabels=False
                     )
    audmap.set_xlabel('Phase Frequency (Hz)', fontsize=18)
    audmap.set_ylabel('Amplitude Frequency (Hz)\n', fontsize=18)
    audmap.set_title(SEX + ' - AUD, ' + vs, pad=12)
    audmap.set_xticks(ix, xtv)
    audmap.set_yticks(iy, ytv)
    audmap.tick_params(direction='out', labelsize=16, length=4, width=1, colors='k', bottom=True, left=True)
    audmap.collections[0].colorbar.ax.tick_params(labelsize=16)
    audmap.figure.axes[-1].yaxis.label.set_size(18)
    ax = plt.gca()
    for _,spine in ax.spines.items():
        spine.set_visible(True)
    plt.show()
    output = plt.Axes.get_figure(audmap)
    output.savefig(write_dir + ttl + '_AUD_comod.png', bbox_inches='tight', format=image_format)
    plt.close(output)
    
    unafmap = sns.heatmap(unafcomod, 
                     cmap="jet", 
                     cbar_kws={'label': '\nAverage PAC strength'}, 
                     # vmin=180, vmax=195,
                     vmin=vmn, vmax=vmx,
                     xticklabels=False,
                     yticklabels=False
                     )
    unafmap.set_xlabel('Phase Frequency (Hz)', fontsize=18)
    unafmap.set_ylabel('Amplitude Frequency (Hz)\n', fontsize=18)
    unafmap.set_title(SEX + ' - UNAFFECTED, ' + vs, pad=12)
    unafmap.set_xticks(ix, xtv)
    unafmap.set_yticks(iy, ytv)
    unafmap.tick_params(direction='out', labelsize=16, length=4, width=1, colors='k', bottom=True, left=True)
    unafmap.collections[0].colorbar.ax.tick_params(labelsize=16)
    unafmap.figure.axes[-1].yaxis.label.set_size(18)
    ax = plt.gca()
    for _,spine in ax.spines.items():
        spine.set_visible(True)
    plt.show()
    output = plt.Axes.get_figure(unafmap)
    output.savefig(write_dir + ttl + '_CTL_comod.png', bbox_inches='tight', format=image_format)
    plt.close(output)
    
        # REGIONAL PAC
        # M
        # fpl = 9
        # fph = 12
        # fal = 35
        # fah = 45
        
        # fpl = 4
        # fph = 7
        # fal = 33
        # fah = 40
        
        # fpl = 8
        # fph = 12
        # fal = 18
        # fah = 22
        
        # fpl = 3.5
        # fph = 5
        # fal = 11
        # fah = 14
        
        # F
        # fpl = 10
        # fph = 13
        # fal = 30
        # fah = 45
        
        # fpl = 2.4
        # fph = 3.6
        # fal = 26
        # fah = 41
        
        # fpl = 7
        # fph = 11
        # fal = 14
        # fah = 17
        
        # fpl = 3
        # fph = 4.5
        # fal = 8
        # fah = 15
    
    if 1:
        if SEX=='M':
            PAS = [[9,12,30,45],[4,7,33,40],[8,12,18,22],[3.5,5,11,14]]
        else:
            PAS = [[10,13,30,45],[2.4,3.6,26,41],[7,11,14,17],[3,4.5,8,15]]
    
        for pa in PAS:    
            fpl = pa[0]
            fph = pa[1]
            fal = pa[2]
            fah = pa[3]
            
            # GET INDICES FOR PHASE AND AMPLITUDE FREQUENCIES TO DO PAC REGION STATISTICS
            fp = pd.DataFrame([float(f) for f in freq_pha], columns=['freq'])
            fp_lo = fp[(fp.freq>=fpl)].index[0]
            fp_hi = fp[(fp.freq>=fph)].index[0]
            fa = pd.DataFrame([float(f) for f in freq_amp], columns=['freq'])
            fa_lo = fa[(fa.freq<=fal)].index[0]
            fa_hi = fa[(fa.freq<=fah)].index[0]
            fal_lbl = str(fal).replace('.','_') 
            fah_lbl = str(fah).replace('.','_')
            fpl_lbl =  str(fpl).replace('.','_') 
            fph_lbl = str(fph).replace('.','_')
            reglbl = '_fp_' + fpl_lbl + '__' + fph_lbl +'_fa_' + fal_lbl + '__' + fah_lbl
            es_region = es.iloc[fa_hi:fa_lo, fp_lo:fp_hi]
            pv_region = pv.iloc[fa_hi:fa_lo, fp_lo:fp_hi]
            
            regions = [0]*len(images)
            for thispac in range(len(images)):
                regions[thispac] = np.mean(images[thispac,fa_hi:fa_lo, fp_lo:fp_hi])
            regions = np.array(regions)
                
                
            print('doing statistics on PAC frequency pair region')
            alc_i = np.where(pac_age.AUD==1)
            unaff_i = np.where(pac_age.AUD==0)
            alc_pac = regions[alc_i]
            unaff_pac = regions[unaff_i]      
        
            if ('PAC' in pac_age):
                print('\nUPDATING PAC COLUMN IN pac_age')
                del pac_age['PAC']
                pac_age.insert(0, 'PAC', regions)
                # plt.scatter(pac_age.iloc[alc_i].audcnt,pac_age.iloc[alc_i].PAC)
                plt.scatter(pac_age.PAC, pac_age[aud_symp])
                plt.title(ttl + reglbl)
                plt.show()
            else:
                pac_age.insert(0, 'PAC', regions)
                # plt.scatter(pac_age.iloc[alc_i].audcnt,pac_age.iloc[alc_i].PAC)
                plt.scatter(pac_age.PAC, pac_age[aud_symp])
                plt.title(ttl + reglbl)
                plt.show()
                     
            # stats = ttest_ind(alc_pac, unaff_pac, equal_var=False)
            stats = mannwhitneyu(alc_pac, unaff_pac)
            symptoms = ['ad1',
                        'ad2',
                        'ad3',
                        'ad4',
                        'ad5',
                        'ad6',
                        'ad7',
                        'aa1',
                        'aa2',
                        'aa3',
                        'aa4'
                        ]
            
            print('\nDSM-4 symptoms has or has not in AUD')
            for smp in symptoms:
                pac_age[smp] = pac_age[smp].fillna(1)
                alc_has_i = np.where((pac_age.AUD==1) & (pac_age[smp]==5))
                alc_hasnt_i = np.where((pac_age.AUD==1) & (pac_age[smp]==1))
                symptom_stats = mannwhitneyu(regions[alc_has_i],regions[alc_hasnt_i])
                print(sympt_lbls[smp] + 'p = ' + str(round(symptom_stats.pvalue,4)))
                
            if stats.pvalue<=1:
                nn = np.array( es_region)
                lenx = np.shape(nn)[1]
                leny = np.shape(nn)[0]
                
                yy = np.arange(fal,fah,((fah-fal)/np.shape(nn)[0]))
                ytks = np.array([str(np.round(y,1)) for y in yy])
                intvly = round((np.shape(nn)[0])/7)
                # iy = np.arange(0, np.shape(nn)[0],intvly)
                iya = np.arange(0, leny,intvly)
                ytv = ytks[iya]    
                iy = np.arange(leny,0,-intvly) # NEED TO USE A REVERSED INDEX LIST BECAUSE OF HJOW sms HEATMAP STORES VALUES
                # iy = np.array(iya.tolist()[::-1]) # NEED TO USE A REVERSED INDEX LIST BECAUSE OF HJOW sms HEATMAP STORES VALUES 
            
                xx = np.arange(fpl,fph,((fph-fpl)/lenx))
                xtks = np.array([str(np.round(x,1)) for x in xx])
                intvlx = round((np.shape(nn)[1])/7)
                ix = np.arange(0,lenx,intvlx)
                xtv = xtks[ix]
                hmr = sns.heatmap(es_region, 
                                 cmap="icefire", 
                                 cbar_kws={'label': 'PAC strength change\n(AUD - unaff)'}, 
                                 vmin=vmin, vmax=vmax, 
                                 xticklabels=False, 
                                 yticklabels=False
                                 )
                hmr.set_xlabel('Phase Frequency (Hz)', fontsize=16)
                hmr.set_ylabel('Amplitude Frequency (Hz)\n', fontsize=16)
                hmr.set_xticks(ix, xtv)
                hmr.set_yticks(iy, ytv)
                hmr.tick_params(direction='out', labelsize=14, length=4, width=1, colors='k', bottom=True, left=True)
                hmr.collections[0].colorbar.ax.tick_params(labelsize=14)
                hmr.figure.axes[-1].yaxis.label.set_size(16)
                ax = plt.gca()
                for _,spine in ax.spines.items():
                    spine.set_visible(True)
                plt.show()
                output = plt.Axes.get_figure(hmr)   
                output.savefig(write_dir + ttl + reglbl + '-EFFECTSIZE_reg.png', bbox_inches='tight', format=image_format)
                plt.close(output)
                
                height = [np.mean(alc_pac), np.mean(unaff_pac), 0]
                width = [0.2 , 0.2 , 1]
                x_pos = [0.25 , 0.5 , 1]
                bars = ['AUD','Unaff','']
                plt.title(ttl + reglbl + '\np = ' + str(stats.pvalue), fontsize = 9)
                plt.bar(x_pos, height , width=width, color='w', edgecolor='k')
                plt.errorbar(x_pos, height ,yerr=[sem(alc_pac),sem(unaff_pac),0], fmt='o', color='r')
                # plt.grid(axis='y')
                plt.xticks(x_pos, bars)
                plt.ylim([180,195])
                plt.ylabel('PAC')
                # plt.axis
                ax = plt.gca()
                ax.set_facecolor('0.9')
                ax.spines['bottom'].set_color('black')
                ax.spines['left'].set_color('black')
                fig = plt.gcf()
                plt.show()   
                plt.draw()
                fig.savefig(write_dir + ttl + reglbl + '-EFFECTSIZE_reg_bar.png', bbox_inches='tight', format=image_format)
                plt.close()
                print('\nAUD: ' + str(np.mean(alc_pac)) + ' +/- ' + str(sem(alc_pac)))
                print('Unaff: ' + str(np.mean(unaff_pac)) + ' +/- ' + str(sem(unaff_pac)))
        
                pa_alc = pac_age[(pac_age[aud_symp]>=6)] # & (pac_age.sex=='F')]
                pa_ctl = pac_age[(pac_age[aud_symp]==0)] # & (pac_age.sex=='F')]
                # ax1 = pa_ctl.plot(y='PAC',x='age', c='b', kind='scatter', label='Unaff', title=ttl + '\n' + reglbl, s=2)
                # pa_alc.plot(y='PAC',x='age', c='r', kind='scatter',label='AUD', ax=ax1, s= 2)
                # plt.xlim([24,51])
                # plt.ylim([100,255])
                # plt.show()
                
                # pa_alc.plot(y='PAC',x='age', c='r', kind='scatter',label='AUD', s= 2, title=ttl + '\n' + reglbl)
                # plt.xlim([24,51])
                # plt.ylim([100,255])
                # plt.show()
                
                # pa_ctl.plot(y='PAC',x='age', c='b', kind='scatter',label='unaff', s= 2, title=ttl + '\n' + reglbl)
                # plt.xlim([24,51])
                # plt.ylim([100,255])
                # plt.show()
                                  
                X = pa_alc.age.values #.reshape((-1, 1)) 
                y = pa_alc.PAC.values 
                res = linregress(X,y)
                plt.plot(X,y,'o', label='AUD')
                plt.plot(X, res.intercept + res.slope*X, 'r', label='fitted line')
                plt.xlim([24,51])
                plt.ylim([100,255])
                plt.xlabel('AGE')
                plt.ylabel('PAC')
                plt.title(SEX + ', p = ' + str(round(res.pvalue,3)) + ' AUD ' + reglbl)
                plt.legend()
                plt.show()
                
                print('\nAUD')                 
                print('p = ' + str(res.pvalue))
                print('slope = ' + str(res.slope))
                print('intercept = ' + str(res.intercept))
                print('r_value = ' + str(res.rvalue))
                print('std_err = ' + str(res.stderr))
                print('r^2 = ' + str((res.rvalue**2)))
                print('\n')
                
                X = pa_ctl.age.values #.reshape((-1, 1)) 
                y = pa_ctl.PAC.values 
                res = linregress(X,y)
                plt.plot(X,y,'o', label='Unaffected')
                plt.plot(X, res.intercept + res.slope*X, 'r', label='fitted line')
                plt.xlim([24,51])
                plt.ylim([100,255])
                plt.xlabel('AGE')
                plt.ylabel('PAC')
                plt.title(SEX + ', p = ' + str(round(res.pvalue,3)) + ' Unaffected ' + reglbl)
                plt.legend()
                plt.show()
                
                print('Unaffected')                 
                print('p = ' + str(res.pvalue))
                print('slope = ' + str(res.slope))
                print('intercept = ' + str(res.intercept))
                print('r_value = ' + str(res.rvalue))
                print('std_err = ' + str(res.stderr))
                print('r^2 = ' + str((res.rvalue**2)))
                print('\n')
    
    if 0:
        
        #  MEN
        interval_numy = 7
        interval_numx = 7
        nn = np.array( es)
        lenx = np.shape(nn)[1]
        leny = np.shape(nn)[0]
        yy = np.arange(4,50,((50-4)/leny))
        ytks = np.array([str(np.round(y,1)) for y in yy])
        intvly = round((leny)/interval_numy)
        iya = np.arange(0, leny,intvly)
        ytv = ytks[iya]  
        iy = np.arange(leny,0,-intvly) # NEED TO USE A REVERSED INDEX LIST BECAUSE OF HJOW sms HEATMAP STORES VALUES 
        # iy = np.array(iya.tolist()[::-1]) # NEED TO USE A REVERSED INDEX LIST BECAUSE OF HJOW sms HEATMAP STORES VALUES 
        print('lengths: ytv= ' + str(len(ytv)) + ', iy= ' + str(len(iy)))
        xx = np.arange(0,13,((13-0)/lenx))
        xx = np.append(xx,13) 
        xtks = np.array([str(np.round(x,1)) for x in xx])
        intvlx = round((lenx)/interval_numx)
        ix = np.arange(0,lenx,intvlx)
        xtv = xtks[ix]  
        ix = np.append(ix,lenx)
        xtv = np.append(xtv,13)
        print('lengths: xtv= ' + str(len(xtv)) + ', ix= ' + str(len(ix)))
        pv_fig = pv.copy()
        pv_fig[pv_fig>=alpha] = 0
        hmm = sns.heatmap(es,
                          cmap="icefire", 
                          cbar_kws={'label': 'PAC strength change\n(AUD - unaff)'}, 
                          mask=(pv_fig==0), 
                          vmin=vmin, vmax=vmax,
                          xticklabels=False,
                          yticklabels=False
                          )
        hmm.set_xlabel('Phase Frequency (Hz)', fontsize=18)
        hmm.set_ylabel('Amplitude Frequency (Hz)\n', fontsize=18)
        hmm.set_xticks(ix, xtv)
        hmm.set_yticks(iy, ytv)
        hmm.tick_params(direction='out', labelsize=16, length=4, width=1, colors='k', bottom=True, left=True)
        hmm.collections[0].colorbar.ax.tick_params(labelsize=16)
        hmm.figure.axes[-1].yaxis.label.set_size(18)
        ax = plt.gca()
        fpl = 9
        fph = 12
        fal = 30
        fah = 45
        fp = pd.DataFrame([float(f) for f in freq_pha], columns=['freq'])
        fp_lo = fp[(fp.freq>=fpl)].index[0]
        fp_hi = fp[(fp.freq>=fph)].index[0]
        fa = pd.DataFrame([float(f) for f in freq_amp], columns=['freq'])
        fa_lo = fa[(fa.freq<=fal)].index[0]
        fa_hi = fa[(fa.freq<=fah)].index[0]
        ax.add_patch(
            patches.Rectangle(
                (fp_lo,fa_hi), 
                fp_hi-fp_lo, 
                fa_lo-fa_hi,
                edgecolor='red',
                fill=False,
                lw=2 ))
        fpl = 4
        fph = 7
        fal = 33
        fah = 40
        fp = pd.DataFrame([float(f) for f in freq_pha], columns=['freq'])
        fp_lo = fp[(fp.freq>=fpl)].index[0]
        fp_hi = fp[(fp.freq>=fph)].index[0]
        fa = pd.DataFrame([float(f) for f in freq_amp], columns=['freq'])
        fa_lo = fa[(fa.freq<=fal)].index[0]
        fa_hi = fa[(fa.freq<=fah)].index[0]
        ax.add_patch(
            patches.Rectangle(
                (fp_lo,fa_hi), 
                fp_hi-fp_lo, 
                fa_lo-fa_hi,
                edgecolor='red',
                fill=False,
                lw=2 ))
        fpl = 3.5
        fph = 5
        fal = 11
        fah = 14
        fp = pd.DataFrame([float(f) for f in freq_pha], columns=['freq'])
        fp_lo = fp[(fp.freq>=fpl)].index[0]
        fp_hi = fp[(fp.freq>=fph)].index[0]
        fa = pd.DataFrame([float(f) for f in freq_amp], columns=['freq'])
        fa_lo = fa[(fa.freq<=fal)].index[0]
        fa_hi = fa[(fa.freq<=fah)].index[0]
        ax.add_patch(
            patches.Rectangle(
                (fp_lo,fa_hi), 
                fp_hi-fp_lo, 
                fa_lo-fa_hi,
                edgecolor='red',
                fill=False,
                lw=2 ))
        fpl = 8
        fph = 12
        fal = 18
        fah = 22
        fp = pd.DataFrame([float(f) for f in freq_pha], columns=['freq'])
        fp_lo = fp[(fp.freq>=fpl)].index[0]
        fp_hi = fp[(fp.freq>=fph)].index[0]
        fa = pd.DataFrame([float(f) for f in freq_amp], columns=['freq'])
        fa_lo = fa[(fa.freq<=fal)].index[0]
        fa_hi = fa[(fa.freq<=fah)].index[0]
        ax.add_patch(
            patches.Rectangle(
                (fp_lo,fa_hi), 
                fp_hi-fp_lo, 
                fa_lo-fa_hi,
                edgecolor='red',
                fill=False,
                lw=2 ))
        for _,spine in ax.spines.items():
            spine.set_visible(True)
        plt.show()
        
        
        
        # WOMEN
        interval_numy = 7
        interval_numx = 7
        nn = np.array( es)
        lenx = np.shape(nn)[1]
        leny = np.shape(nn)[0]
        yy = np.arange(4,50,((50-4)/leny))
        ytks = np.array([str(np.round(y,1)) for y in yy])
        intvly = round((leny)/interval_numy)
        iya = np.arange(0, leny,intvly)
        ytv = ytks[iya]  
        iy = np.arange(leny,0,-intvly) # NEED TO USE A REVERSED INDEX LIST BECAUSE OF HJOW sms HEATMAP STORES VALUES 
        # iy = np.array(iya.tolist()[::-1]) # NEED TO USE A REVERSED INDEX LIST BECAUSE OF HJOW sms HEATMAP STORES VALUES 
        print('lengths: ytv= ' + str(len(ytv)) + ', iy= ' + str(len(iy)))
        xx = np.arange(0,13,((13-0)/lenx))
        xx = np.append(xx,13) 
        xtks = np.array([str(np.round(x,1)) for x in xx])
        intvlx = round((lenx)/interval_numx)
        ix = np.arange(0,lenx,intvlx)
        xtv = xtks[ix]  
        ix = np.append(ix,lenx)
        xtv = np.append(xtv,13)
        print('lengths: xtv= ' + str(len(xtv)) + ', ix= ' + str(len(ix)))
        pv_fig = pv.copy()
        pv_fig[pv_fig>=alpha] = 0
        hmm = sns.heatmap(es,
                          cmap="icefire", 
                          cbar_kws={'label': 'PAC strength change\n(AUD - unaff)'}, 
                          mask=(pv_fig==0), 
                          vmin=vmin, vmax=vmax,
                          xticklabels=False,
                          yticklabels=False
                          )
        hmm.set_xlabel('Phase Frequency (Hz)', fontsize=18)
        hmm.set_ylabel('Amplitude Frequency (Hz)\n', fontsize=18)
        hmm.set_xticks(ix, xtv)
        hmm.set_yticks(iy, ytv)
        hmm.tick_params(direction='out', labelsize=16, length=4, width=1, colors='k', bottom=True, left=True)
        hmm.collections[0].colorbar.ax.tick_params(labelsize=16)
        hmm.figure.axes[-1].yaxis.label.set_size(18)
        ax = plt.gca()
        fpl = 10
        fph = 13
        fal = 35
        fah = 45
        fp = pd.DataFrame([float(f) for f in freq_pha], columns=['freq'])
        fp_lo = fp[(fp.freq>=fpl)].index[0]
        fp_hi = fp[(fp.freq>=fph)].index[0]
        fa = pd.DataFrame([float(f) for f in freq_amp], columns=['freq'])
        fa_lo = fa[(fa.freq<=fal)].index[0]
        fa_hi = fa[(fa.freq<=fah)].index[0]
        ax.add_patch(
            patches.Rectangle(
                (fp_lo,fa_hi), 
                fp_hi-fp_lo, 
                fa_lo-fa_hi,
                edgecolor='red',
                fill=False,
                lw=2 ))
        
        fpl = 2.4
        fph = 3.6
        fal = 26
        fah = 41
        fp = pd.DataFrame([float(f) for f in freq_pha], columns=['freq'])
        fp_lo = fp[(fp.freq>=fpl)].index[0]
        fp_hi = fp[(fp.freq>=fph)].index[0]
        fa = pd.DataFrame([float(f) for f in freq_amp], columns=['freq'])
        fa_lo = fa[(fa.freq<=fal)].index[0]
        fa_hi = fa[(fa.freq<=fah)].index[0]
        ax.add_patch(
            patches.Rectangle(
                (fp_lo,fa_hi), 
                fp_hi-fp_lo, 
                fa_lo-fa_hi,
                edgecolor='red',
                fill=False,
                lw=2 ))
        
        fpl = 7
        fph = 11
        fal = 14
        fah = 17
        fp = pd.DataFrame([float(f) for f in freq_pha], columns=['freq'])
        fp_lo = fp[(fp.freq>=fpl)].index[0]
        fp_hi = fp[(fp.freq>=fph)].index[0]
        fa = pd.DataFrame([float(f) for f in freq_amp], columns=['freq'])
        fa_lo = fa[(fa.freq<=fal)].index[0]
        fa_hi = fa[(fa.freq<=fah)].index[0]
        ax.add_patch(
            patches.Rectangle(
                (fp_lo,fa_hi), 
                fp_hi-fp_lo, 
                fa_lo-fa_hi,
                edgecolor='red',
                fill=False,
                lw=2 ))
        fpl = 3
        fph = 4.5
        fal = 8
        fah = 15
        fp = pd.DataFrame([float(f) for f in freq_pha], columns=['freq'])
        fp_lo = fp[(fp.freq>=fpl)].index[0]
        fp_hi = fp[(fp.freq>=fph)].index[0]
        fa = pd.DataFrame([float(f) for f in freq_amp], columns=['freq'])
        fa_lo = fa[(fa.freq<=fal)].index[0]
        fa_hi = fa[(fa.freq<=fah)].index[0]
        ax.add_patch(
            patches.Rectangle(
                (fp_lo,fa_hi), 
                fp_hi-fp_lo, 
                fa_lo-fa_hi,
                edgecolor='red',
                fill=False,
                lw=2 ))
        for _,spine in ax.spines.items():
            spine.set_visible(True)
        plt.show()
    
    

    


