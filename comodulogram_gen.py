# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 16:14:57 2024

@author: lifep
"""

import numpy as np
from PIL import Image
import coga_support_defs as csd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, mannwhitneyu, sem, linregress
import os
from age_matcher import AgeMatcher

SEX = 'M' 
channel = 'FZ'
age_groups = [25,50]
grps = [0,0,0]
vmin = -3
vmax = 3
shuf_seeds = 424242
severity_scores = [6,11,'SEVERE']
alpha = 0.05
which_dx = 'AUD' # AUD ALAB ALD
# race = ''

fldrname = 'COMODULOGRAMS'


write_dir = 'C:\\Users\\lifep\\OneDrive - Downstate Medical Center\\PAC stats paper\\' + fldrname + '\\'
if not os.path.exists(write_dir):
    os.makedirs(write_dir) 
                
#  FREQUENCY VALUES FOR PHASE AND AMPLITUDE 
xax = np.arange(0,13,(13/224))
yax = np.arange(4,50,(46/224))

freq_pha = [str(round(x,2)) for x in xax]
# freq_amp = [str(round(x,2)) for x in yax]
freq_amp = [str(round(x,2)) for x in np.sort(yax)[::-1]]

min_age = age_groups[0]
max_age = age_groups[1]

channelstr = channel.lower()
targ_folder = '' + channelstr + '_' + str(min_age) + '_' + str(max_age) + '_' + which_dx + '_' + SEX + '_' + severity_scores[2] + '_' + str(severity_scores[0]) + '_' + str(severity_scores[1])

# MAKE SURE THAT THE INDEXING IS IDENTICAL BETWEEN pac_all AND images_all 
# THEY MUST ALSO HAVE THE SAME LENGTH, E.G., 8781
images_all = np.load('C:\\Users\\lifep\\OneDrive\\Documents\\pac_3d_fz_1_99_AUD__ALL_0_11.npy')
pac_all = pd.read_pickle('C:\\Users\\lifep\\OneDrive\\Documents\\pac_age_fz_1_99_AUD__ALL_0_11.pkl')
if SEX=='':
    pac_all = pac_all[(pac_all.age>=min_age) & (pac_all.age<=max_age)]
else:
    pac_all = pac_all[(pac_all.age>=min_age) & (pac_all.age<=max_age) & (pac_all.sex==SEX)]    


aaa = set(pac_all[(pac_all.AUD==True) & (pac_all.audcnt>=severity_scores[0]) & (pac_all.audcnt<=severity_scores[1])].ID)
uuu = set(pac_all[(pac_all.AUD==False)].ID)
# SINCE THERE ARE MULIPLE VISITS THE SAME SUBJECT CAN SHOW UP AS AUD OR NOT DEPENDING ON HOW YOUNG THEY WERE ON THEIR FIRST VISIT TO MAXIMIZE AUD SAMPLE SIZE 
alcs = list( aaa.difference(uuu) )
boths = list( aaa.intersection(uuu) )
# NOW WE MAKE SURE THAT CONTROLS HAVE NEVER BEEN DIAGNOSED WITH AUD TO EXCLUDE SUBJECTS THAT GO FROM UNAFFECTED TO AUD 
uuu = set(pac_all[(pac_all.AUD==False) & (pac_all.audcnt==0)].ID)
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
    matcher = AgeMatcher(age_tol=5, age_col='age', sex_col='sex', strategy='greedy', shuffle_df=True, random_state=shuf_seeds)
matched_cases, matched_controls = matcher(alc, ctl)
pac_age = pd.concat([matched_cases,matched_controls])
match_i = pac_age.index

ttl = targ_folder + ' alc_' + str(len(matched_cases)) + ' unaff_' + str(len(matched_controls)) 


# EXTRACTING SPECTRAL POWER INFO FROM pacdat
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

pval_mx = np.zeros((224,224))
effect_mx = np.zeros((224,224))
# aud_mx = np.zeros((224,224))
# aud_sem_mx = np.zeros((224,224))
# ctl_mx = np.zeros((224,224))
# ctl_sem_mx = np.zeros((224,224))
print('doing statistics on all PAC frequency pairs')
# images = np.load('C:\\Users\\lifep\\OneDrive\\Documents\\pac_3d_' + targ_folder + '.npy')
images = images_all[match_i]
for x in range(224):
    for y in range(224):
        # print(str(x) + ' ' + str(y))
        alc_i = np.where(pac_age.AUD==1)
        unaff_i = np.where(pac_age.AUD==0)
        alc_pac = images[alc_i,x,y][0]
        unaff_pac = images[unaff_i,x,y][0]
        # stats = ttest_ind(alc_pac[0], unaff_pac[0], equal_var=False)
        stats = mannwhitneyu(alc_pac, unaff_pac)
        pval_mx[x,y] = stats.pvalue
        effect_mx[x,y] = np.mean(alc_pac) -  np.mean(unaff_pac)
        # aud_mx[x,y] = np.mean(alc_pac)
        # aud_sem_mx[x,y] = np.std(alc_pac)/np.sqrt(len(alc_pac))
        # ctl_mx[x,y] = np.mean(unaff_pac)
        # ctl_sem_mx[x,y] = np.std(unaff_pac)/np.sqrt(len(unaff_pac))
        
        
pv = pd.DataFrame(pval_mx,  columns=freq_pha, index=freq_amp)
pv[pv>0.05] = 0.05
hmmin = str(round(np.min(pv),6))
hm = sns.heatmap(pv, vmax=0.05,cmap="rocket_r", cbar_kws={'label': 'p-value (min=' + hmmin + ')'})
plt.title(targ_folder, fontsize = 9)
plt.xlabel('Phase Frequency (Hz)', fontsize = 9) 
plt.ylabel('Amplitude Frequency (Hz)', fontsize = 9)    
output = plt.Axes.get_figure(hm)
    
output.savefig(write_dir + ttl + '-PVALUES.jpg', bbox_inches='tight')
plt.show()
plt.close(output)

es = pd.DataFrame(effect_mx,  columns=freq_pha, index=freq_amp)
hm = sns.heatmap(es, cmap="icefire", cbar_kws={'label': 'diff mean PAC strength (AUD - unaff)'}, vmin=vmin, vmax=vmax)
plt.title(ttl, fontsize = 9)
plt.xlabel('Phase Frequency (Hz)', fontsize = 9) 
plt.ylabel('Amplitude Frequency (Hz)', fontsize = 9)
output = plt.Axes.get_figure(hm)
plt.show()
output.savefig(write_dir + ttl + '_EFFECTSIZE', bbox_inches='tight')
plt.close(output)

# # low alpha-mid beta
# fpl = 7.83
# fph = 10.97
# fal = 20.02
# fah = 24.95

# # alpha-gamma MEN
# fpl = 8.01
# fph = 12.19
# fal = 29.46
# fah = 47.54

# # alpha-gamma WOMEN
# fpl = 10.04
# fph = 12.19
# fal = 29.46
# fah = 47.54

# # # amplitude gamma
# fpl = 1.04
# fph = 12.54
# fal = 29.46
# fah = 47.54

# low-theta-gamma
# fpl = 3.02
# fph = 3.37
# fal = 37.68
# fah = 48.36

# alpha-gamma2
# fpl = 8.98
# fph = 9.92
# fal = 40.76
# fah = 45.28

# # theta-gamma MEN
# fpl = 3.95
# fph = 6.96
# fal = 30.29
# fah = 40.76

# # theta-gamma WOMEN
# fpl = 4.0
# fph = 4.99
# fal = 39.73
# fah = 45.28

# # theta-gamma WOMEN
# fpl = 2.55
# fph = 3.6
# fal = 24.95
# fah = 40.76

# # theta-beta WOMEN
# fpl = 2.09
# fph = 4.7
# fal = 9.13
# fah = 15.91

# # theta-beta MEN AND WOMEN
# fpl = 3.6
# fph = 4.35
# fal = 12.21
# fah = 15.09

# alpha-delta MEN AND WOMEN
fpl = 9.75
fph = 10.97
fal = 36.24
fah = 44.87

# GET INDICES FOR PHASE AND AMPLITUDE FREQUENCIES TO DO PAC REGION STATISTICS
fp = pd.DataFrame([float(f) for f in freq_pha], columns=['freq'])
fp_lo = fp[(fp.freq==fpl)].index[0]
fp_hi = fp[(fp.freq==fph)].index[0]
# WE REVERSE THIS SO THAT Y-AXIS IS PLOTTED CORRECTLY
# freq_amp.reverse()
fa = pd.DataFrame([float(f) for f in freq_amp], columns=['freq'])
fa_lo = fa[(fa.freq==fal)].index[0]
fa_hi = fa[(fa.freq==fah)].index[0]

fal_lbl = str(fal).replace('.','_') 
fah_lbl = str(fah).replace('.','_')
fpl_lbl =  str(fpl).replace('.','_') 
fph_lbl = str(fph).replace('.','_')
reglbl = '_fp_' + fpl_lbl + '__' + fph_lbl +'_fa_' + fal_lbl + '__' + fah_lbl
# reglbl = '_fa_' + str(fa_hi) + '_' + str(fa_lo) + '_fp_' + str(fp_lo) + '_' + str(fp_hi)                   

# # # HELPS TO GET AVAILABLE FREQUENCIES
# fa[(fa.freq>=15) & (fa.freq<=17)]
# fp[(fp.freq>=4) & (fp.freq<=6)]

es_region = es.iloc[fa_hi:fa_lo, fp_lo:fp_hi]
regions = [0]*len(images)
for thispac in range(len(images)):
    regions[thispac] = np.mean(images[thispac,fa_hi:fa_lo, fp_lo:fp_hi])
regions = np.array(regions)

pac_age.insert(0, 'PAC', regions)

print('doing statistics on PAC frequency pair region')
alc_i = np.where(pac_age.AUD==1)
unaff_i = np.where(pac_age.AUD==0)
alc_pac = regions[alc_i]
unaff_pac = regions[unaff_i]                   
# stats = ttest_ind(alc_pac, unaff_pac, equal_var=False)
stats = mannwhitneyu(alc_pac, unaff_pac)
# if stats.pvalue<=0.05:
if stats.pvalue<=1:
    hm = sns.heatmap(es_region, cmap="icefire", cbar_kws={'label': 'diff mean PAC strength (AUD - unaff)'}, vmin=vmin, vmax=vmax)
    plt.title(ttl + reglbl, fontsize = 9)
    plt.xlabel('Phase Frequency (Hz)', fontsize = 9) 
    plt.ylabel('Amplitude Frequency (Hz)', fontsize = 9)
    output = plt.Axes.get_figure(hm)
    plt.show()
    output.savefig(write_dir + ttl + '-EFFECTSIZE_reg' + reglbl, bbox_inches='tight')
    plt.close(output)

    plt.title(ttl + reglbl + '\np = ' + str(stats.pvalue), fontsize = 9)

    plt.errorbar(['alc','unaff'],[np.mean(alc_pac),np.mean(unaff_pac)],yerr=[sem(alc_pac),sem(unaff_pac)])
    plt.show()
    plt.savefig(write_dir + ttl + '-EFFECTSIZE_reg_bar' + reglbl, bbox_inches='tight')
    plt.close()



pv[pv>=alpha] = 0
hm = sns.heatmap(es, cmap="icefire", cbar_kws={'label': 'diff mean PAC strength (AUD - unaff)'}, mask=(pv==0), vmin=vmin, vmax=vmax)
plt.title(ttl, fontsize = 9)
plt.xlabel('Phase Frequency (Hz)', fontsize = 9) 
plt.ylabel('Amplitude Frequency (Hz)', fontsize = 9)
output = plt.Axes.get_figure(hm)
output.savefig(write_dir + ttl + '-EFFECTSIZExPVAL', bbox_inches='tight')
plt.show()
plt.close(output)

# es = pd.DataFrame(aud_mx,  columns=freq_pha, index=freq_amp)
# hm = sns.heatmap(es, cmap="Reds", cbar_kws={'label': 'AUD mean PAC strength'},vmin=185,vmax=200) #, vmin= np.min(es), vmax=200)
# plt.title( ttl + '_AUDMEAN', fontsize = 9)
# plt.xlabel('Phase Frequency (Hz)', fontsize = 9) 
# plt.ylabel('Amplitude Frequency (Hz)', fontsize = 9)
# output = plt.Axes.get_figure(hm)
# plt.show()
# output.savefig(write_dir + ttl + '_AUDMEAN', bbox_inches='tight')
# plt.close(output)

# es = pd.DataFrame(ctl_mx,  columns=freq_pha, index=freq_amp)
# hm = sns.heatmap(es, cmap="Reds", cbar_kws={'label': 'Unaffected mean PAC strength'},vmin=185,vmax=200) # , vmin= np.min(es), vmax=200)
# plt.title(ttl + '_UNAFFMEAN', fontsize = 9)
# plt.xlabel('Phase Frequency (Hz)', fontsize = 9) 
# plt.ylabel('Amplitude Frequency (Hz)', fontsize = 9)
# output = plt.Axes.get_figure(hm)
# plt.show()
# output.savefig(write_dir + ttl + '_UNAFFMEAN', bbox_inches='tight')
# plt.close(output)

# es = pd.DataFrame(aud_sem_mx,  columns=freq_pha, index=freq_amp)
# hm = sns.heatmap(es, cmap="Reds", cbar_kws={'label': 'AUD SEM PAC strength'})#,vmin=0,vmax=1.2) #, vmin= np.min(es), vmax=200)
# plt.title( ttl + '_AUDMEAN', fontsize = 9)
# plt.xlabel('Phase Frequency (Hz)', fontsize = 9) 
# plt.ylabel('Amplitude Frequency (Hz)', fontsize = 9)
# output = plt.Axes.get_figure(hm)
# plt.show()
# output.savefig(write_dir + ttl + '_AUDMEAN', bbox_inches='tight')
# plt.close(output)

# es = pd.DataFrame(ctl_sem_mx,  columns=freq_pha, index=freq_amp)
# hm = sns.heatmap(es, cmap="Reds", cbar_kws={'label': 'Unaffected SEM PAC strength'})#,vmin=0,vmax=1.2) # , vmin= np.min(es), vmax=200)
# plt.title(ttl + '_UNAFFMEAN', fontsize = 9)
# plt.xlabel('Phase Frequency (Hz)', fontsize = 9) 
# plt.ylabel('Amplitude Frequency (Hz)', fontsize = 9)
# output = plt.Axes.get_figure(hm)
# plt.show()
# output.savefig(write_dir + ttl + '_UNAFFMEAN', bbox_inches='tight')
# plt.close(output)




