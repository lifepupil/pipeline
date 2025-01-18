# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 13:18:19 2025

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
import statsmodels.api as sm
import statsmodels.formula.api as smf

new_test = True
SEX = 'M' 
channel = 'FZ'
# min_age = 25
# max_age = 50
min_age = 1
max_age = 99
severity_scores = [6,11,'SEVERE']
alpha = 0.05
which_dx = 'AUD' # AUD ALAB ALD
fldrname = 'LMM'
vmin = -3
vmax = 3


write_dir = 'C:\\Users\\lifep\\OneDrive - Downstate Medical Center\\PAC stats paper\\' + fldrname + '\\'
if not os.path.exists(write_dir):
    os.makedirs(write_dir) 
                
#  FREQUENCY VALUES FOR PHASE AND AMPLITUDE 
xax = np.arange(0,13,(13/224))
yax = np.arange(4,50,(46/224))

freq_pha = [str(round(x,2)) for x in xax]
freq_amp = [str(round(x,2)) for x in np.sort(yax)[::-1]]

targ_folder = '' + channel + '_' + str(min_age) + '_' + str(max_age) + '_' + which_dx + '_' + SEX + '_' + severity_scores[2] + '_' + str(severity_scores[0]) + '_' + str(severity_scores[1])

# MAKE SURE THAT THE INDEXING IS IDENTICAL BETWEEN pac_all AND images_all 
# THEY MUST ALSO HAVE THE SAME LENGTH, E.G., 8781
images_all = np.load('C:\\Users\\lifep\\OneDrive\\Documents\\pac_3d_fz_1_99_AUD__ALL_0_11.npy')
pac_all = pd.read_pickle('C:\\Users\\lifep\\OneDrive\\Documents\\pac_age_fz_1_99_AUD__ALL_0_11.pkl')
del pac_all['PAC']

if SEX=='':
    pac_age = pac_all[(pac_all.age>=min_age) & (pac_all.age<=max_age)]
else:
    pac_age = pac_all[(pac_all.age>=min_age) & (pac_all.age<=max_age) & (pac_all.sex==SEX)]    
match_i = pac_age.index
images = images_all[match_i]

pa_alc = pac_age[(pac_age.audcnt>=severity_scores[0])] 
pa_ctl = pac_age[(pac_age.audcnt==0)]
ttl = targ_folder + ' alc_' + str(len(pa_alc)) + ' unaff_' + str(len(pa_ctl)) 
    
print('\n ' + ttl)


if new_test:    
    pval_mx = np.zeros((224,224))
    effect_mx = np.zeros((224,224))
    
    print('doing statistics on all PAC frequency pairs')
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
        
    pv = pd.DataFrame(pval_mx,  columns=freq_pha, index=freq_amp)
    hmmin = str(round(np.min(pv),6))
    pv[pv>alpha] = alpha    
    hm = sns.heatmap(pv, vmax=alpha,cmap="rocket_r", cbar_kws={'label': 'p-value (min=' + hmmin + ')'})
    plt.title(ttl, fontsize = 9)
    plt.xlabel('Phase Frequency (Hz)', fontsize = 9) 
    plt.ylabel('Amplitude Frequency (Hz)', fontsize = 9)    
    plt.show()
    # output = plt.Axes.get_figure(hm)    
    # output.savefig(write_dir + ttl + '-PVALUES.jpg', bbox_inches='tight')
    # plt.close(output)

    es = pd.DataFrame(effect_mx,  columns=freq_pha, index=freq_amp)
    hm = sns.heatmap(es, cmap="icefire", cbar_kws={'label': 'diff mean PAC strength (AUD - unaff)'})
    plt.title(ttl, fontsize = 9)
    plt.xlabel('Phase Frequency (Hz)', fontsize = 9) 
    plt.ylabel('Amplitude Frequency (Hz)', fontsize = 9)
    plt.show()
    # output = plt.Axes.get_figure(hm)
    # output.savefig(write_dir + ttl + '_EFFECTSIZE', bbox_inches='tight')
    # plt.close(output)
    
    pv[pv>=alpha] = 0
    # hm = sns.heatmap(es, cmap="icefire", cbar_kws={'label': 'diff mean PAC strength (AUD - unaff)'}, mask=(pv==0), vmin=vmin, vmax=vmax)
    hm = sns.heatmap(es, cmap="icefire", cbar_kws={'label': 'diff mean PAC strength (AUD - unaff)'}, mask=(pv==0))
    plt.title(ttl, fontsize = 9)
    plt.xlabel('Phase Frequency (Hz)', fontsize = 9) 
    plt.ylabel('Amplitude Frequency (Hz)', fontsize = 9)
    plt.show()
    # output = plt.Axes.get_figure(hm)
    # output.savefig(write_dir + ttl + '-EFFECTSIZExPVAL', bbox_inches='tight')
    # plt.close(output)
    
    es.to_pickle(write_dir + 'effect_mx_tmp.pkl')
    pv.to_pickle(write_dir + 'pval_mx_tmp.pkl')
else:
    es = pd.read_pickle(write_dir + 'effect_mx_tmp.pkl')
    pv = pd.read_pickle(write_dir + 'pval_mx_tmp.pkl')
    

# # low alpha-mid beta
# fpl = 7.83
# fph = 10.97
# fal = 20.02
# fah = 24.95

# # # alpha-gamma MEN
# fpl = 8.01
# fph = 12.19
# fal = 29.46
# fah = 47.54

# # alpha-gamma WOMEN
# fpl = 10.04
# fph = 12.19
# fal = 29.46
# fah = 47.54

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

# # alpha-delta MEN AND WOMEN
# fpl = 9.75
# fph = 10.97
# fal = 36.24
# fah = 44.87

# explore set
fpl = 4.88
fph = 4.99
fal = 36.45
fah = 37.68

# GET INDICES FOR PHASE AND AMPLITUDE FREQUENCIES TO DO PAC REGION STATISTICS
fp = pd.DataFrame([float(f) for f in freq_pha], columns=['freq'])
fa = pd.DataFrame([float(f) for f in freq_amp], columns=['freq'])
# HELPS TO GET AVAILABLE FREQUENCIES
# fa[(fa.freq>=15) & (fa.freq<=17)]
# fp[(fp.freq>=4) & (fp.freq<=6)]

fp_lo = fp[(fp.freq==fpl)].index[0]
fp_hi = fp[(fp.freq==fph)].index[0]
fa_lo = fa[(fa.freq==fal)].index[0]
fa_hi = fa[(fa.freq==fah)].index[0]
fal_lbl = str(fal).replace('.','_') 
fah_lbl = str(fah).replace('.','_')
fpl_lbl =  str(fpl).replace('.','_') 
fph_lbl = str(fph).replace('.','_')
reglbl = '_fp_' + fpl_lbl + '__' + fph_lbl +'_fa_' + fal_lbl + '__' + fah_lbl
# reglbl = '_fa_' + str(fa_hi) + '_' + str(fa_lo) + '_fp_' + str(fp_lo) + '_' + str(fp_hi)                   

es_region = es.iloc[fa_hi:fa_lo, fp_lo:fp_hi]
# hm = sns.heatmap(es_region, cmap="icefire", cbar_kws={'label': 'diff mean PAC strength (AUD - unaff)'}, vmin=vmin, vmax=vmax)
hm = sns.heatmap(es_region, cmap="icefire", cbar_kws={'label': 'diff mean PAC strength (AUD - unaff)'})
plt.title(ttl + reglbl, fontsize = 9)
plt.xlabel('Phase Frequency (Hz)', fontsize = 9) 
plt.ylabel('Amplitude Frequency (Hz)', fontsize = 9)
plt.show()
# output = plt.Axes.get_figure(hm)
# output.savefig(write_dir + ttl + '-EFFECTSIZE_reg' + reglbl, bbox_inches='tight')
# plt.close(output)

pv[pv>=alpha] = 0
pv_region = pv.iloc[fa_hi:fa_lo, fp_lo:fp_hi]
# hm = sns.heatmap(es_region, cmap="icefire", cbar_kws={'label': 'diff mean PAC strength (AUD - unaff)'}, mask=(pv_region==0), vmin=-1.25, vmax=2.25)
hm = sns.heatmap(es_region, cmap="icefire", cbar_kws={'label': 'diff mean PAC strength (AUD - unaff)'}, mask=(pv_region==0))
plt.title(ttl, fontsize = 9)
plt.xlabel('Phase Frequency (Hz)', fontsize = 9) 
plt.ylabel('Amplitude Frequency (Hz)', fontsize = 9)
plt.show()

regions = [0]*len(images)
for thispac in range(len(images)):
    regions[thispac] = np.mean(images[thispac,fa_hi:fa_lo, fp_lo:fp_hi])
regions = np.array(regions)
pac_age.insert(0, 'PAC', regions)
pa_alc = pac_age[(pac_age.audcnt>=severity_scores[0])] 
pa_ctl = pac_age[(pac_age.audcnt==0)]
ttl = targ_folder + ' alc_' + str(len(pa_alc)) + ' unaff_' + str(len(pa_ctl)) 

print('doing statistics on PAC frequency pair region')
alc_i = np.where(pac_age.audcnt>=severity_scores[0])
unaff_i = np.where(pac_age.audcnt==0)
alc_pac = regions[alc_i]
unaff_pac = regions[unaff_i]                   
# stats = ttest_ind(alc_pac, unaff_pac, equal_var=False)
stats = mannwhitneyu(alc_pac, unaff_pac)
print('\n p = ' + str(stats.pvalue) + '\n')
if stats.pvalue<=0.05:
    plt.title(ttl + reglbl + '\np = ' + str(stats.pvalue), fontsize = 9)
    plt.errorbar(['alc','unaff'],[np.mean(alc_pac),np.mean(unaff_pac)],yerr=[sem(alc_pac),sem(unaff_pac)])
    plt.show()
    plt.savefig(write_dir + ttl + '-EFFECTSIZE_reg_bar' + reglbl, bbox_inches='tight')
    plt.close()
    


ax1 = pa_ctl.plot(y='PAC',x='age', c='b', kind='scatter', label='Unaff', title=ttl + '\n' + reglbl, s=2)
pa_alc.plot(y='PAC',x='age', c='r', kind='scatter',label='AUD', ax=ax1, s= 2)
plt.xlim([10,75])
plt.ylim([0,255])
plt.show()

pa_alc.plot(y='PAC',x='age', c='r', kind='scatter',label='AUD', s= 2, title=ttl + '\n' + reglbl)
plt.xlim([10,75])
plt.ylim([0,255])
plt.show()

pa_ctl.plot(y='PAC',x='age', c='b', kind='scatter',label='unaff', s= 2, title=ttl + '\n' + reglbl)
plt.xlim([10,75])
plt.ylim([0,255])
plt.show()

                  
X = pa_alc.age.values #.reshape((-1, 1)) 
y = pa_alc.PAC.values 
slope, intercept, r_value, p_value, std_err = linregress(X,y)

print('AUD PAC x Age')                 
print('slope = ' + str(slope))
print('intercept = ' + str(intercept))
print('r_value = ' + str(r_value))
print('std_err = ' + str(std_err))
print('r^2 = ' + str((r_value**2)))
print('\n')

X = pa_ctl.age.values #.reshape((-1, 1)) 
y = pa_ctl.PAC.values 
slope, intercept, r_value, p_value, std_err = linregress(X,y)

print('Unaffected PAC x Age')
print('slope = ' + str(slope))
print('intercept = ' + str(intercept))
print('r_value = ' + str(r_value))
print('std_err = ' + str(std_err))
print('r^2 = ' + str((r_value**2)))


# bins = 20
# minpac = int(round(min(pac_age.PAC),0))-1
# maxpac = int(round(max(pac_age.PAC),0))+1
# pacrng = maxpac-minpac
# pacrng = 20
# bins = np.linspace(minpac, maxpac, pacrng)
bins = np.linspace(155, 220, 30)
lgd = [ 'AUD', 'Unaff']
ax2 = pa_alc.PAC.hist(bins=bins,legend=False, histtype='step', color='blue') #, alpha=1, edgecolor='blue', linewidth=1)
pa_ctl.PAC.hist(bins=bins, ax=ax2, alpha=0.4, color='red') #, edgecolor='red', linewidth=1)
plt.legend(lgd)
plt.title(ttl)
plt.xlabel('PAC')
plt.ylabel('Counts')
plt.show()


paa = pd.concat([pa_alc,pa_ctl]).reset_index(drop=True)  
formula = 'PAC ~ C(AUD)'
md = smf.mixedlm(formula, paa, groups=paa["ID"], re_formula='ID:age')
mdf = md.fit()
# mdf = md.fit(method=["lbfgs"])
print(mdf.summary())


# paa = pa_alc[pa_alc.visit>0]
# paa = pa_ctl[pa_ctl.visit>1]
# paa = pa_ctl.copy()
# paa = pa_alc.copy()
# sbs = pd.unique(paa.ID)
# lines = []
# for s in sbs:
#     tmp = paa[paa.ID==s]
#     # if len(tmp)>1:
#     # tmp = paa[paa.ID==sbs[]]
#     x = tmp['age'].to_list()
#     y = tmp.PAC.to_list()
#     # z = np.vstack((x, y)).T
#     z = list(zip(x,y))
#     if len(z)<2:
#         z.append(tuple([z[0][0],int(z[0][1])]))
#     lines.append(z)
# cmap = plt.get_cmap('viridis')
# fig, ax = plt.subplots(1)
# line_segments = LineCollection(lines, cmap=cmap)
# ax.add_collection(line_segments)
# ax.autoscale_view()
# plt.xlim([10,40])
# plt.ylim([150,220])
# plt.show()



# subjects = pd.unique(pa_ctl.ID)
# i = 0
# ax1 = pa_ctl[pa_ctl.ID==subjects[595]].plot(y='PAC',x='age', c='b', kind='line', label='Unaff', title=ttl + '\n' + reglbl)
# plt.xlim([10,75])
# plt.ylim([100,220])
# while not(i==len(subjects)):
#     thissub = pa_ctl[pa_ctl.ID==subjects[i]]
#     if len(thissub)>0: 
#         thissub.plot(y='PAC',x='age', c='r', kind='line',label='AUD', ax=ax1)
#     i=i+1
#     print(str(i))
# plt.show()


# subjects = pd.unique(pa_alc.ID)
# i = 0                            
# while not(i==len(subjects)):
#     thissub = []
#     while len(thissub)==0: 
#         i=i+1
#         thissub = pa_ctl[pa_ctl.ID==subjects[i]]
#         thissub.plot(y='PAC',x='age', c='r', kind='line',label='AUD', ax=ax1)
    
    

