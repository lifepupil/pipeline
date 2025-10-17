# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 15:34:54 2025

@author: lifep
"""
from scipy import stats

from matplotlib import pyplot as plt 
import pandas as pd
import numpy as np

#  TO DO - GENERATE HISTOGRAM USING EACH OF THE P-A FREQUENCY DOMAINS SO DISTRIBUTION REFLECTS WHAT WE ANALYZED
rng = (0,1) # (0,255) (0,1) 78,248
do_logscale = True


data_info = '_src_new_pac_fz_AVG_0_3' # _0_3_NOBORDER _new_pac_fz_AVG_0_3 _src_new_pac_fz_AVG_0_3
# sex = 'M'

info_fn = 'pac_info_ages_25_50_AUD__ALL_0_11' + data_info + '.pkl'
pac_fn = 'pac_3d_ages_25_50_AUD__ALL_0_11' + data_info + '.npy'
base_dir = 'C:\\Users\\lifep\\OneDrive\\Documents\\'

pac_all = pd.read_pickle(base_dir + info_fn)
images_all = np.load(base_dir + pac_fn)

mcsf = pd.read_pickle('matched_cases_F.pkl')
mctf = pd.read_pickle('matched_controls_F.pkl')
mcsm = pd.read_pickle('matched_cases_M.pkl')
mctm = pd.read_pickle('matched_controls_M.pkl')
pac_age = pd.concat([pac_all.loc[mcsf.index],pac_all.loc[mctf.index],pac_all.loc[mcsm.index],pac_all.loc[mctm.index]])

alc_i = np.where(pac_age.AUD==1)
unaff_i = np.where(pac_age.AUD==0)

match_i = pac_age.index

# images = np.load('C:\\Users\\lifep\\OneDrive\\Documents\\pac_3d_' + targ_folder + '.npy')
images = images_all[match_i]

if rng[1]==1:
    images = images/255
    # all_pacs = all_pacs/255
    # valsaud = valsaud/255
    # valsunaf = valsunaf/255

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pac_len = 224
border_tb = 7
border_rl = 5
phase_start_hz = 0.1
phase_end_hz = 13
amp_start_hz = 4
amp_end_hz = 50


iwb = images[:,border_tb:pac_len-border_tb,border_rl:pac_len-border_rl].copy()
# iwb = images[:,6:218,5:219].copy()


lenx = np.shape(iwb)[2]
leny = np.shape(iwb)[1]
# BETTER WAY TO MAKE X AND Y AXIS LABELS AND TICKS
altxx = np.arange(phase_start_hz,phase_end_hz,((phase_end_hz-phase_start_hz)/lenx))
altyy = np.arange(amp_start_hz,amp_end_hz,((amp_end_hz-amp_start_hz)/leny))
altyy = altyy[::-1]

PAS = [
       [0.1,13,13,50],
       [0.1,8,8,12.99]
       ]

all_pacs = []
for pa in PAS:
    fpl = pa[0]
    fph = pa[1]
    fal = pa[2]
    fah = pa[3]
    fp_lo = np.where(altxx==altxx[(np.abs(altxx - fpl)).argmin()])[0][0]
    fp_hi = np.where(altxx==altxx[(np.abs(altxx - fph)).argmin()])[0][0]
    fa_lo = np.where(altyy==altyy[(np.abs(altyy - fal)).argmin()])[0][0]
    fa_hi = np.where(altyy==altyy[(np.abs(altyy - fah)).argmin()])[0][0]
    # plt.imshow(iwb[0,fa_hi:fa_lo,fp_lo:fp_hi])
    vals = iwb[:,fa_hi:fa_lo,fp_lo:fp_hi].flatten()
    all_pacs = all_pacs + vals.tolist()
   
all_pacs = np.array(all_pacs)
all_pacs = all_pacs.flatten()

vals_avg = np.mean(all_pacs)
vals_std = np.std(all_pacs)
vals_mdn = np.median(all_pacs)
all_pacs = np.array(all_pacs)


imaud = images[alc_i]
iwbaud = imaud[:,border_tb:pac_len-border_tb,border_rl:pac_len-border_rl].copy()
alc_pacs = []
for pa in PAS:
    fpl = pa[0]
    fph = pa[1]
    fal = pa[2]
    fah = pa[3]
    fp_lo = np.where(altxx==altxx[(np.abs(altxx - fpl)).argmin()])[0][0]
    fp_hi = np.where(altxx==altxx[(np.abs(altxx - fph)).argmin()])[0][0]
    fa_lo = np.where(altyy==altyy[(np.abs(altyy - fal)).argmin()])[0][0]
    fa_hi = np.where(altyy==altyy[(np.abs(altyy - fah)).argmin()])[0][0]
    vals = iwbaud[:,fa_hi:fa_lo,fp_lo:fp_hi].flatten()
    alc_pacs = alc_pacs + vals.tolist()

alc_pacs = np.array(alc_pacs)
valsaud = alc_pacs.flatten()
valsaud_avg = np.mean(valsaud)
valsaud_std = np.std(valsaud)
valsaud_mdn = np.median(valsaud)

imunaf = images[unaff_i]
iwbunaf = imunaf[:,border_tb:pac_len-border_tb,border_rl:pac_len-border_rl].copy()
ctl_pacs = []
for pa in PAS:
    fpl = pa[0]
    fph = pa[1]
    fal = pa[2]
    fah = pa[3]
    fp_lo = np.where(altxx==altxx[(np.abs(altxx - fpl)).argmin()])[0][0]
    fp_hi = np.where(altxx==altxx[(np.abs(altxx - fph)).argmin()])[0][0]
    fa_lo = np.where(altyy==altyy[(np.abs(altyy - fal)).argmin()])[0][0]
    fa_hi = np.where(altyy==altyy[(np.abs(altyy - fah)).argmin()])[0][0]
    vals = iwbunaf[:,fa_hi:fa_lo,fp_lo:fp_hi].flatten()
    ctl_pacs = ctl_pacs + vals.tolist()

ctl_pacs = np.array(ctl_pacs)

valsunaf = ctl_pacs.flatten()

valsunaf_avg = np.mean(valsunaf)
valsunaf_std = np.std(valsunaf)
valsunaf_mdn = np.median(valsunaf)

    
print('minimum PAC = ' + str(np.min(all_pacs)))
print('maximum PAC = ' + str(all_pacs.max()))
print('mean PAC = ' + str(vals_avg))
print('stdev PAC = ' + str(vals_std))
print('median PAC = ' + str(vals_mdn))


plt.hist(all_pacs, log=do_logscale)
plt.xlim(rng)
plt.xlabel('PAC')
plt.ylabel('Number of participants')
plt.show()
plt.close()

mins = np.array([(np.floor(np.min(all_pacs)*100))])
maxs = np.array([(np.ceil(np.max(all_pacs)*100))])
lo = mins.min()
hi = maxs.max()
hilolbl = 'high: ' + str(hi) +' - low: ' + str(lo) + ' = ' + str(hi-lo) + ' '
binnum = int(hi-lo + 1)
bins = np.linspace(lo/100,hi/100,binnum)     
ulbl = 'all' 
plt.hist(all_pacs,label=ulbl, alpha=0.8, bins=bins, color='black', histtype='step', log=do_logscale) #, edgecolor='k')
plt.title('all PAC within target domains')
plt.xlabel('PAC')
plt.ylabel('Number of participants')
plt.legend()
plt.show()
plt.close()


print('AUD')
print('minimum PAC ' + str(rng[0]) + '-' + str(rng[1]) + ' = ' + str(valsaud.min()))
print('maximum PAC ' + str(rng[0]) + '-' + str(rng[1]) + ' = ' + str(valsaud.max()))
print('mean PAC = ' + str(valsaud_avg))
print('stdev PAC = ' + str(valsaud_std))
print('median PAC = ' + str(valsaud_mdn))
print('UNAFFECTED')
print('minimum PAC ' + str(rng[0]) + '-' + str(rng[1]) + ' = ' + str(valsunaf.min()))
print('maximum PAC ' + str(rng[0]) + '-' + str(rng[1]) + ' = ' + str(valsunaf.max()))
print('mean PAC = ' + str(valsunaf_avg))
print('stdev PAC = ' + str(valsunaf_std))
print('median PAC = ' + str(valsunaf_mdn))

plt.hist(valsunaf, bins=255, alpha=0.5,label='UNAFF', log=do_logscale)
plt.hist(valsaud, bins=255, alpha=0.5,label='AUD', log=do_logscale)
plt.xlim(rng)
plt.xlabel('PAC')
plt.ylabel('Number of participants')
plt.legend()
plt.show()
plt.close()








# iwbnormed, lmda = stats.yeojohnson(vals)
# plt.hist(iwbnormed, bins=40, log=True)
# plt.title('ALL ' + ' lambda=' + str(lmda))
# # plt.xlim([0,255])
# plt.show()


# iwbnormed, lmda = stats.boxcox(vals)
# lmda = 3.3807041197994443
# (vals**lmda - 1)/lmda
# array([18089111.26442902, 16046250.31743524, 18089111.26442902, ...,
#        19335198.39007009, 17786660.51769949, 18089111.26442902])
# iwbnormed
# array([18089111.26442899, 16046250.31743527, 18089111.26442899, ...,
#        19335198.3900701 , 17786660.51769947, 18089111.26442899])
# stats.normaltest(iwbnormed)
# Out[47]: NormaltestResult(statistic=4052.5435467443863, pvalue=0.0)

# svraud = pac_all[(pac_all.audcnt>=6) & (pac_all.sex==sex)].index
# iwbaud = images_all[svraud,6:218,5:219].copy()
# vals = iwbaud.flatten()
# vals, lmda = stats.boxcox(vals)
# plt.hist(vals, bins=40, log=True)
# plt.title('AUD ' + sex + ' lambda=' + str(lmda))
# # plt.xlim([0,255])
# plt.show()

# unaf = pac_all[(pac_all.AUD==0) & (pac_all.sex==sex)].index
# iwbunaf = images_all[unaf,6:218,5:219].copy()
# vals = iwbunaf.flatten()
# vals, lmda = stats.boxcox(vals)
# plt.hist(vals, bins=40, log=True)
# plt.title('UNAFF ' + sex + ' lambda=' + str(lmda))
# # plt.xlim([0,255])
# plt.show()


