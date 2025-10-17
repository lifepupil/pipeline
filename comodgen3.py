# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 16:37:34 2025

@author: lifep
"""



import numpy as np
# from PIL import Image
import coga_support_defs as csd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, sem, false_discovery_control, linregress, f_oneway #, ttest_ind
import pingouin as pg
import os
from age_matcher import AgeMatcher
import matplotlib.patches as patches

SEX = '   m   '.upper().strip()
# do_other_sex_clusters = True # False True
do_other_sex_clusters = False # False True
age_groups = [25,50]
vmin = -4
vmax = 4
vmn = 184
vmx = 196
shuf_seeds = 424242
severity_scores = [6,11,'SEVERE']
alpha = 0.05
which_dx = 'AUD' # AUD ALAB ALD
aud_symp = 'audcnt' # audcnt ald5sx_max_cnt
channel = 'FZ'
alpha_test = 0.05

load_age_match = True
do_age_match = False
# load_age_match = False
# do_age_match = True
do_all = False # USE THIS TO GENERATE PAC DISTRIBUTIONS AND FOR BIG LINEAR MIXED EFFECT MODELS
# img_rows = 212
# img_cols = 214
img_rows = 224
img_cols = 224
data_info = '_src_new_pac_fz_AVG_0_3' # _0_3_NOBORDER _new_pac_fz_AVG_0_3 _src_new_pac_fz_AVG_0_3
image_format = 'png'
pac_len = 224
# TO REMOVE BORDER SET BELOW TO NON-ZERO INTEGER VALUES CORRESPONDING 
# TO NUMBER OF ROWS OR COLUMNS TO REMOVE FROM MATRIX
border_rl = 6
border_tb = 6
phase_start_hz = 0.1
phase_end_hz = 13
amp_start_hz = 4
amp_end_hz = 50

load_new_dat = True # False True
info_fn = 'pac_info_ages_25_50_AUD__ALL_0_11' + data_info + '.pkl'
pac_fn = 'pac_3d_ages_25_50_AUD__ALL_0_11' + data_info + '.npy'
# base_dir = 'D:\\COGA_eec\\' 
base_dir = 'C:\\Users\\lifep\\OneDrive\\Documents\\'
which_pacdat = 'D:\\COGA_eec\\pacdat_MASTER.pkl' 

fldrname = 'COMODULOGRAMS'

    
# PAC DOMAINS - WOMEN
FPAS = [
        [0.5,1.5,9,13],
        [1.25,2.25,28,34],
        [2.5,3.5,26,40],
        [3.25,4.25,21,25],
        [3,4.5,8,12],
        [3.5,4.5,12,15],
        [4,5,40,47],
        [4.25,5.25,35,40],
        [5.5,6.5,10,13],
        # [7,8,14,17],
        [7,8.5,14,17],
        # [8,9,14,17],
        [10.5,11.5,35,43],
        [12,13,30,42]
        ]
# PAC DOMAINS - MEN
MPAS = [
        [1.8,2.8,35,48],
        [2,3,19,27],
        [2.5,3.5,7,9],
        [2.7,3.7,15,20],
        [3.5,4.5,11,14],
        [4,5,33,42],
        [6.5,7.5,35,38],
        [7.5,8.5,15,17],
        [8,9,20,22],
        # [8,11.5,18.5,22],
        [9,11,39,44],
        [10.5,11.5,18,21],
        [11,12.5,29,32]
        ]


PACdomainsStdFreqBand = [
    [8,13,28,50],
    [3,8,28,50],
    [0.1,3,28,50],
    [8,13,20,28],
    [3,8,20,28],   
    [0.1,3,20,28],
    [8,13,13,20],
    [3,8,13,20],  
    [0.1,3,13,20],
    [0.1,3,8,13],
    [3,8,8,13]
    ]


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

write_dir = 'C:\\Users\\lifep\\OneDrive\\PAC stats paper\\' + fldrname + '\\'
if not os.path.exists(write_dir):
    os.makedirs(write_dir) 
                
min_age = age_groups[0]
max_age = age_groups[1]


if (SEX=='F'):
    sexlbl = 'Women'
    if do_other_sex_clusters:
        PAS = MPAS
    else:
        PAS = FPAS

    
if (SEX=='M'):
    sexlbl = 'Men'
    if do_other_sex_clusters:
        PAS = FPAS
    else:
        PAS = MPAS

channelstr = channel.lower()
targ_folder = 'ages_' + str(min_age) + '_' + str(max_age) + '_' + SEX + '_' + severity_scores[2] + '_' + str(severity_scores[0]) + '_' + str(severity_scores[1])
# MAKE SURE THAT THE INDEXING IS IDENTICAL BETWEEN pac_all AND images_all 
# THEY MUST ALSO HAVE THE SAME LENGTH, E.G., 8781
if ( not(['images_all' in locals()][0]) ) | load_new_dat:
    images_all = np.load(base_dir + pac_fn)
    pac_all = pd.read_pickle(base_dir + info_fn)

if SEX=='':
    pac_all = pac_all[(pac_all.age>=min_age) & (pac_all.age<=max_age)]
else:
    pac_all = pac_all[(pac_all.age>=min_age) & (pac_all.age<=max_age) & (pac_all.sex==SEX)]  
    
    
    
if (load_age_match):
    matched_cases = pd.read_pickle('matched_cases_' + SEX + '.pkl')
    matched_controls = pd.read_pickle('matched_controls_' + SEX + '.pkl')
    # matched_cases = pd.read_pickle('matched_cases.pkl') 
    # matched_controls = pd.read_pickle('matched_controls.pkl')
    pac_age = pd.concat([matched_cases,matched_controls])
    
    # pval_mx = np.load('pval_mx.npy')
    # effect_mx = np.load('effect_mx.npy')
    # aud_mx = np.load('aud_mx.npy')
    # ctl_mx = np.load('ctl_mx.npy')
    
    pval_mx = np.load('pval_mx' + '_' + SEX +'.npy')
    effect_mx = np.load('effect_mx' + '_' + SEX +'.npy')
    aud_mx = np.load('aud_mx' + '_' + SEX +'.npy')
    ctl_mx = np.load('ctl_mx' + '_' + SEX +'.npy')

    aud_mx = aud_mx/255
    ctl_mx = ctl_mx/255
    
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

elif (not(do_age_match)) & do_all:
    pac_age = pac_all
    ttl = targ_folder + ' alc_' + str(len(pac_all[pac_all.AUD==True])) + ' unaff_' + str(len(pac_all[pac_all.AUD==False])) 


match_i = pac_age.index
# images = np.load('C:\\Users\\lifep\\OneDrive\\Documents\\pac_3d_' + targ_folder + '.npy')
# images = images_all[match_i]
images = images_all[match_i]/255

# THIS CODE SNIPPET IS TO CHECK WHETHER THE SYMPTOM COUNTS FOR DSM4 AND DSM5 ARE THE SAME OR NOT
# THIS DATA CHECK TO SEE WHETHER CAN USE DSM4 SYMPTOMS FOR CORRELATIONS WITH PAC CLUSTERS
pac_age['dsm4cnt'] = 0
pac_age['dsmdif'] = 0
pac_age['ald5sx_max_cnt'] = pac_age.pop('ald5sx_max_cnt')

for i in range(len(pac_age)): 
    pac_age.loc[pac_age.iloc[[i]].index,'dsm4cnt'] = sum(pac_age.iloc[i,13:24]==5)
    pac_age.loc[pac_age.iloc[[i]].index,'dsmdif'] = pac_age.loc[pac_age.iloc[[i]].index,'audcnt'] - pac_age.loc[pac_age.iloc[[i]].index,'dsm4cnt']
    
pac_age['audcnt'] = pac_age.pop('audcnt')
pac_age['dsmdif'] = pac_age.pop('dsmdif')
perc0 = str(round((len(pac_age.loc[pac_age['dsmdif']==0]) / len(pac_age[pac_age['AUD']==1]))*100,1))
pac_age['dsmdif'].hist()
plt.title('Diffs DSM4 vs DSM5 AUD symptom counts (' +  perc0 + '% identical)')
plt.show()
plt.close()

pacdat = pd.read_pickle(which_pacdat)
df_list = []
pacdat = pacdat[(pacdat.channel=='FZ') & (pacdat.age_this_visit>=25) & (pacdat.age_this_visit<=50) & (pacdat.sex_x==SEX)]
for i in range(0,len(pac_age)):
    tmp = pacdat[(pacdat.ID==pac_age.iloc[i].ID) & (pacdat.age_this_visit==pac_age.iloc[i].age)].copy()
    df_list.append(tmp)
pac_df = pd.concat(df_list)
pac_df.rename(columns={'sex_x' : 'sex'}, inplace=True)
pac_df.rename(columns={'race_x' : 'race'}, inplace=True)
csd.print_demo_vals(pac_df)


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

if not(load_age_match):
    pval_mx = np.zeros((img_rows,img_cols))
    effect_mx = np.zeros((img_rows,img_cols))
    aud_mx = np.zeros((224,224))
    # aud_sem_mx = np.zeros((224,224))
    ctl_mx = np.zeros((224,224))
    # ctl_sem_mx = np.zeros((224,224))
    
    
    print('\ndoing statistics on all PAC frequency pairs')
    # images = np.load('C:\\Users\\lifep\\OneDrive\\Documents\\pac_3d_' + targ_folder + '.npy')
    images = images_all[match_i]
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
            # effect_mx[x,y] = np.median(alc_pac) -  np.median(unaff_pac)
            aud_mx[x,y] = np.median(alc_pac)
            # aud_sem_mx[x,y] = np.std(alc_pac)/np.sqrt(len(alc_pac))
            ctl_mx[x,y] = np.median(unaff_pac)
            # ctl_sem_mx[x,y] = np.std(unaff_pac)/np.sqrt(len(unaff_pac))
    np.save('pval_mx' + '_' + SEX +'.npy', pval_mx)
    np.save('effect_mx' + '_' + SEX +'.npy', effect_mx)
    np.save('aud_mx' + '_' + SEX +'.npy', aud_mx)
    np.save('ctl_mx' + '_' + SEX +'.npy', ctl_mx)
      


# TRIM BORDER
# padd = np.full(shape=224, fill_value=255)
# pval_mx = np.column_stack((pval_mx, padd))
# effect_mx = np.column_stack((effect_mx, padd))
# # aud_mx = np.load('aud_mx' + '_' + SEX +'.npy')
# aud_mx = np.column_stack((aud_mx, padd))
# ctl_mx = np.column_stack((ctl_mx, padd))
newpaclen = aud_mx.shape[1]
bord_shift = newpaclen - pac_len

# plt.imshow(aud_mx, cmap='jet')
# plt.show()

p2 = pval_mx[border_tb:pac_len-border_tb,border_rl:newpaclen-(border_rl+bord_shift)].copy()
e2 = effect_mx[border_tb:pac_len-border_tb,border_rl:newpaclen-(border_rl+bord_shift)].copy()
aud2 = aud_mx[border_tb:pac_len-border_tb,border_rl:newpaclen-(border_rl+bord_shift)].copy()
unaf2 = ctl_mx[border_tb:pac_len-border_tb,border_rl:newpaclen-(border_rl+bord_shift)].copy()

# plt.imshow(aud2, cmap='jet')
# plt.show()

# RESIZED X AND Y AXIS LABELS
xax = np.arange(phase_start_hz,phase_end_hz,((phase_end_hz - phase_start_hz)/(pac_len-border_tb*2)))
freq_pha = [str(round(x,2)) for x in xax]
yax = np.arange(amp_start_hz,amp_end_hz,((amp_end_hz - amp_start_hz)/(pac_len-border_tb*2)))
yax = yax[::-1]
freq_amp = [str(round(x,2)) for x in yax]

pv = pd.DataFrame(p2,  columns=freq_pha, index=freq_amp)
es = pd.DataFrame(e2,  columns=freq_pha, index=freq_amp)
audcomod = pd.DataFrame(aud2,  columns=freq_pha, index=freq_amp)
unafcomod = pd.DataFrame(unaf2,  columns=freq_pha, index=freq_amp)

# plt.imshow(audcomod, cmap='jet')
# plt.show()


# HERE WE ARE BUILDING THE AXIS TICK VALUES FOR COMODULOGRAMS
nn = np.array( es)
lenx = np.shape(nn)[1]
leny = np.shape(nn)[0]

# plt.imshow(aud2, cmap='jet')
# plt.colorbar(im, label='p-value\n' + '(minimum p = ' + str(round(pv.min(axis=None),4)))
# plt.title(ttl)
# # plt.xticks(my_xticks_i, my_xticks)
# # plt.yticks(my_xticks_i, my_xticks)
# plt.xlabel('Phase frequency (Hz)')
# plt.ylabel('Amplitude frequency (Hz)')
# plt.show()

# BETTER WAY TO MAKE X AND Y AXIS LABELS AND TICKS
altxx = np.arange(phase_start_hz,phase_end_hz,((phase_end_hz-phase_start_hz)/lenx))
altyy = np.arange(amp_start_hz,amp_end_hz,((amp_end_hz-amp_start_hz)/leny))
altyy = altyy[::-1]

my_xticks = np.array([0.1,2,4,6,8,10,12])
my_xticks_i = []
for fz in my_xticks:
    my_xticks_i.append(np.where(altxx==altxx[(np.abs(altxx - fz)).argmin()])[0][0])
my_xticks = [str(int(x)) for x in my_xticks]
my_xticks[0] = '0.1'

my_yticks = np.array([4,10,15,20,25,30,35,40,45,50])
my_yticks_i = []
for fz in my_yticks:
    my_yticks_i.append(np.where(altyy==altyy[(np.abs(altyy - fz)).argmin()])[0][0])
    
pdsfb = [
    [8,13,28,50],
    [3,8,28,50],
    [0.1,3,28,50],
    [8,13,20,28],
    [3,8,20,28],   
    [0.1,3,20,28],
    [8,13,13,20],
    [3,8,13,20],  
    [0.1,3,13,20],
    [0.1,3,8,13],
    [3,8,8,13]
    ]

xstdFreqs = np.sort(list(set([x[0] for x in pdsfb] + [x[1] for x in pdsfb])))
xstdFreqs_i = []
for fz in xstdFreqs:
    xstdFreqs_i.append(np.where(altxx==altxx[(np.abs(altxx - fz)).argmin()])[0][0])

ystdFreqs = np.sort(list(set([x[2] for x in pdsfb] + [x[3] for x in pdsfb])))
ystdFreqs_i = []
for fz in ystdFreqs:
    ystdFreqs_i.append(np.where(altyy==altyy[(np.abs(altyy - fz)).argmin()])[0][0])
    



# # DIFFERENCE COMODULOGRAM FIGURE 
# hm = sns.heatmap(es, 
#                  cmap="jet", 
#                  cbar_kws={'label': '\nPAC strength change\n(AUD - Unaff)'}, 
#                  vmin=vmin, vmax=vmax,
#                  xticklabels=False,
#                  yticklabels=False
#                  )
# hm.set_xlabel('Phase Frequency (Hz)', fontsize=18)
# hm.set_ylabel('Amplitude Frequency (Hz)\n', fontsize=18)
# hm.set_title(ttl)
# hm.set_xticks(ix_all, xtv_all)
# hm.set_yticks(iy_all, ytv_all)
# hm.tick_params(direction='out', labelsize=16, length=4, width=1, colors='k', bottom=True, left=True)
# hm.collections[0].colorbar.ax.tick_params(labelsize=16)
# hm.figure.axes[-1].yaxis.label.set_size(18)
# ax = plt.gca()
# for _,spine in ax.spines.items():
#     spine.set_visible(True)
# plt.show()
# output = plt.Axes.get_figure(hm)
# output.savefig(write_dir + ttl + '_EFFECTSIZE.png', bbox_inches='tight', format=image_format)
# plt.close(output)

# DIFFERENCE COMODULOGRAM FIGURE WITH CLUSTER DOMAINS MARKED
hm = sns.heatmap(es, 
                 cmap="jet", 
                 cbar_kws={'label': '\nPAC strength change\n(AUD - Unaff)'}, 
                 vmin=vmin, vmax=vmax,
                 xticklabels=False,
                 yticklabels=False
                 )
hm.set_xlabel('Phase Frequency (Hz)', fontsize=18)
hm.set_ylabel('Amplitude Frequency (Hz)\n', fontsize=18)
hm.set_title(ttl)
hm.set_xticks(my_xticks_i, my_xticks)
hm.set_yticks(my_yticks_i, my_yticks)
# hm.set_xticks(ix_all, xtv_all)
# hm.set_yticks(iy_all, ytv_all)


hm.tick_params(direction='out', labelsize=16, length=4, width=1, colors='k', bottom=True, left=True)
hm.collections[0].colorbar.ax.tick_params(labelsize=16)
hm.figure.axes[-1].yaxis.label.set_size(18)
ax = plt.gca()

    
# for pa in PACdomainsStdFreqBand:
#     fpl = pa[0]
#     fph = pa[1]
#     fal = pa[2]
#     fah = pa[3]
#     fp = pd.DataFrame([float(f) for f in freq_pha], columns=['freq'])
#     fp_lo = fp[(fp.freq>=fpl)].index[0]
#     fp_hi = fp[(fp.freq>=fph)].index[0]
#     fa = pd.DataFrame([float(f) for f in freq_amp], columns=['freq'])
#     fa_lo = fa[(fa.freq<=fal)].index[0]
#     fa_hi = fa[(fa.freq<=fah)].index[0]        
#     ax.add_patch(
#         patches.Rectangle(
#             (fp_lo,fa_hi), 
#             fp_hi-fp_lo, 
#             fa_lo-fa_hi,
#             edgecolor='white',
#             fill=False,
#             linestyle=':',
#             lw=2 ))
    
for pa in PAS:
    fpl = pa[0]
    fph = pa[1]
    fal = pa[2]
    fah = pa[3]
    # fp = pd.DataFrame([float(f) for f in freq_pha], columns=['freq'])
    # fp_lo = fp[(fp.freq>=fpl)].index[0]
    # fp_hi = fp[(fp.freq>=fph)].index[0]
    # fa = pd.DataFrame([float(f) for f in freq_amp], columns=['freq'])
    # fa_lo = fa[(fa.freq<=fal)].index[0]
    # fa_hi = fa[(fa.freq<=fah)].index[0]     
        
    fp = pd.DataFrame([float(f) for f in freq_pha], columns=['freq'])
    fp_lo = np.where(altxx==altxx[(np.abs(altxx - fpl)).argmin()])[0][0]
    fp_hi = np.where(altxx==altxx[(np.abs(altxx - fph)).argmin()])[0][0]
    fa = pd.DataFrame([float(f) for f in freq_amp], columns=['freq'])   
    fa_lo = np.where(altyy==altyy[(np.abs(altyy - fal)).argmin()])[0][0]
    fa_hi = np.where(altyy==altyy[(np.abs(altyy - fah)).argmin()])[0][0]
    ax.add_patch(
        patches.Rectangle(
            (fp_lo,fa_hi), 
            fp_hi-fp_lo, 
            fa_lo-fa_hi,
            edgecolor='black',
            fill=False,
                lw=2))
    
for _,spine in ax.spines.items():
    spine.set_visible(True)
plt.show()
output = plt.Axes.get_figure(hm)
output.savefig(write_dir + ttl + '_EFFECTSIZE_with_clusters.png', bbox_inches='tight', format=image_format)
plt.close(output)


# pv_fig = pv.copy()
# hmmin = str(round(np.min(pv_fig),6))
# pvf = sns.heatmap(pv_fig, 
#                  cmap="rocket_r", 
#                  cbar_kws={'label': 'p-value\n(min=' + hmmin + ')'},
#                  xticklabels=False,
#                  yticklabels=False
#                  )
# pvf.set_xlabel('Phase Frequency (Hz)', fontsize=18)
# pvf.set_ylabel('Amplitude Frequency (Hz)\n', fontsize=18)
# pvf.set_title(ttl)
# pvf.set_xticks(ix_all, xtv_all)
# pvf.set_yticks(iy_all, ytv_all)
# pvf.tick_params(direction='out', labelsize=16, length=4, width=1, colors='k', bottom=True, left=True)
# pvf.collections[0].colorbar.ax.tick_params(labelsize=16)
# pvf.figure.axes[-1].yaxis.label.set_size(18)
# ax = plt.gca()
# for _,spine in ax.spines.items():
#     spine.set_visible(True)
# plt.show()
# plt.close()



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
pvf.set_xlabel('Phase Frequency (Hz)', fontsize=16)
pvf.set_ylabel('Amplitude Frequency (Hz)\n', fontsize=16)
pvf.set_title(ttl)
# pvf.set_xticks(ix_all, xtv_all)
# pvf.set_yticks(iy_all, ytv_all)
pvf.set_xticks(my_xticks_i, my_xticks)
pvf.set_yticks(my_yticks_i, my_yticks)
pvf.tick_params(direction='out', labelsize=14, length=4, width=1, colors='k', bottom=True, left=True)
pvf.collections[0].colorbar.ax.tick_params(labelsize=14)
pvf.figure.axes[-1].yaxis.label.set_size(16)
ax = plt.gca()
# NOW WE ADD RECTANGLES SURROUNDING THE PAC DOMAINS BEING ANALZED 
for pa in PAS:
    fpl = pa[0]
    fph = pa[1]
    fal = pa[2]
    fah = pa[3]
    # fp = pd.DataFrame([float(f) for f in freq_pha], columns=['freq'])
    # fp_lo = fp[(fp.freq>=fpl)].index[0]
    # fp_hi = fp[(fp.freq>=fph)].index[0]
    # fa = pd.DataFrame([float(f) for f in freq_amp], columns=['freq'])
    # fa_lo = fa[(fa.freq<=fal)].index[0]
    # fa_hi = fa[(fa.freq<=fah)].index[0]      
    
    fp = pd.DataFrame([float(f) for f in freq_pha], columns=['freq'])
    fp_lo = np.where(altxx==altxx[(np.abs(altxx - fpl)).argmin()])[0][0]
    fp_hi = np.where(altxx==altxx[(np.abs(altxx - fph)).argmin()])[0][0]
    fa = pd.DataFrame([float(f) for f in freq_amp], columns=['freq'])   
    fa_lo = np.where(altyy==altyy[(np.abs(altyy - fal)).argmin()])[0][0]
    fa_hi = np.where(altyy==altyy[(np.abs(altyy - fah)).argmin()])[0][0]

    ax.add_patch(
        patches.Rectangle(
            (fp_lo,fa_hi), 
            fp_hi-fp_lo, 
            fa_lo-fa_hi,
            edgecolor='white',
            fill=False,
            lw=1 ))
for pa in PACdomainsStdFreqBand:
    fpl = pa[0]
    fph = pa[1]
    fal = pa[2]
    fah = pa[3]
    fp = pd.DataFrame([float(f) for f in freq_pha], columns=['freq'])
    fp_lo = np.where(altxx==altxx[(np.abs(altxx - fpl)).argmin()])[0][0]
    fp_hi = np.where(altxx==altxx[(np.abs(altxx - fph)).argmin()])[0][0]
    # fp_lo = fp[(fp.freq>=fpl)].index[0]
    # fp_hi = fp[(fp.freq>=fph)].index[0]
    fa = pd.DataFrame([float(f) for f in freq_amp], columns=['freq'])
    # fa_lo = fa[(fa.freq<=fal)].index[0]
    # fa_hi = fa[(fa.freq<=fah)].index[0]        
    fa_lo = np.where(altyy==altyy[(np.abs(altyy - fal)).argmin()])[0][0]
    fa_hi = np.where(altyy==altyy[(np.abs(altyy - fah)).argmin()])[0][0]
    ax.add_patch(
        patches.Rectangle(
            (fp_lo,fa_hi), 
            fp_hi-fp_lo, 
            fa_lo-fa_hi,
            edgecolor='green',
            fill=False,
            linestyle=':',
            lw=1 ))

for _,spine in ax.spines.items():
    spine.set_visible(True)    
output = plt.Axes.get_figure(pvf)
plt.show()
output.savefig(write_dir + ttl + '-PVALUES.png', bbox_inches='tight', format=image_format)
plt.close(output)


# pv_fig = pv.copy()
# pv_fig[pv_fig>alpha] = alpha
# hmmin = str(round(np.min(pv_fig),6))
# pvf = sns.heatmap(pv_fig, 
#                  vmax=0.05,
#                  cmap="rocket_r", 
#                  cbar_kws={'label': 'p-value\n(min=' + hmmin + ')'},
#                  xticklabels=False,
#                  yticklabels=False
#                  )
# pvf.set_xlabel('Phase Frequency (Hz)', fontsize=18)
# pvf.set_ylabel('Amplitude Frequency (Hz)\n', fontsize=18)
# pvf.set_title(ttl)
# pvf.set_xticks(ix_all, xtv_all)
# pvf.set_yticks(iy_all, ytv_all)
# pvf.tick_params(direction='out', labelsize=16, length=4, width=1, colors='k', bottom=True, left=True)
# pvf.collections[0].colorbar.ax.tick_params(labelsize=16)
# pvf.figure.axes[-1].yaxis.label.set_size(18)
# ax = plt.gca()
# # NOW WE ADD RECTANGLES SURROUNDING THE PAC DOMAINS BEING ANALZED 
# for pa in FPAS:    
#     fpl = pa[0]
#     fph = pa[1]
#     fal = pa[2]
#     fah = pa[3]
#     fp = pd.DataFrame([float(f) for f in freq_pha], columns=['freq'])
#     fp_lo = fp[(fp.freq>=fpl)].index[0]
#     fp_hi = fp[(fp.freq>=fph)].index[0]
#     fa = pd.DataFrame([float(f) for f in freq_amp], columns=['freq'])
#     fa_lo = fa[(fa.freq<=fal)].index[0]
#     fa_hi = fa[(fa.freq<=fah)].index[0]        
#     ax.add_patch(
#         patches.Rectangle(
#             (fp_lo,fa_hi), 
#             fp_hi-fp_lo, 
#             fa_lo-fa_hi,
#             edgecolor='white',
#             fill=False,
#             lw=1 ))
# for pa in MPAS:    
#     fpl = pa[0]
#     fph = pa[1]
#     fal = pa[2]
#     fah = pa[3]
#     fp = pd.DataFrame([float(f) for f in freq_pha], columns=['freq'])
#     fp_lo = fp[(fp.freq>=fpl)].index[0]
#     fp_hi = fp[(fp.freq>=fph)].index[0]
#     fa = pd.DataFrame([float(f) for f in freq_amp], columns=['freq'])
#     fa_lo = fa[(fa.freq<=fal)].index[0]
#     fa_hi = fa[(fa.freq<=fah)].index[0]        
#     ax.add_patch(
#         patches.Rectangle(
#             (fp_lo,fa_hi), 
#             fp_hi-fp_lo, 
#             fa_lo-fa_hi,
#             edgecolor='green',
#             fill=False,
#             lw=1 ))
# for _,spine in ax.spines.items():
#     spine.set_visible(True)    
# output = plt.Axes.get_figure(pvf)
# plt.show()
# output.savefig(write_dir + ttl + '-PVALUES_MFclusters.png', bbox_inches='tight', format=image_format)
# plt.close(output)





# im = plt.imshow(audcomod, cmap='jet')
# plt.colorbar(im, label='p-value\n' + '(minimum p = ' + str(round(pv.min(axis=None),4)))
# plt.title(ttl)
# # plt.xticks(my_xticks_i, my_xticks)
# # plt.yticks(my_xticks_i, my_xticks)
# plt.xlabel('Phase frequency (Hz)')
# plt.ylabel('Amplitude frequency (Hz)')
# plt.show()
# plt.close()


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
hmm.set_title(ttl)
hmm.set_xticks(my_xticks_i, my_xticks)
hmm.set_yticks(my_yticks_i, my_yticks)

hmm.tick_params(direction='out', labelsize=16, length=4, width=1, colors='k', bottom=True, left=True)
hmm.collections[0].colorbar.ax.tick_params(labelsize=16)
hmm.figure.axes[-1].yaxis.label.set_size(18)
ax = plt.gca()
# NOW WE ADD RECTANGLES SURROUNDING THE PAC DOMAINS BEING ANALZED 
for pa in PAS:
    fpl = pa[0]
    fph = pa[1]
    fal = pa[2]
    fah = pa[3]
      
        
    fp = pd.DataFrame([float(f) for f in freq_pha], columns=['freq'])
    fp_lo = np.where(altxx==altxx[(np.abs(altxx - fpl)).argmin()])[0][0]
    fp_hi = np.where(altxx==altxx[(np.abs(altxx - fph)).argmin()])[0][0]
    fa = pd.DataFrame([float(f) for f in freq_amp], columns=['freq'])   
    fa_lo = np.where(altyy==altyy[(np.abs(altyy - fal)).argmin()])[0][0]
    fa_hi = np.where(altyy==altyy[(np.abs(altyy - fah)).argmin()])[0][0]
    ax.add_patch(
        patches.Rectangle(
            (fp_lo,fa_hi), 
            fp_hi-fp_lo, 
            fa_lo-fa_hi,
            edgecolor='black',
            fill=False,
            lw=1 ))
for pa in PACdomainsStdFreqBand:
    fpl = pa[0]
    fph = pa[1]
    fal = pa[2]
    fah = pa[3]
    
        
    fp = pd.DataFrame([float(f) for f in freq_pha], columns=['freq'])
    fp_lo = np.where(altxx==altxx[(np.abs(altxx - fpl)).argmin()])[0][0]
    fp_hi = np.where(altxx==altxx[(np.abs(altxx - fph)).argmin()])[0][0]
    fa = pd.DataFrame([float(f) for f in freq_amp], columns=['freq'])   
    fa_lo = np.where(altyy==altyy[(np.abs(altyy - fal)).argmin()])[0][0]
    fa_hi = np.where(altyy==altyy[(np.abs(altyy - fah)).argmin()])[0][0]
    ax.add_patch(
        patches.Rectangle(
            (fp_lo,fa_hi), 
            fp_hi-fp_lo, 
            fa_lo-fa_hi,
            edgecolor='green',
            fill=False,
            linestyle=':',
            lw=1 ))

for _,spine in ax.spines.items():
    spine.set_visible(True)
plt.show()
output = plt.Axes.get_figure(hmm)
output.savefig(write_dir + ttl + '-EFFECTSIZExPVAL.png', bbox_inches='tight', format=image_format)
plt.close(output)




# 1. Group neighboring clusters where significant PAC found
# 2. PAC domains restricted as much as possible to where raw sig pvals found
# 3. If PAC differences within test domain are not significant then cluster rejected
# 4. If no spectral power at PAC frequencies then reject cluster?
# 5. Only clusters with raw sig pvals spanning at least 1 Hz width of phase or amplitude frequency (minimum size 1Hz x 1 Hz)
# 6. Must have FDR sig p in PAC domain where cluster is found otherwise reject
# 7. Lumping stage after splitting stage guided by where maximum PAC differences appear
# using these rules, we identified X number of clusters for further testing
if 1:
    
    
    # THIS BLOCK OF CODE GENERATES COMODULOGRAM FIGURE WHERE VALUES ARE DISPLAYED ON WITHIN PAC DOMAINS TARGETING PAC CLUSTERS
    # EXAMPLES HERE FOR HOW TO SELECTIVELY ZERO OUT OR PRESERVE 2-D MATRIX DATA FOR DISPLAY IN SCIENTIFIC FIGURES
    pac_domain_mask = np.zeros([pv.shape[0], pv.shape[1]])
    for pa in PACdomainsStdFreqBand:    
        fpl = pa[0]
        fph = pa[1]
        fal = pa[2]
        fah = pa[3]
            
        fp = pd.DataFrame([float(f) for f in freq_pha], columns=['freq'])
        fp_lo = np.where(altxx==altxx[(np.abs(altxx - fpl)).argmin()])[0][0]
        fp_hi = np.where(altxx==altxx[(np.abs(altxx - fph)).argmin()])[0][0]
        fa = pd.DataFrame([float(f) for f in freq_amp], columns=['freq'])   
        fa_lo = np.where(altyy==altyy[(np.abs(altyy - fal)).argmin()])[0][0]
        fa_hi = np.where(altyy==altyy[(np.abs(altyy - fah)).argmin()])[0][0]

        # print(str(fp_lo),str(fp_hi),str(fa_lo),str(fa_hi) ) 
        pac_domain_mask[fa_hi:fa_lo, fp_lo:fp_hi] = 1
    pac_domain_mask = pac_domain_mask.astype(bool)
    
    # pac_domain_diffs = e2[border_tb:pac_len-border_tb,border_rl:pac_len-border_rl].copy()
    pac_domain_diffs = e2.copy()
    pdd = np.multiply(pac_domain_diffs, pac_domain_mask)
    pddiffs_df = pd.DataFrame(pdd,  columns=freq_pha, index=freq_amp)
    
    pac_domain_pvals = p2.copy()
    pdp = np.multiply(pac_domain_pvals, pac_domain_mask)
    pdpvals_df = pd.DataFrame(pdp,  columns=freq_pha, index=freq_amp)
    
    pac_domain_aud = aud2.copy()
    pdpaud = np.multiply(pac_domain_aud, pac_domain_mask)
    pdpaudvals_df = pd.DataFrame(pdpaud,  columns=freq_pha, index=freq_amp)
    
    # NOW WE GENERATE EFFECT SIZE COMODULOGRAM WITH PAC CLUSTER DOMAINS HIGHLIGHTED pval_mx
    hmm = sns.heatmap(pdpaudvals_df,
				  cmap="jet", 
				  cbar_kws={'label': '\nAverage PAC strength'}, 
				  mask=(pdpvals_df==0), 
				  vmin=vmn/255, vmax=vmx/255,
				  xticklabels=False,
				  yticklabels=False
				  )
    audmap = sns.heatmap(audcomod, 
                     cmap="jet", 
                     cbar=None, 
                     mask=(pdpvals_df>0), 
                     vmin=vmn/255, vmax=vmx/255,
                     xticklabels=False,
                     yticklabels=False,
                     alpha=0.3
                     )
    audmap.set_xlabel('Phase Frequency (Hz)', fontsize=16)
    audmap.set_ylabel('Amplitude Frequency (Hz)\n', fontsize=16)
    # audmap.set_title('Median PAC for ' + SEX + ' - AUD ')
    audmap.set_xticks(xstdFreqs_i, xstdFreqs)
    audmap.set_yticks(ystdFreqs_i, ystdFreqs)
    audmap.tick_params(direction='out', labelsize=14, length=4, width=1, colors='k', bottom=True, left=True)
    audmap.collections[0].colorbar.ax.tick_params(labelsize=14)
    audmap.figure.axes[-1].yaxis.label.set_size(16)
    ax = plt.gca()
    for pa in PACdomainsStdFreqBand:
        fpl = pa[0]
        fph = pa[1]
        fal = pa[2]
        fah = pa[3]
        fp = pd.DataFrame([float(f) for f in freq_pha], columns=['freq'])
        fp_lo = np.where(altxx==altxx[(np.abs(altxx - fpl)).argmin()])[0][0]
        fp_hi = np.where(altxx==altxx[(np.abs(altxx - fph)).argmin()])[0][0]
        fa = pd.DataFrame([float(f) for f in freq_amp], columns=['freq'])   
        fa_lo = np.where(altyy==altyy[(np.abs(altyy - fal)).argmin()])[0][0]
        fa_hi = np.where(altyy==altyy[(np.abs(altyy - fah)).argmin()])[0][0]    
        ax.add_patch(
            patches.Rectangle(
                (fp_lo,fa_hi), 
                fp_hi-fp_lo, 
                fa_lo-fa_hi,
                edgecolor='black',
                fill=False,
                linestyle=':',
                lw=1 ))
    # hz4 = np.where(altxx==altxx[(np.abs(altxx - 4)).argmin()])[0][0]
    # hz3 = np.where(altxx==altxx[(np.abs(altxx - 3)).argmin()])[0][0]
    # plt.axvline(x=border_rl,color='red',linestyle='--', lw=0.5)
    # plt.axvline(x=224-border_rl,color='red',linestyle='--', lw=0.5)
    # plt.axhline(y=border_tb,color='red',linestyle='--', lw=0.5)
    # plt.axhline(y=224-border_tb,color='red',linestyle='--', lw=0.5)
    # plt.axvline(x=hz4,color='red',linestyle='--', lw=0.5)
    # plt.axvline(x=hz3,color='red',linestyle='--', lw=0.5)
    for _,spine in ax.spines.items():
        spine.set_visible(True)
    thisttl = 'Median PAC for ' + SEX + ' - AUD' # : 4hz at ' + str(hz4) + ', rl=' + str(border_rl) + ', tb=' + str(border_tb)
    plt.title(thisttl)
    plt.show()
    output = plt.Axes.get_figure(audmap)
    output.savefig(write_dir + ttl + '_AUD_comod.png', bbox_inches='tight', format=image_format)
    plt.close(output)
    
    
    
    pac_domain_unaf = unaf2.copy()
    pdpunaf = np.multiply(pac_domain_unaf, pac_domain_mask)
    pdpunafvals_df = pd.DataFrame(pdpunaf,  columns=freq_pha, index=freq_amp)
        
    hmm = sns.heatmap(pdpunafvals_df,
				  cmap="jet", 
				  cbar_kws={'label': '\nAverage PAC strength'}, 
				  mask=(pdpvals_df==0), 
				  vmin=vmn/255, vmax=vmx/255,
				  xticklabels=False,
				  yticklabels=False
				  )    
    unafmap = sns.heatmap(unafcomod, 
                     cmap="jet", 
                     cbar=None, 
                     mask=(pdpvals_df>0),
                     vmin=vmn/255, vmax=vmx/255,
                     xticklabels=False,
                     yticklabels=False,
                     alpha=0.3
                     )
    unafmap.set_xlabel('Phase Frequency (Hz)', fontsize=16)
    unafmap.set_ylabel('Amplitude Frequency (Hz)\n', fontsize=16)
    unafmap.set_title('Median PAC for ' + SEX + ' - unaffected ')
    # unafmap.set_xticks(ix_all, xtv_all)
    # unafmap.set_yticks(iy_all, ytv_all)
    unafmap.set_xticks(xstdFreqs_i, xstdFreqs)
    unafmap.set_yticks(ystdFreqs_i, ystdFreqs)
    unafmap.tick_params(direction='out', labelsize=14, length=4, width=1, colors='k', bottom=True, left=True)
    unafmap.collections[0].colorbar.ax.tick_params(labelsize=14)
    unafmap.figure.axes[-1].yaxis.label.set_size(16)
    ax = plt.gca()
    for pa in PACdomainsStdFreqBand:
        fpl = pa[0]
        fph = pa[1]
        fal = pa[2]
        fah = pa[3]
        fp = pd.DataFrame([float(f) for f in freq_pha], columns=['freq'])
        fp_lo = np.where(altxx==altxx[(np.abs(altxx - fpl)).argmin()])[0][0]
        fp_hi = np.where(altxx==altxx[(np.abs(altxx - fph)).argmin()])[0][0]
        fa = pd.DataFrame([float(f) for f in freq_amp], columns=['freq'])   
        fa_lo = np.where(altyy==altyy[(np.abs(altyy - fal)).argmin()])[0][0]
        fa_hi = np.where(altyy==altyy[(np.abs(altyy - fah)).argmin()])[0][0]
        # fp = pd.DataFrame([float(f) for f in freq_pha], columns=['freq'])
        # fp_lo = fp[(fp.freq>=fpl)].index[0]
        # fp_hi = fp[(fp.freq>=fph)].index[0]
        # fa = pd.DataFrame([float(f) for f in freq_amp], columns=['freq'])
        # fa_lo = fa[(fa.freq<=fal)].index[0]
        # fa_hi = fa[(fa.freq<=fah)].index[0]        
        ax.add_patch(
            patches.Rectangle(
                (fp_lo,fa_hi), 
                fp_hi-fp_lo, 
                fa_lo-fa_hi,
                edgecolor='black',
                fill=False,
                linestyle=':',
                lw=1 ))
    for _,spine in ax.spines.items():
        spine.set_visible(True)
    plt.show()
    output = plt.Axes.get_figure(unafmap)
    output.savefig(write_dir + ttl + '_CTL_comod.png', bbox_inches='tight', format=image_format)
    plt.close(output)
    
    # hmm = sns.heatmap(pddiffs_df,
    #                   cmap="jet", 
    #                   cbar_kws={'label': 'PAC strength change\n(AUD - unaff)'}, 
    #                   mask=(pdpvals_df==0), 
    #                   vmin=vmin, vmax=vmax,
    #                   xticklabels=False,
    #                   yticklabels=False
    #                   )
    # sns.heatmap(es,
    #             cmap="jet", 
    #             cbar=None,
    #             mask=(pdpvals_df>0), 
    #             vmin=vmin, vmax=vmax,
    #             xticklabels=False,
    #             yticklabels=False,
    #             alpha=0.4
    #             )
    # hmm.set_xlabel('Phase Frequency (Hz)', fontsize=15)
    # hmm.set_ylabel('Amplitude Frequency (Hz)\n', fontsize=15)
    # hmm.set_xticks(ix, xtv)
    # hmm.set_yticks(iy, ytv)
    # hmm.set_title(ttl)
    # hmm.tick_params(direction='out', labelsize=13, length=4, width=1, colors='k', bottom=True, left=True)
    # hmm.collections[0].colorbar.ax.tick_params(labelsize=13)
    # # hmm.figure.axes[-1].yaxis.label.set_size(8)
    # # hmm.figure.axes[-1].tick_params(labelsize=22)
    # ax = plt.gca()
    # # NOW WE ADD RECTANGLES SURROUNDING THE PAC DOMAINS BEING ANALZED 
    # for pa in PACdomainsStdFreqBand:    
    #     fpl = pa[0]
    #     fph = pa[1]
    #     fal = pa[2]
    #     fah = pa[3]
    #     fp = pd.DataFrame([float(f) for f in freq_pha], columns=['freq'])
    #     fp_lo = fp[(fp.freq>=fpl)].index[0]
    #     fp_hi = fp[(fp.freq>=fph)].index[0]
    #     fa = pd.DataFrame([float(f) for f in freq_amp], columns=['freq'])
    #     fa_lo = fa[(fa.freq<=fal)].index[0]
    #     fa_hi = fa[(fa.freq<=fah)].index[0]        
    #     ax.add_patch(
    #         patches.Rectangle(
    #             (fp_lo,fa_hi), 
    #             fp_hi-fp_lo, 
    #             fa_lo-fa_hi,
    #             edgecolor='white',
    #             fill=False,
    #             lw=2 ))
    # for _,spine in ax.spines.items():
    #     spine.set_visible(True)
    # cba = ax.figure.axes[-1] # LAST ELEMENT IN THIS ARRAY USUALLY HAS COLOR BAR
    # cba.yaxis.label.set_size(10)
    # cba.tick_params(labelsize=12)
    # plt.show()
    # output = plt.Axes.get_figure(hmm)
    # output.savefig(write_dir + ttl + '_diff_comod_with_PAC_cluster_domains.png', bbox_inches='tight', format=image_format)
    # plt.close(output)
    
    # # THEN PVALUE COMODULOGRAM OF PAC CLUSTER DOMAINS
    # hmm = sns.heatmap(pdpvals_df,
    #                   cmap="jet", 
    #                   cbar_kws={'label': 'PAC strength change\n(AUD - unaff)'}, 
    #                   mask=(pdpvals_df==0), 
    #                   # vmin=vmin, vmax=vmax,
    #                   xticklabels=False,
    #                   yticklabels=False
    #                   )
    # sns.heatmap(pv,
    #             cmap="jet", 
    #             cbar=None,
    #             mask=(pdpvals_df>0), 
    #             # vmin=vmin, vmax=vmax,
    #             xticklabels=False,
    #             yticklabels=False,
    #             alpha=0.3
    #             )
    # hmm.set_xlabel('Phase Frequency (Hz)', fontsize=15)
    # hmm.set_ylabel('Amplitude Frequency (Hz)\n', fontsize=15)
    # hmm.set_xticks(ix, xtv)
    # hmm.set_yticks(iy, ytv)
    # hmm.set_title(ttl)
    # hmm.tick_params(direction='out', labelsize=13, length=4, width=1, colors='k', bottom=True, left=True)
    # hmm.collections[0].colorbar.ax.tick_params(labelsize=13)
    # # hmm.figure.axes[-1].yaxis.label.set_size(8)
    # # hmm.figure.axes[-1].tick_params(labelsize=22)
    # ax = plt.gca()
    # # NOW WE ADD RECTANGLES SURROUNDING THE PAC DOMAINS BEING ANALZED 
    # for pa in PACdomainsStdFreqBand:    
    #     fpl = pa[0]
    #     fph = pa[1]
    #     fal = pa[2]
    #     fah = pa[3]
    #     fp = pd.DataFrame([float(f) for f in freq_pha], columns=['freq'])
    #     fp_lo = fp[(fp.freq>=fpl)].index[0]
    #     fp_hi = fp[(fp.freq>=fph)].index[0]
    #     fa = pd.DataFrame([float(f) for f in freq_amp], columns=['freq'])
    #     fa_lo = fa[(fa.freq<=fal)].index[0]
    #     fa_hi = fa[(fa.freq<=fah)].index[0]        
    #     ax.add_patch(
    #         patches.Rectangle(
    #             (fp_lo,fa_hi), 
    #             fp_hi-fp_lo, 
    #             fa_lo-fa_hi,
    #             edgecolor='red',
    #             fill=False,
    #             lw=2 ))
    # for _,spine in ax.spines.items():
    #     spine.set_visible(True)
    # cba = ax.figure.axes[-1] # LAST ELEMENT IN THIS ARRAY USUALLY HAS COLOR BAR
    # cba.yaxis.label.set_size(10)
    # cba.tick_params(labelsize=12)
    # plt.show()
    # output = plt.Axes.get_figure(hmm)
    # output.savefig(write_dir + ttl + '_diff_comod_with_PAC_cluster_domains.png', bbox_inches='tight', format=image_format)
    # plt.close(output)


    #     # PAS = [[9,11,40,43],
    #     #        [11,12,29,32],
    #     #        [10.75,11.25,18,20],
    #     #        [9,10,19,20],
    #     #        [8,9,20,22],
    #     #        [7,8,21,23],
    #     #        [4,5,33,40],
    #     #        [3,5,12,15],
    #     #        [2,4,6,10],
    #     #        [0.5,2,28,47]]

    # NOW WE START ANALYZES FOR EACH CLUSTER 
    for pa in PAS:  
        # pa = [7,12,18,23]
        fpl = pa[0]
        fph = pa[1]
        fal = pa[2]
        fah = pa[3]
        
        # GET INDICES FOR PHASE AND AMPLITUDE FREQUENCIES TO DO PAC REGION STATISTICS
        # fp = pd.DataFrame([float(f) for f in freq_pha], columns=['freq'])
        # fp_lo = fp[(fp.freq>=fpl)].index[0]
        # fp_hi = fp[(fp.freq>=fph)].index[0]
        # fa = pd.DataFrame([float(f) for f in freq_amp], columns=['freq'])
        # fa_lo = fa[(fa.freq<=fal)].index[0]
        # fa_hi = fa[(fa.freq<=fah)].index[0]
        
        fp = pd.DataFrame([float(f) for f in freq_pha], columns=['freq'])
        fp_lo = np.where(altxx==altxx[(np.abs(altxx - fpl)).argmin()])[0][0]
        fp_hi = np.where(altxx==altxx[(np.abs(altxx - fph)).argmin()])[0][0]
        fa = pd.DataFrame([float(f) for f in freq_amp], columns=['freq'])   
        fa_lo = np.where(altyy==altyy[(np.abs(altyy - fal)).argmin()])[0][0]
        fa_hi = np.where(altyy==altyy[(np.abs(altyy - fah)).argmin()])[0][0]
        
        fal_lbl = str(fal).replace('.','_') 
        fah_lbl = str(fah).replace('.','_')
        fpl_lbl =  str(fpl).replace('.','_') 
        fph_lbl = str(fph).replace('.','_')
        reglbl = '_fp_' + fpl_lbl + '__' + fph_lbl +'_fa_' + fal_lbl + '__' + fah_lbl
        reglbl2 = 'Phase freq: (' + str(fpl) + ' - ' + str(fph) +' Hz), Amplitude freq: (' + str(fal) + ' - ' + str(fah) + ' Hz)'
        es_region = es.iloc[fa_hi:fa_lo, fp_lo:fp_hi]
        pv_region = pv.iloc[fa_hi:fa_lo, fp_lo:fp_hi]
        
        regions = [0]*len(images)
        for thispac in range(len(images)):
            regions[thispac] = np.mean(images[thispac,fa_hi:fa_lo, fp_lo:fp_hi])
        regions = np.array(regions)
            
            
        print('doing statistics on possible PAC cluster in frequency pair domain')
        print(reglbl2 + '\n')
        alc_i = np.where(pac_age.AUD==1)
        unaff_i = np.where(pac_age.AUD==0)
        alc_pac = regions[alc_i]
        unaff_pac = regions[unaff_i]      
        # stats = ttest_ind(alc_pac, unaff_pac, equal_var=False)
        clust_stats = mannwhitneyu(alc_pac, unaff_pac)
        # clust_stats = pg.mwu(alc_pac, unaff_pac)
       

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
        all_pvals = []
        for smp in symptoms:
            pac_age[smp] = pac_age[smp].fillna(1)
            alc_has_i = np.where((pac_age.audcnt>=severity_scores[0]) & (pac_age[smp]==5))
            alc_hasnt_i = np.where((pac_age.audcnt>=severity_scores[0]) & (pac_age[smp]==1))
            symptom_stats = mannwhitneyu(regions[alc_has_i],regions[alc_hasnt_i])
            print(sympt_lbls[smp] + 'p = ' + str(round(symptom_stats.pvalue,4)))
            all_pvals.append(symptom_stats.pvalue)

        # NOW WE TEST 
        sympt_fdr = false_discovery_control(all_pvals)
        print('minimum FDR corrected p-value = ' +str(round(sympt_fdr.min(),4)) )
        if np.sum(sympt_fdr<=0.05)>0:
            for smp in symptoms: 
                pac_age[smp] = pac_age[smp].fillna(1)
                alc_has_i = np.where((pac_age.audcnt>=severity_scores[0]) & (pac_age[smp]==5))
                alc_hasnt_i = np.where((pac_age.audcnt>=severity_scores[0]) & (pac_age[smp]==1))
                symptom_stats = mannwhitneyu(regions[alc_has_i],regions[alc_hasnt_i])
                if symptom_stats.pvalue<=0.05:
                    info_ttl = SEX + ' - has=' + str(len(alc_has_i[0])) + ', hasnt=' + str(len(alc_hasnt_i[0])) + ', p=' + str(round(symptom_stats.pvalue,4))


                    med_has = np.median(regions[alc_has_i])
                    med_hasnt = np.median(regions[alc_hasnt_i])
                    
                    # USE WITH RAW DATA RANGING FROM 0-255 
                    # mins = np.array([(np.floor(regions[alc_has_i].min()/10))*10, (np.floor(regions[alc_hasnt_i].min()/10))*10])
                    # maxs = np.array([(np.ceil(regions[alc_has_i].max()/10))*10, (np.ceil(regions[alc_hasnt_i].max()/10))*10])
                    # lo = mins.min()
                    # hi = maxs.max()
                    # bins = np.linspace(lo,hi,int(2*((hi-lo)/10) + 1))
                                        
                    mins = np.array([(np.floor(regions[alc_has_i].min()*100)), (np.floor(regions[alc_hasnt_i].min()*100))])
                    maxs = np.array([(np.ceil(regions[alc_has_i].max()*100)), (np.ceil(regions[alc_hasnt_i].max()*100))])
                    lo = mins.min()
                    hi = maxs.max()
                    binnum = int(hi-lo + 1)
                    bins = np.linspace(lo/100,hi/100,binnum)

                    # plt.hist(regions[alc_hasnt_i],label='has not', alpha=1, bins=bins, edgecolor='k', color='white')
                    # plt.hist(regions[alc_has_i],label='has', alpha=0.7, bins=bins, color='gray')
                    # plt.hist(regions[alc_has_i],label='has', alpha=1, bins=bins, edgecolor='k', color='white')
                    # plt.hist(regions[alc_hasnt_i],label='has not', alpha=0.7, bins=bins, color='gray')
                    
                    plt.hist(regions[alc_has_i],label='has', alpha=0.5, bins=bins, edgecolor='k')
                    plt.hist(regions[alc_hasnt_i],label='has not', alpha=0.5, bins=bins)
                    # plt.yscale('log')
                    plt.title(info_ttl + '\n' + 'PAC distributions for AUD with vs without ' + sympt_lbls[smp].strip('\t') + '\n' + reglbl2)
                    plt.xlabel('PAC')
                    plt.ylabel('Number of participants')
                    plt.legend()
                    plt.show()
                    plt.close()

        # GENERATE CUSTOM AXIS TICKS FOR THIS PAC SUBDOMAIN WITHIN COMODULOGRAM
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
        pvf.set_xlabel('Phase Frequency (Hz)', fontsize=16)
        pvf.set_ylabel('Amplitude Frequency (Hz)\n', fontsize=16)
        pvf.set_title(ttl + '\n' + reglbl2)
        # pvf.set_xticks(ix_all, xtv_all)
        # pvf.set_yticks(iy_all, ytv_all)
        pvf.set_xticks(my_xticks_i, my_xticks)
        pvf.set_yticks(my_yticks_i, my_yticks)
        pvf.tick_params(direction='out', labelsize=14, length=4, width=1, colors='k', bottom=True, left=True)
        pvf.collections[0].colorbar.ax.tick_params(labelsize=14)
        pvf.figure.axes[-1].yaxis.label.set_size(16)
        ax = plt.gca()
        # NOW WE ADD RECTANGLES SURROUNDING THE PAC DOMAINS BEING ANALZED 
        fp = pd.DataFrame([float(f) for f in freq_pha], columns=['freq'])
        fp_lo = np.where(altxx==altxx[(np.abs(altxx - fpl)).argmin()])[0][0]
        fp_hi = np.where(altxx==altxx[(np.abs(altxx - fph)).argmin()])[0][0]
        fa = pd.DataFrame([float(f) for f in freq_amp], columns=['freq'])   
        fa_lo = np.where(altyy==altyy[(np.abs(altyy - fal)).argmin()])[0][0]
        fa_hi = np.where(altyy==altyy[(np.abs(altyy - fah)).argmin()])[0][0]
        # fp = pd.DataFrame([float(f) for f in freq_pha], columns=['freq'])
        # fp_lo = fp[(fp.freq>=fpl)].index[0]
        # fp_hi = fp[(fp.freq>=fph)].index[0]
        # fa = pd.DataFrame([float(f) for f in freq_amp], columns=['freq'])
        # fa_lo = fa[(fa.freq<=fal)].index[0]
        # fa_hi = fa[(fa.freq<=fah)].index[0]        
        ax.add_patch(
            patches.Rectangle(
                (fp_lo,fa_hi), 
                fp_hi-fp_lo, 
                fa_lo-fa_hi,
                edgecolor='white',
                fill=False,
                lw=2 ))
        for _,spine in ax.spines.items():
            spine.set_visible(True)    
        output = plt.Axes.get_figure(pvf)
        plt.show()
        
                
        cp = false_discovery_control(pv_region)
        cp2 = cp.copy()
        cp_min = str(round(cp2.min(axis=None),4))
        cp2[cp2>alpha_test] = alpha_test
        im = plt.imshow(cp2, cmap='rocket_r', aspect='auto')
        plt.colorbar(im, label='p-value\n' + '(minimum p = ' + cp_min + ')')
        plt.xticks(ix, xtv)
        plt.xlabel('Phase frequency (Hz)')
        plt.yticks(iy, ytv)
        plt.ylabel('Amplitude frequency (Hz)')
        plt.title(ttl + '\n' + reglbl2)
        plt.show()
        plt.close()
        
        sig_fdr_num = np.array([cp<=alpha_test]).sum()
        if sig_fdr_num>0:
                
            if ('PAC' in pac_age):
                print('\nUPDATING PAC COLUMN IN pac_age')
                del pac_age['PAC']
            pac_age.insert(0, 'PAC', regions)            
                
            DSM5_symptom_counts = pac_age[aud_symp]
            s0 = pac_age.PAC[DSM5_symptom_counts==0]
            s6 = pac_age.PAC[DSM5_symptom_counts==6]
            s7 = pac_age.PAC[DSM5_symptom_counts==7]
            s8 = pac_age.PAC[DSM5_symptom_counts==8]
            s9 = pac_age.PAC[DSM5_symptom_counts==9]
            s10 = pac_age.PAC[DSM5_symptom_counts==10]
            s11 = pac_age.PAC[DSM5_symptom_counts==11]
            f_st, p_v = f_oneway(s0, s6,s7,s8,s9,s10,s11)

            if p_v<=0.05:
                # plt.scatter(pac_age.iloc[alc_i].audcnt,pac_age.iloc[alc_i].PAC)
                
                plt.scatter(pac_age[aud_symp], pac_age.PAC)
                plt.title(ttl + reglbl)
                plt.xlabel('Number of AUD Symptoms (0 to 11) among participants')
                plt.ylabel('PAC')
                plt.show()
                     
            
            U = clust_stats.statistic
            n1, n2 = len(alc_pac), len(unaff_pac)
            U2 = n1*n2 - U
                            
            med_alc = np.median(alc_pac)
            med_unaf = np.median(unaff_pac)

            # # MANN WHITNEY BY HAND, THOUGH NOT VALIDATED
            # u1df = pd.DataFrame({'pac': alc_pac, 'auddx': 1})
            # u2df = pd.DataFrame({'pac': unaff_pac, 'auddx': 0})
            # udf = pd.concat([u1df, u2df])
            # udf = udf.sort_values(by='pac')
            # udf.reset_index(inplace=True)
            # udf['rs_raw'] = udf.index
            # udf['rs'] = udf.index
            # # ties = udf[udf.duplicated(subset='pac')]
            # # for t in ties.pac:
            # #     idx  = udf[udf.pac==t].index
            # #     new_rs = np.mean(udf[udf.pac==t].rs)
            # #     udf.loc[idx,'rs'] = new_rs
            # rs1 = sum(udf[udf.auddx==1].rs)
            # rs2 = sum(udf[udf.auddx==0].rs)
            # U1 = n1*n2 + (n1*(n1 + 1))/2 - rs1
            # U2 = n1*n2 + (n2*(n2 + 1))/2 - rs2
            
            # NON-PARAMETRIC STATISTICS FOR EFFECT SIZE
            roc_auc = U/(n1*n2) # CALLED THETA CAP OR THETA HAT IN "Confidence intervals for the Mann-Whitney test", Perme and Manevski, 2018. CHARACTERIZES THE DEGREE OF SEPARATION
            rank_biserial_corr_r = 1-((2*U2)/(n1*n2))
            # NON-PARAMETRIC STATISTICS FOR STANDARD ERROR/CONFIDENCE INTERVALS
            # https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test#Normal_approximation_and_tie_correction
            m_u = (n1*n2)/2 # EXPECTED VALUE OF U, MU SUB-U 
            s_u = np.sqrt((n1*n2*(n1 + n2 + 1))/12) # STANDARD ERROR OF U, SIGMA SUB-U
            z_u = (U - m_u)/s_u # Z-VALUE FOR U STATISTIC
            r_u = z_u/np.sqrt(n1 + n2) # COMMON LANGUAGE EFFECT SIZE (CLES), R < 0.3 SMALL EFFECT, 0.3 < R < 0.5 MEDIUM EFFECT, R > 0.5 LARGE EFFECT

            sulbl = str(round(s_u,3))
            ulbl = ', U=' + str(round(U,0))
            u2lbl = ', U2=' + str(round(U2,0))
            plbl = ', p=' + str(round(clust_stats.pvalue,4))
            zlbl = ', z=' + str(round(z_u,4))
            cleslbl = ', CLES=' + str(round(roc_auc,3))
            rlbl = ', r=' + str(round(rank_biserial_corr_r,3))
            
            clust_stats_ttl = sexlbl + ulbl + zlbl + plbl + cleslbl + rlbl                        
            cs = pg.mwu(alc_pac, unaff_pac)
            U2cs = n1*n2 - cs['U-val'].values[0]
            cs_ttl = sexlbl + ', U=' + str(round(cs['U-val'].values[0],0)) + ', U2=' + str(round(U2cs,0)) + ', p=' + str(round(cs['p-val'].values[0],4)) + ', z=' + str(round(z_u,4)) + ', CLES=' + str(round(cs['CLES'].values[0],3)) + ', r=' + str(round(cs['RBC'].values[0],3))                        
            # cs_ttl_TEST = sexlbl + ', p=' + str(round(clust_stats.pvalue,4)) + ', z=' + str(round(z_u,4)) + ', CLES=' + str(round(cs['CLES'].values[0],3)) + ', r=' + str(abs(round(cs['RBC'].values[0],3)))                        
            print(clust_stats_ttl)
            # if not(clust_stats_ttl==cs_ttl):
            #     print('~~~~~~~~~~~~ FLAG ~~~~~~~~~~~~~~~~')
            #     print('hand calculations:' + clust_stats_ttl)
            #     print('pengouin package: ' + cs_ttl)
            #     # clust_stats_ttl = clust_stats_ttl + ' FLAG'
            #     print('~~~~~~~~~~~~ FLAG ~~~~~~~~~~~~~~~~')
            # else:
            #     print(clust_stats_ttl)

            
            # USE WITH RAW DATA RANGING FROM 0-255             
            # mins = np.array([(np.floor(alc_pac.min()/10))*10, (np.floor(unaff_pac.min()/10))*10])
            # maxs = np.array([(np.ceil(alc_pac.max()/10))*10, (np.ceil(unaff_pac.max()/10))*10])
            # lo = mins.min()
            # hi = maxs.max()
            # bins = np.linspace(lo,hi,int(2*((hi-lo)/10) + 1))
            
            mins = np.array([(np.floor(alc_pac.min()*100)), (np.floor(unaff_pac.min()*100))])
            maxs = np.array([(np.ceil(alc_pac.max()*100)), (np.ceil(unaff_pac.max()*100))])
            lo = mins.min()
            hi = maxs.max()
            hilolbl = 'high: ' + str(hi) +' - low: ' + str(lo) + ' = ' + str(hi-lo) + ' '
            
            binnum = int(hi-lo + 1)
            # binnum = int((hi-lo)/2 + 1)
            if binnum<17:
                binnum = 17
            # binnum = int(hi-lo + 1)
            # binnum = int((hi-lo)/2 + 2)
            # binnum = 17
            
            bins = np.linspace(lo/100,hi/100,binnum)
                        
            # plt.hist(alc_pac,label='AUD', alpha=0.3, bins=bins, edgecolor='k', color='white')
            # plt.hist(unaff_pac,label='Unaffected', alpha=0.7, bins=bins) #, edgecolor='k')
            ulbl = 'Unaffected (N=' + str(len(alc_pac)) + ')'

            plt.hist(unaff_pac,label=ulbl, alpha=0.8, bins=bins, color='black', histtype='step') #, edgecolor='k')
            plt.hist(alc_pac,label='AUD (N=' + str(len(unaff_pac)) + ')', alpha=0.7, bins=bins, color='grey') #, histtype='step')  #, edgecolor='k')
            plt.axvline(x=med_alc,color='red',linestyle='--')
            plt.axvline(x=med_unaf,color='black',linestyle=':')
            # plt.yscale('log')
            plt.title(reglbl2 + '\n' + clust_stats_ttl + '\n' + hilolbl + 'binnum=' + str(binnum))
            plt.xlabel('PAC')
            plt.ylabel('Number of participants')
            plt.legend()
            plt.show()
            plt.close()

            # plt.figure(figsize=(8, 6))
            im = plt.imshow(pv_region, cmap='jet', aspect='auto')
            plt.colorbar(im, shrink=0.9, label='p-value\n' + '(minimum p = ' + str(round(pv_region.min(axis=None),4)))
            plt.xticks(ix, xtv)
            plt.xlabel('Phase frequency (Hz)')
            plt.yticks(iy, ytv)
            plt.ylabel('Amplitude frequency (Hz)')
            plt.title(reglbl2 + '\n' + clust_stats_ttl)
            plt.show()
            plt.close()
        
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
            hmr.set_title(reglbl2 + '\n' + clust_stats_ttl)
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
            
            
            # # BAR GRAPHS FOR THIS PAC CLUSTER DOMAIN
            # height = [np.mean(alc_pac), np.mean(unaff_pac), 0]
            # width = [0.2 , 0.2 , 1]
            # x_pos = [0.25 , 0.5 , 1]
            # bars = ['AUD','Unaff','']
            # plt.title(reglbl + '\np = ' + str(round(clust_stats.pvalue,4)) + ', ' + clust_stats_ttl, fontsize = 9)
            # plt.bar(x_pos, height , width=width, color='w', edgecolor='k')
            # plt.errorbar(x_pos, height ,yerr=[sem(alc_pac),sem(unaff_pac),0], fmt='o', color='r')
            # # plt.grid(axis='y')
            # plt.xticks(x_pos, bars)
            # plt.ylim([180,195])
            # plt.ylabel('PAC')
            # # plt.axis
            # ax = plt.gca()
            # ax.set_facecolor('0.9')
            # ax.spines['bottom'].set_color('black')
            # ax.spines['left'].set_color('black')
            # fig = plt.gcf()
            # plt.show()   
            # plt.draw()
            # fig.savefig(write_dir + ttl + reglbl + '-EFFECTSIZE_reg_bar.png', bbox_inches='tight', format=image_format)
            # plt.close()
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
                              
            # X = pa_alc.age.values #.reshape((-1, 1)) 
            X = pa_alc['years_alc'].values
            if len(X[~np.isnan(X)])>0:
                indx = np.where(~np.isnan(X))
                X = X[indx]
                y = pa_alc.PAC.values 
                y = y[indx]
                res = linregress(X,y)
                if res.pvalue<=alpha:
                    plt.plot(X,y,'o', label='AUD')
                    plt.plot(X, res.intercept + res.slope*X, 'r', label='fitted line')
                    # plt.xlim([24,51])
                    plt.ylim([bins[0],bins[-1]])
                    plt.xlabel('years with AUD diagnosis')
                    plt.ylabel('PAC')
                    plt.title(SEX + ', p = ' + str(round(res.pvalue,3)) + ' AUD ' + reglbl)
                    plt.legend()
                    plt.show()
                    plt.close()
        
                    print('\nPAC x Age in AUD ')                 
                    print('p = ' + str(res.pvalue))
                    print('slope = ' + str(res.slope))
                    print('intercept = ' + str(res.intercept))
                    print('r_value = ' + str(res.rvalue))
                    print('std_err = ' + str(res.stderr))
                    print('r^2 = ' + str((res.rvalue**2)))
                    print('\n')
            
            # X = pa_ctl.age.values #.reshape((-1, 1)) 
            # X = pa_ctl['years_alc'].values
            # if len(X[~np.isnan(X)])>0:
            #     indx = np.where(~np.isnan(X))
            #     X = X[indx]
            #     y = pa_ctl.PAC.values 
            #     y = y[indx]
            #     res = linregress(X,y)
            #     if res.pvalue<=alpha:
            #         plt.plot(X,y,'o', label='Unaffected')
            #         plt.plot(X, res.intercept + res.slope*X, 'r', label='fitted line')
            #         # plt.xlim([24,51])
            #         plt.ylim([100,255])
            #         plt.xlabel('years with AUD diagnosis')
            #         plt.ylabel('PAC')
            #         plt.title(SEX + ', p = ' + str(round(res.pvalue,3)) + ' Unaffected ' + reglbl)
            #         plt.legend()
            #         plt.show()
                
            #         print('\nPAC x Age in Unaffected ')                                  
            #         print('p = ' + str(res.pvalue))
            #         print('slope = ' + str(res.slope))
            #         print('intercept = ' + str(res.intercept))
            #         print('r_value = ' + str(res.rvalue))
            #         print('std_err = ' + str(res.stderr))
            #         print('r^2 = ' + str((res.rvalue**2)))
            #         print('\n')

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




