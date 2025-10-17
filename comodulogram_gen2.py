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
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mticker
# from PIL import Image
# import mne
import scipy

SEX = '   m   '.upper().strip()
do_other_sex_clusters = False # THIS WILL CALCULATE CLUSTER STATISTICS USING THE CLUSTER PA DOMAINS FROM THE OPPOSITE SEX 
age_groups = [25,50]
severity_scores = [6,11,'SEVERE']

do_first_figures = False # IF THIS IS SET TRUE THEN ONLY THE FIGURES SHOWING ALL CLUSTERS WILL BE GENERATED
do_cluster_figures = True  # IF THIS IS TRUE THEN ONLY THE PA DOMAIN SPECIFIC CLUSTER ANALYSIS FIGURES ARE GENERATED
save_figs = False
do_info_figs = False # THIS WILL GENERATE FIGURES WITH CLUSTER STATISTICS IN TITLE -- NOT TO BE USED TO GENERATE FINAL SET OF FIGURES

add_clusters = True # THIS ADDS CLUSTERS TO THE MEDIAN AUD AND UNAFFECTED COMODULOGRAMS
do_all_clusters = True # THIS TOGGLES WHETHER ALL CLUSTERS OR ONLY THE SIGNIFICANT ONES ARE ADDED TO FIGURES
fig_fname_marker = '' # PUT TEXT HERE TO DIFFERENTIATE SIMILAR FIGURES

alpha = 0.05
which_dx = 'AUD' # AUD ALAB ALD
aud_symp = 'audcnt' # audcnt ald5sx_max_cnt
channel = 'FZ'
alpha_test = 0.5 # CHANGE THIS SETTING TO REVEAL CLUSTER STATISTICS FOR NON-SIGNIFICANT RESULTS

load_age_match = True
do_age_match = False
process_age_matches = False
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
border_tb = 7
border_rl = 5
phase_start_hz = 0.1
phase_end_hz = 13
amp_start_hz = 4
amp_end_hz = 50

load_new_dat = True # False True
info_fn = 'pac_info_ages_25_50_AUD__ALL_0_11' + data_info + '.pkl'
pac_fn = 'pac_3d_ages_25_50_AUD__ALL_0_11' + data_info + '.npy'
# base_dir = 'D:\\COGA_eec\\' 
base_dir = 'C:\\Users\\lifep\\OneDrive\\Documents\\'

fldrname = 'COMODULOGRAMS_revisions\\' + SEX
vmin = -5.2/255
vmax = 5.2/255
vmn = 184/255
vmx = 196/255
    
# PAC DOMAINS - WOMEN
if do_all_clusters:
    FPAS = [
            [0.1,1.1,32,38],
            [0.4,1.4,19,22],
            [0.5,1.5,9,14],
            [1.3,2.3,29,33],
            [2,3,12,15],
            [2.5,3.5,26,33],
            [2.6,3.6,35,39],
            [3,4.5,8,11],
            [3.25,4.25,21,25],
            [3.5,4.5,12,15],
            [4,5,41,47],
            [4.25,5.25,35,40],
            [5.5,6.5,10,13],
            [7,8.5,14,17],
            [9.7,10.7,16,19],
            [10.5,11.5,35,43],
            [12,13,30,42]
            ]
else:
    FPAS = [
            [2.5,3.5,26,33],
            [3,4.5,8,11],
            [10.5,11.5,35,43],
            [12,13,30,42]
            ]

# PAC DOMAINS - MEN
if do_all_clusters:
    MPAS = [
            [0.1,1.1,23,28],            
            [0.6,1.6,32,36],        
            [0.7,1.7,28,32],
            [1,2,36,40],
            [1.2,2.2,14,17],
            [1.4,2.4,41,47],
            [2,3,21,25],
            [2.7,3.7,7,9],
            [3.6,4.7,12,14],
            [4,5,33,40],
            [6.5,7.5,35,38],
            [7,8,20,23],
            [7.5,8.5,14,17],
            [8,9,20,22],
            [9,11,38,44],
            [10.3,11.3,18,21],
            [11,12,28,32]
            ]
else: 
    MPAS = [
            [0.6,1.6,32,36],
            [4,5,33,40],
            [9,11,38,44]
            ]

PACdomainsStdFreqBand = [
    [8,13,28,49.6],
    [3,8,28,49.6],
    [0.1,3,28,49.6],
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
shuf_seeds = 424242


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
    if not(process_age_matches):
        pval_mx = np.load('pval_mx' + '_' + SEX +'.npy')
        effect_mx = np.load('effect_mx' + '_' + SEX +'.npy')
        aud_mx = np.load('aud_mx' + '_' + SEX +'.npy')
        ctl_mx = np.load('ctl_mx' + '_' + SEX +'.npy')
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


# THIS WILL ADD UP DSM4 SYMPTOMS FOR AA AND AD
# for i in range(len(pac_age)): pac_age.loc[i, 'dsm4cnt'] = np.sum([pac_age.iloc[i,13:24]==5])

match_i = pac_age.index
images = images_all[match_i]/255



yalc_avg = np.round(pac_age[(pac_age.sex==SEX) & (pac_age.AUD==1)].years_alc.mean(),1)
yalc_sem = np.round(pac_age[(pac_age.sex==SEX) & (pac_age.AUD==1)].years_alc.sem(),1)
yalc_min = pac_age[(pac_age.sex==SEX) & (pac_age.AUD==1)].years_alc.min()
yalc_max = pac_age[(pac_age.sex==SEX) & (pac_age.AUD==1)].years_alc.max()


ed_aud = len(pac_age[(pac_age.sex==SEX) & (pac_age.AUD==1) & (pac_age.ever_drink==1)]) # 1=no , 5=yes
ed_no_ctl = len(pac_age[(pac_age.sex==SEX) & (pac_age.AUD==0) & (pac_age.ever_drink==1)]) # 1=no , 5=yes
ed_yes_ctl = len(pac_age[(pac_age.sex==SEX) & (pac_age.AUD==0) & (pac_age.ever_drink==5)]) # 1=no , 5=yes
ed_no_aud = len(pac_age[(pac_age.sex==SEX) & (pac_age.AUD==1) & (pac_age.ever_drink==1)]) # 1=no , 5=yes
ed_yes_aud = len(pac_age[(pac_age.sex==SEX) & (pac_age.AUD==1) & (pac_age.ever_drink==5)]) # 1=no , 5=yes

afd_aud_avg = np.round(pac_age[(pac_age.sex==SEX) & (pac_age.AUD==1)].age_first_got_drunk.mean(),1)
afd_aud_sem = np.round(pac_age[(pac_age.sex==SEX) & (pac_age.AUD==1)].age_first_got_drunk.sem(),1)
afd_aud_min = np.round(pac_age[(pac_age.sex==SEX) & (pac_age.AUD==1)].age_first_got_drunk.min(),1)
afd_aud_max = np.round(pac_age[(pac_age.sex==SEX) & (pac_age.AUD==1)].age_first_got_drunk.max(),1)

afd_ctl_avg = np.round(pac_age[(pac_age.sex==SEX) & (pac_age.AUD==0)].age_first_got_drunk.mean(),1)
afd_ctl_sem = np.round(pac_age[(pac_age.sex==SEX) & (pac_age.AUD==0)].age_first_got_drunk.sem(),1)
afd_ctl_min = np.round(pac_age[(pac_age.sex==SEX) & (pac_age.AUD==0)].age_first_got_drunk.min(),1)
afd_ctl_max = np.round(pac_age[(pac_age.sex==SEX) & (pac_age.AUD==0)].age_first_got_drunk.max(),1)

yalc_hist = pac_age[(pac_age.sex==SEX) & (pac_age.AUD==1)].years_alc.hist(label='years AUD')
plt.title(sexlbl + ' (AUD group)')
plt.xlabel('Years since AUD diagnosis')
plt.ylabel('Counts')
plt.show()
plt.close()

afd_aud_hist = pac_age[(pac_age.sex==SEX) & (pac_age.AUD==1)].age_first_got_drunk.hist(bins=15, alpha=0.5, label='age first drink (AUD)')
afd_ctl_hist = pac_age[(pac_age.sex==SEX) & (pac_age.AUD==0)].age_first_got_drunk.hist(bins=15, alpha=0.5, label='age first drink (Unaf)')
plt.title(sexlbl)
plt.xlabel('Age (years)')
plt.ylabel('Counts')
plt.legend()
plt.show()
plt.close()

print('\n' + sexlbl + ', ever drink (AUD): yes=' + str(ed_yes_aud) + ', no==' + str(ed_no_aud))
print(sexlbl + ', ever drink (unaffected): yes=' + str(ed_yes_ctl) + ', no==' + str(ed_no_ctl))
print(sexlbl + ', years alcoholic: (MEAN +/- SEM) ' + str(yalc_avg) + ' +/- ' + str(yalc_sem) + ' years old; min, max ' + str(yalc_min) + ', ' + str(yalc_max)  )
print(sexlbl + ', first drink: AUD ' + str(afd_aud_avg) + ' +/- ' + str(afd_aud_sem) + ' years old; min, max ' + str(afd_aud_min) + ', ' + str(afd_aud_max)  )
print(sexlbl + ', first drink: unaffected ' + str(afd_ctl_avg) + ' +/- ' + str(afd_ctl_sem) + ' years old; min, max ' + str(afd_ctl_min) + ', ' + str(afd_ctl_max) + '\n' )



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~ NOW THAT WE HAVE OUR DATASET WE CAN START WORKING ~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


which_pacdat = 'D:\\COGA_eec\\pacdat_MASTER.pkl' 
pacdat = pd.read_pickle(which_pacdat)
c1 = pacdat.columns.get_loc('alc_dep_sx1')
c2 = pacdat.columns.get_loc('alc_abuse_max_sx4')

df_list = []
pacdat = pacdat[(pacdat.channel=='FZ') & (pacdat.age_this_visit>=25) & (pacdat.age_this_visit<=50) & (pacdat.sex_x==SEX)]
for i in range(0,len(pac_age)):
    tmp = pacdat[(pacdat.ID==pac_age.iloc[i].ID) & (pacdat.age_this_visit==pac_age.iloc[i].age)].copy()
    df_list.append(tmp)
    # c1 = tmp.columns.get_loc('alc_dep_sx1')
    # c2 = tmp.columns.get_loc('alc_abuse_max_sx4')
pac_df = pd.concat(df_list)
pac_df.rename(columns={'sex_x' : 'sex'}, inplace=True)
pac_df.rename(columns={'race_x' : 'race'}, inplace=True)
csd.print_demo_vals(pac_df)




# EXTRACTING SPECTRAL POWER INFO FROM pacdat
if 0:
    which_pacdat = 'D:\\COGA_eec\\pacdat_MASTER.pkl' 
    pacdat = pd.read_pickle(which_pacdat)
    pacdat = pacdat[(pacdat.channel==channel) & (pacdat.age_this_visit>=min_age) & (pacdat.age_this_visit<=max_age) & (pacdat.sex_x==SEX)]
    
    # HANDY TO GET A PRINTOUT OF ALL COLUMNS IN A PANDAS DATAFRAME
    # pd.DataFrame(pac_df.columns).head(50)
    psds = []
    auddx = []
    for i in range(0,len(pac_df)):
        fn = pac_df.iloc[i].eeg_file_name
        sf = pac_df.iloc[i].sample_rate
        adx = pac_df.iloc[i].AUD_this_visit
        apth = 'D:\\COGA_eec\\FZ\\'
        aeeg = pd.read_csv(apth + fn + '.csv')
        f, Pxx_den = scipy.signal.welch(aeeg.FZ.values, fs=sf, nperseg=sf*2, noverlap=sf)
        mask = (f >= 0.1) & (f <= 50)
        f_limited = f[mask]
        Pxx_limited = Pxx_den[mask]
        psds.append(Pxx_limited)
        auddx.append(adx)
        # plt.figure(figsize=(10, 6))
        # plt.semilogy(f_limited, Pxx_limited)
        # # plt.semilogy(f, Pxx_den)
        # plt.xlabel('Frequency [Hz]')
        # plt.ylabel('PSD [V^2/Hz]')
        # plt.title('EEG Power Spectral Density')
        # plt.grid(True)
        # plt.show()
    psd_df = pd.DataFrame({'psd' : psds, 'dx' : auddx})
    apsd = psd_df[psd_df.dx==True]
    audpsd = apsd['psd'].mean()
    upsd = psd_df[psd_df.dx==False]
    unafpsd = upsd['psd'].mean()
    plt.figure(figsize=(10, 6))
    plt.semilogy(f_limited, audpsd)
    plt.semilogy(f_limited, unafpsd)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD [V^2/Hz]')
    plt.title('EEG Power Spectral Density')
    plt.grid(True)
    plt.show()
    # plt.plot(audpsd)
    # plt.plot(unafpsd)

    # df_list = []
    # print('\n ')
    # for s in range(len(pac_age)):
    #     sbj = pac_age.iloc[s]
    #     df_list.append(pacdat[(pacdat.ID==sbj.ID) & (pacdat.age_this_visit==sbj.age)].copy())
    # pac_df = pd.concat(df_list)
    # freq_bands = ['delta', 'theta','alpha','low_beta','high_beta','gamma']
    # for fd in freq_bands:
    #     aa = pac_df[pac_df.ald5dx==1][[fd]][fd].to_list()
    #     uu = pac_df[pac_df.ald5dx==5][[fd]][fd].to_list()
    #     mw = mannwhitneyu(aa,uu)
    #     diff = np.mean(aa) - np.mean(uu)
    #     print(fd + ' ' + str(mw.pvalue)[:6] + '      AUD - unaff = ' + str(np.mean(diff)))

if process_age_matches:
    pval_mx = np.zeros((img_rows,img_cols))
    effect_mx = np.zeros((img_rows,img_cols))
    aud_mx = np.zeros((224,224))
    # aud_sem_mx = np.zeros((224,224))
    ctl_mx = np.zeros((224,224))
    # ctl_sem_mx = np.zeros((224,224))
    
    
    print('\ndoing statistics on all PAC frequency pairs')
    # images = np.load('C:\\Users\\lifep\\OneDrive\\Documents\\pac_3d_' + targ_folder + '.npy')
    # images = images_all[match_i]
    for x in range(img_rows):
        for y in range(img_cols):
            # print(str(x) + ' ' + str(y))
            alc_i = np.where(pac_age.AUD==1)
            unaff_i = np.where(pac_age.AUD==0)
            
            alc_pac = images[alc_i,x,y][0]
            unaff_pac = images[unaff_i,x,y][0]
            
            stats = mannwhitneyu(alc_pac, unaff_pac)
            # stats = ttest_ind(alc_pac[0], unaff_pac[0], equal_var=False)
            pval_mx[x,y] = stats.pvalue
            # effect_mx[x,y] = np.mean(alc_pac) -  np.mean(unaff_pac)
            effect_mx[x,y] = np.median(alc_pac) -  np.median(unaff_pac)
            aud_mx[x,y] = np.median(alc_pac)
            # aud_sem_mx[x,y] = np.std(alc_pac)/np.sqrt(len(alc_pac))
            ctl_mx[x,y] = np.median(unaff_pac)
            # ctl_sem_mx[x,y] = np.std(unaff_pac)/np.sqrt(len(unaff_pac))
    np.save('pval_mx' + '_' + SEX +'.npy', pval_mx)
    np.save('effect_mx' + '_' + SEX +'.npy', effect_mx)
    np.save('aud_mx' + '_' + SEX +'.npy', aud_mx)
    np.save('ctl_mx' + '_' + SEX +'.npy', ctl_mx)
      


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ NOW WE GENERATE FIGURES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# TRIM BORDER
# RESIZED X AND Y AXIS LABELS
xax = np.arange(phase_start_hz,phase_end_hz,((phase_end_hz - phase_start_hz)/(pac_len-border_rl*2)))
freq_pha = [str(round(x,2)) for x in xax]
yax = np.arange(amp_start_hz,amp_end_hz,((amp_end_hz - amp_start_hz)/(pac_len-border_tb*2)))
yax = yax[::-1]
freq_amp = [str(round(x,2)) for x in yax]

p2 = pval_mx[border_tb:pac_len-border_tb,border_rl:pac_len-border_rl].copy()
pv = pd.DataFrame(p2,  columns=freq_pha, index=freq_amp)
e2 = effect_mx[border_tb:pac_len-border_tb,border_rl:pac_len-border_rl].copy()
es = pd.DataFrame(e2,  columns=freq_pha, index=freq_amp)
aud2 = aud_mx[border_tb:pac_len-border_tb,border_rl:pac_len-border_rl].copy()
audcomod = pd.DataFrame(aud2,  columns=freq_pha, index=freq_amp)
unaf2 = ctl_mx[border_tb:pac_len-border_tb,border_rl:pac_len-border_rl].copy()
unafcomod = pd.DataFrame(unaf2,  columns=freq_pha, index=freq_amp)

# HERE WE ARE BUILDING THE AXIS TICK VALUES FOR COMODULOGRAMS
nn = np.array( es)
lenx = np.shape(nn)[1]
leny = np.shape(nn)[0]
comod_clusters = np.zeros((leny,lenx))

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
    
# THIS IS TO GENERATE TICKS FOR PAC DOMAIN BOUNDARY FREQUENCIES
pdsfb = [
    [8,13,28,50],
    [3,8,28,50],
    [0.1,3,28,50],
    [8,13,13,28],
    [3,8,13,28],   
    [0.1,3,13,28],
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
    pac_domain_mask[fa_hi:fa_lo, fp_lo:fp_hi] = 1
pac_domain_mask = pac_domain_mask.astype(bool)

# NOW THAT WE HAVE OUR MASK, WE CAN APPLY IT TO OUR COMODULOGRAM DERIVED RESULTS
pac_domain_diffs = effect_mx[border_tb:pac_len-border_tb,border_rl:pac_len-border_rl].copy()
pdd = np.multiply(pac_domain_diffs, pac_domain_mask)
pddiffs_df = pd.DataFrame(pdd,  columns=freq_pha, index=freq_amp)

pac_domain_pvals = pval_mx[border_tb:pac_len-border_tb,border_rl:pac_len-border_rl].copy()
pdp = np.multiply(pac_domain_pvals, pac_domain_mask)
pdpvals_df = pd.DataFrame(pdp,  columns=freq_pha, index=freq_amp)

pac_domain_aud = aud_mx[border_tb:pac_len-border_tb,border_rl:pac_len-border_rl].copy()
pdpaud = np.multiply(pac_domain_aud, pac_domain_mask)
pdpaudvals_df = pd.DataFrame(pdpaud,  columns=freq_pha, index=freq_amp)

pac_domain_unaf = ctl_mx[border_tb:pac_len-border_tb,border_rl:pac_len-border_rl].copy()
pdpunaf = np.multiply(pac_domain_unaf, pac_domain_mask)
pdpunafvals_df = pd.DataFrame(pdpunaf,  columns=freq_pha, index=freq_amp)    
    

if do_first_figures:
    pac_age[(pac_age.audcnt!=0)].audcnt.hist(bins=5)
    plt.title('AUD by symptom counts - ' + sexlbl)
    plt.show()
    plt.close()
        
    # DIFFERENCE COMODULOGRAM FIGURE WITH CLUSTER DOMAINS MARKED
    hm = sns.heatmap(es, 
                     cmap="jet", 
                     cbar_kws={'shrink': 0.9, 'pad': 0.1, 'aspect': 10}, #, 'orientation': 'horizontal'}, 
                     vmin=vmin, vmax=vmax,
                     xticklabels=False,
                     yticklabels=False
                     )
    hm.set_xlabel('Phase Frequency (Hz)', fontsize=16, labelpad=10)
    hm.set_ylabel('Amplitude Frequency (Hz)\n', fontsize=16, labelpad=-2)
    # hm.set_title(ttl)
    cbar = hm.collections[0].colorbar
    cbar.set_label('\nPAC difference (AUD - Unaff)', labelpad=-98, color='black')
    hm.tick_params(direction='out', labelsize=16, length=4, width=1, colors='k', bottom=True, left=True)
    # THE NEXT TWO LINES CONTROL WHAT SIZE THE TEXT IS FOR COLORBAR TICKS AND LABEL
    hm.collections[0].colorbar.ax.tick_params(labelsize=12) # COLORBAR LABEL TEXT SIZE
    hm.figure.axes[-1].yaxis.label.set_size(14) # COLORBAR TICKS TEXT SIZE
    hm.figure.axes[-1].locator_params(axis='y', nbins=5)
    hm.set_title(sexlbl + '\nComodulogram Difference Matrix', fontsize=17, pad=12)
    # hm.set_title('Difference Matrix - ' + sexlbl, fontsize=17, pad=8)
    hm.set_xticks(my_xticks_i, my_xticks)
    hm.set_yticks(my_yticks_i, my_yticks)
    ax = plt.gca()    
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
                    lw=2))
    for _,spine in ax.spines.items():
        spine.set_visible(True)
    plt.show()
    if save_figs:
        output = plt.Axes.get_figure(hm)
        output.savefig(write_dir + sexlbl + '_EFFECTSIZE_with_cluster' + fig_fname_marker + '.png', bbox_inches='tight', format=image_format, dpi=300)
        plt.close(output)
    
    
    
    
    # COMODULOGRAM SIGNIFICANCE MATRIX - WHITE CLUSTERS - GREEN PAC DOMAINS
    pv_fig = pv.copy()
    pv_fig[pv_fig>alpha] = alpha
    hmmin = str(round(np.min(pv_fig),6))
    pvf = sns.heatmap(pv_fig, 
                     vmax=0.05,
                     cmap="rocket_r", 
                     cbar_kws={'label': 'p-value', 'shrink': 0.9, 'pad': 0.1, 'aspect': 10},
                     xticklabels=False,
                     yticklabels=False
                     )
    pvf.set_xlabel('Phase Frequency (Hz)', fontsize=16, labelpad=10)
    pvf.set_ylabel('Amplitude Frequency (Hz)\n', fontsize=16, labelpad=-2)
    # pvf.set_title(ttl)
    pvf.set_title(sexlbl + '\nComodulogram Significance Matrix', fontsize=17, pad=12)
    pvf.set_xticks(my_xticks_i, my_xticks)
    pvf.set_yticks(my_yticks_i, my_yticks)
    cbar = pvf.collections[0].colorbar
    this_cbar_lbl = cbar.ax.yaxis.get_label().get_text()
    cbar.set_label(this_cbar_lbl, labelpad=-75, color='black')
    pvf.tick_params(direction='out', labelsize=16, length=4, width=1, colors='k', bottom=True, left=True)
    # THE NEXT TWO LINES CONTROL WHAT SIZE THE TEXT IS FOR COLORBAR TICKS AND LABEL
    pvf.collections[0].colorbar.ax.tick_params(labelsize=12) # COLORBAR LABEL TEXT SIZE
    pvf.figure.axes[-1].yaxis.label.set_size(14) # COLORBAR TICKS TEXT SIZE
    pvf.figure.axes[-1].locator_params(axis='y', nbins=5)
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
                edgecolor='white',
                fill=False,
                lw=1 ))
    # for pa in PACdomainsStdFreqBand:
    for pafb in pdsfb:
        fpl = pafb[0]
        fph = pafb[1]
        fal = pafb[2]
        fah = pafb[3]
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
                edgecolor='lime',
                fill=False,
                linestyle=':',
                lw=1 ))
    for _,spine in ax.spines.items():
        spine.set_visible(True)    
    plt.show()
    if save_figs:
        output = plt.Axes.get_figure(pvf)
        output.savefig(write_dir + sexlbl + '-PVALUES_with_clusters_domains' + fig_fname_marker + '.png', bbox_inches='tight', format=image_format, dpi=300)
        plt.close(output)
    
    
    # THIS LETS US BLANK OUT THE LEGEND BUT KEEP THE SPACING CONSISTENT WITH OTHER COMODULOGRAM FIGURES
    color1 = '#ffffff'
    color2 = '#ffffff'
    cmap_blank = LinearSegmentedColormap.from_list("blank", [color1, color2])
    # NOW WE GENERATE COMODULOGRAM SCHEMA WITH CLUSTER PA DOMAINS MAPPED ONTO IT
    clstrs = sns.heatmap(comod_clusters, 
                     cmap=cmap_blank, 
                     cbar_kws={'shrink': 0.9, 'pad': 0.1, 'aspect': 10}, #, 'orientation': 'horizontal'}, 
                     vmin=vmin, vmax=vmax,
                     xticklabels=False,
                     yticklabels=False
                     )
    clstrs.set_xlabel('Phase Frequency (Hz)', fontsize=16, labelpad=10)
    clstrs.set_ylabel('Amplitude Frequency (Hz)\n', fontsize=16, labelpad=-2)
    # clstrs.set_title(ttl)
    cbar = clstrs.collections[0].colorbar
    cbar.set_label('\n', labelpad=-98, color='white')
    cbar.ax.tick_params(colors='white')
    clstrs.tick_params(direction='out', labelsize=16, length=4, width=1, colors='k', bottom=True, left=True)
    # THE NEXT TWO LINES CONTROL WHAT SIZE THE TEXT IS FOR COLORBAR TICKS AND LABEL
    clstrs.collections[0].colorbar.ax.tick_params(labelsize=12) # COLORBAR LABEL TEXT SIZE
    clstrs.figure.axes[-1].yaxis.label.set_size(14) # COLORBAR TICKS TEXT SIZE
    clstrs.figure.axes[-1].locator_params(axis='y', nbins=5)
    clstrs.set_title(sexlbl + '\nCandidate PAC Clusters', fontsize=17, pad=12)
    # clstrs.set_title('Difference Matrix - ' + sexlbl, fontsize=17, pad=8)
    clstrs.set_xticks(my_xticks_i, my_xticks)
    clstrs.set_yticks(my_yticks_i, my_yticks)
    ax = plt.gca()    
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
                    lw=1))
    for pafb in pdsfb:
        fpl = pafb[0]
        fph = pafb[1]
        fal = pafb[2]
        fah = pafb[3]
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
    # clstrs.cax.set_visible(False)
    # cbar_ax = fig.axes[0]
    # cbar_ax.set_visible(False)
    plt.show()
    if save_figs:
        output = plt.Axes.get_figure(clstrs)
        output.savefig(write_dir + sexlbl + '-candidate_clusters_with_domains' + fig_fname_marker + '.png', bbox_inches='tight', format=image_format, dpi=300)
        plt.close(output)

    
        
    # # THIS BLOCK OF CODE GENERATES COMODULOGRAM FIGURE WHERE VALUES ARE DISPLAYED ON WITHIN PAC DOMAINS TARGETING PAC CLUSTERS
    # # EXAMPLES HERE FOR HOW TO SELECTIVELY ZERO OUT OR PRESERVE 2-D MATRIX DATA FOR DISPLAY IN SCIENTIFIC FIGURES
    # pac_domain_mask = np.zeros([pv.shape[0], pv.shape[1]])
    # for pa in PACdomainsStdFreqBand:    
    #     fpl = pa[0]
    #     fph = pa[1]
    #     fal = pa[2]
    #     fah = pa[3]
            
    #     fp = pd.DataFrame([float(f) for f in freq_pha], columns=['freq'])
    #     fp_lo = np.where(altxx==altxx[(np.abs(altxx - fpl)).argmin()])[0][0]
    #     fp_hi = np.where(altxx==altxx[(np.abs(altxx - fph)).argmin()])[0][0]
    #     fa = pd.DataFrame([float(f) for f in freq_amp], columns=['freq'])   
    #     fa_lo = np.where(altyy==altyy[(np.abs(altyy - fal)).argmin()])[0][0]
    #     fa_hi = np.where(altyy==altyy[(np.abs(altyy - fah)).argmin()])[0][0]
    #     pac_domain_mask[fa_hi:fa_lo, fp_lo:fp_hi] = 1
    # pac_domain_mask = pac_domain_mask.astype(bool)
    
    # # NOW THAT WE HAVE OUR MASK, WE CAN APPLY IT TO OUR COMODULOGRAM DERIVED RESULTS
    # pac_domain_diffs = effect_mx[border_tb:pac_len-border_tb,border_rl:pac_len-border_rl].copy()
    # pdd = np.multiply(pac_domain_diffs, pac_domain_mask)
    # pddiffs_df = pd.DataFrame(pdd,  columns=freq_pha, index=freq_amp)
    
    # pac_domain_pvals = pval_mx[border_tb:pac_len-border_tb,border_rl:pac_len-border_rl].copy()
    # pdp = np.multiply(pac_domain_pvals, pac_domain_mask)
    # pdpvals_df = pd.DataFrame(pdp,  columns=freq_pha, index=freq_amp)
    
    # pac_domain_aud = aud_mx[border_tb:pac_len-border_tb,border_rl:pac_len-border_rl].copy()
    # pdpaud = np.multiply(pac_domain_aud, pac_domain_mask)
    # pdpaudvals_df = pd.DataFrame(pdpaud,  columns=freq_pha, index=freq_amp)
    
    # pac_domain_unaf = ctl_mx[border_tb:pac_len-border_tb,border_rl:pac_len-border_rl].copy()
    # pdpunaf = np.multiply(pac_domain_unaf, pac_domain_mask)
    # pdpunafvals_df = pd.DataFrame(pdpunaf,  columns=freq_pha, index=freq_amp)
    
    # NOW WE GENERATE EFFECT SIZE COMODULOGRAM WITH PAC CLUSTER DOMAINS HIGHLIGHTED pval_mx
    hmm = sns.heatmap(pdpaudvals_df,
    				  cmap="jet", 
                  cbar_kws={'label': 'PAC strength', 'shrink': 0.9, 'pad': 0.1, 'aspect': 10},
    				  mask=(pdpvals_df==0), 
    				  vmin=vmn, vmax=vmx,
    				  xticklabels=False,
    				  yticklabels=False
    				  )
    audmap = sns.heatmap(audcomod, 
                     cmap="jet", 
                     cbar=None, 
                     mask=(pdpvals_df>0), 
                     vmin=vmn, vmax=vmx,
                     xticklabels=False,
                     yticklabels=False,
                     alpha=0.3
                     )
    audmap.set_xlabel('Phase Frequency (Hz)', fontsize=16, labelpad=10)
    audmap.set_ylabel('Amplitude Frequency (Hz)\n', fontsize=16, labelpad=-2)
    # audmap.set_title(ttl)
    audmap.set_title(sexlbl + '\nMedian PAC - AUD group', fontsize=17, pad=8)
    # audmap.set_title('Median PAC in ' + sexlbl + ' (AUD)', fontsize=17, pad=8)
    audmap.set_xticks(xstdFreqs_i, xstdFreqs)
    audmap.set_yticks(ystdFreqs_i, ystdFreqs)
    # audmap.set_xticks(my_xticks_i, my_xticks)
    # audmap.set_yticks(my_yticks_i, my_yticks)
    cbar = audmap.collections[0].colorbar
    this_cbar_lbl = cbar.ax.yaxis.get_label().get_text()
    cbar.set_label(this_cbar_lbl, labelpad=-75, color='black')
    audmap.tick_params(direction='out', labelsize=16, length=4, width=1, colors='k', bottom=True, left=True)
    # THE NEXT TWO LINES CONTROL WHAT SIZE THE TEXT IS FOR COLORBAR TICKS AND LABEL
    audmap.collections[0].colorbar.ax.tick_params(labelsize=12) # COLORBAR LABEL TEXT SIZE
    audmap.figure.axes[-1].yaxis.label.set_size(14) # COLORBAR TICKS TEXT SIZE
    audmap.figure.axes[-1].locator_params(axis='y', nbins=5)
    ax = plt.gca()
    # for pa in PACdomainsStdFreqBand:
    for pafb in pdsfb:
        fpl = pafb[0]
        fph = pafb[1]
        fal = pafb[2]
        fah = pafb[3]
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
    if add_clusters:
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
                        lw=1))
    for _,spine in ax.spines.items():
        spine.set_visible(True)
    plt.show()
    if save_figs:
        output = plt.Axes.get_figure(audmap)
        output.savefig(write_dir + sexlbl + '_median_AUD_comod_with_domains' + fig_fname_marker + '.png', bbox_inches='tight', format=image_format, dpi=300)
        plt.close(output)
    

        
    hmm = sns.heatmap(pdpunafvals_df,
    				  cmap="jet", 
                  cbar_kws={'label': 'PAC strength', 'shrink': 0.9, 'pad': 0.1, 'aspect': 10},
    				  mask=(pdpvals_df==0), 
    				  vmin=vmn, vmax=vmx,
    				  xticklabels=False,
    				  yticklabels=False
    				  )    
    unafmap = sns.heatmap(unafcomod, 
                     cmap="jet", 
                     cbar=None, 
                     mask=(pdpvals_df>0),
                     vmin=vmn, vmax=vmx,
                     xticklabels=False,
                     yticklabels=False,
                     alpha=0.3
                     )
    unafmap.set_xlabel('Phase Frequency (Hz)', fontsize=16, labelpad=10)
    unafmap.set_ylabel('Amplitude Frequency (Hz)\n', fontsize=16, labelpad=-2)
    unafmap.set_title(sexlbl + '\nMedian PAC - unaffected group', fontsize=17, pad=8)
    unafmap.set_xticks(xstdFreqs_i, xstdFreqs)
    unafmap.set_yticks(ystdFreqs_i, ystdFreqs)
    # unafmap.set_xticks(my_xticks_i, my_xticks)
    # unafmap.set_yticks(my_yticks_i, my_yticks)
    cbar = unafmap.collections[0].colorbar
    cbar.set_label(this_cbar_lbl, labelpad=-75, color='black')
    this_cbar_lbl = cbar.ax.yaxis.get_label().get_text()
    unafmap.tick_params(direction='out', labelsize=16, length=4, width=1, colors='k', bottom=True, left=True)
    # THE NEXT TWO LINES CONTROL WHAT SIZE THE TEXT IS FOR COLORBAR TICKS AND LABEL
    unafmap.collections[0].colorbar.ax.tick_params(labelsize=12) # COLORBAR LABEL TEXT SIZE
    unafmap.figure.axes[-1].yaxis.label.set_size(14) # COLORBAR TICKS TEXT SIZE
    unafmap.figure.axes[-1].locator_params(axis='y', nbins=5)
    ax = plt.gca()
    for pa in pdsfb:
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
    if add_clusters:
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
                        lw=1))
    for _,spine in ax.spines.items():
        spine.set_visible(True)
    plt.show()
    if save_figs:
        output = plt.Axes.get_figure(unafmap)
        output.savefig(write_dir + sexlbl + '_median_CTL_comod_with_domains' + fig_fname_marker + '.png', bbox_inches='tight', format=image_format, dpi=300)
        plt.close(output)

# 1. Group neighboring clusters where significant PAC found
# 2. PAC domains restricted as much as possible to where raw sig pvals found
# 3. If PAC differences within test domain are not significant then cluster rejected
# 4. If no spectral power at PAC frequencies then reject cluster?
# 5. Only clusters with raw sig pvals spanning at least 1 Hz width of phase or amplitude frequency (minimum size 1Hz x 1 Hz)
# 6. Must have FDR sig p in PAC domain where cluster is found otherwise reject
# 7. Lumping stage after splitting stage guided by where maximum PAC differences appear
# using these rules, we identified X number of clusters for further testing

if do_cluster_figures:

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~ NOW WE START ANALYZES FOR EACH CLUSTER ~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    clustID = 0
    clust_p = []
    clust_pac_p = []
    for pa in PAS:  
        fpl = pa[0]
        fph = pa[1]
        fal = pa[2]
        fah = pa[3]
        clustID = clustID + 1
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
        reglbl3 = 'P(' + str(fpl) + ' - ' + str(fph) +' Hz), A(' + str(fal) + ' - ' + str(fah) + ' Hz)'
        reglbl4 = 'Cluster ID: ' + str(clustID) + '\nPhase freq: (' + str(fpl) + ' - ' + str(fph) +' Hz)\nAmplitude freq: (' + str(fal) + ' - ' + str(fah) + ' Hz)'
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
        clust_pac_p.append([clust_stats.pvalue])

        # GENERATE CUSTOM AXIS TICKS FOR THIS PAC SUBDOMAIN WITHIN COMODULOGRAM
        nn = np.array( es_region)
        lenx = np.shape(nn)[1]
        leny = np.shape(nn)[0]
        
        # sub_freq_pha = es_region.columns
        # sub_freq_amp = es_region.index
        # sub_freq_amp_i = np.arange(0,len(sub_freq_amp))
        # ytks = np.array([str(np.round(np.float16(y),1)) for y in sub_freq_amp])
        
        yy = np.arange(fal,fah,((fah-fal)/np.shape(nn)[0]))
        ytks = np.array([str(np.round(y,1)) for y in yy])
        intvly = round((np.shape(nn)[0])/7)
        # iy = np.arange(0, np.shape(nn)[0],intvly)
        iya = np.arange(0, leny-1,intvly)
        ytv = ytks[iya]    
        iy = np.arange(leny-1,0,-intvly) # NEED TO USE A REVERSED INDEX LIST BECAUSE OF HJOW sms HEATMAP STORES VALUES
        # iy = np.array(iya.tolist()[::-1]) # NEED TO USE A REVERSED INDEX LIST BECAUSE OF HJOW sms HEATMAP STORES VALUES 
    
        xx = np.arange(fpl,fph,((fph-fpl)/lenx))
        xtks = np.array([str(np.round(x,1)) for x in xx])
        intvlx = round((np.shape(nn)[1])/7)
        ix = np.arange(0,lenx,intvlx)
        xtv = xtks[ix]
        
        # DIFFERENCE COMODULOGRAM FIGURE WITH CLUSTER DOMAINS MARKED
        # hm = sns.heatmap(es, 
        #                  cmap="jet", 
        #                  cbar_kws={'label': '\nPAC strength change\n(AUD - Unaff)', 'shrink': 0.8, 'pad': 0.02}, 
        #                  vmin=vmin, vmax=vmax,
        #                  xticklabels=False,
        #                  yticklabels=False
        #                  )
        # hm.set_xlabel('Phase Frequency (Hz)', fontsize=18)
        # hm.set_ylabel('Amplitude Frequency (Hz)\n', fontsize=18)
        # hm.set_title(ttl)
        # hm.set_xticks(my_xticks_i, my_xticks)
        # hm.set_yticks(my_yticks_i, my_yticks)
        # hm.tick_params(direction='out', labelsize=14, length=4, width=1, colors='k', bottom=True, left=True)
        # hm.collections[0].colorbar.ax.tick_params(labelsize=14)
        # hm.figure.axes[-1].yaxis.label.set_size(16)
        
        hm = sns.heatmap(es, 
                         cmap="jet", 
                         cbar_kws={'shrink': 0.9, 'pad': 0.1, 'aspect': 10}, #, 'orientation': 'horizontal'}, 
                         vmin=vmin, vmax=vmax,
                         xticklabels=False,
                         yticklabels=False
                         )
        hm.set_xlabel('Phase Frequency (Hz)', fontsize=16, labelpad=10)
        hm.set_ylabel('Amplitude Frequency (Hz)\n', fontsize=16, labelpad=-2)
        # hm.set_title(ttl)
        cbar = hm.collections[0].colorbar
        cbar.set_label('\nPAC difference (AUD - Unaff)', labelpad=-98, color='black')
        hm.tick_params(direction='out', labelsize=16, length=4, width=1, colors='k', bottom=True, left=True)
        # THE NEXT TWO LINES CONTROL WHAT SIZE THE TEXT IS FOR COLORBAR TICKS AND LABEL
        hm.collections[0].colorbar.ax.tick_params(labelsize=12) # COLORBAR LABEL TEXT SIZE
        hm.figure.axes[-1].yaxis.label.set_size(14) # COLORBAR TICKS TEXT SIZE
        hm.figure.axes[-1].locator_params(axis='y', nbins=5)
        hm.set_title(sexlbl + '\nDifference Matrix', fontsize=17, pad=8)
        # hm.set_title('Difference Matrix - ' + sexlbl, fontsize=17, pad=8)
        hm.set_xticks(my_xticks_i, my_xticks)
        hm.set_yticks(my_yticks_i, my_yticks)
        ax = plt.gca()
        # NOW WE ADD RECTANGLES SURROUNDING THE PAC DOMAINS BEING ANALZED 
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
                linestyle='-',
                fill=False,
                lw=2 ))
        for pafb in pdsfb:
            fpl = pafb[0]
            fph = pafb[1]
            fal = pafb[2]
            fah = pafb[3]
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
        for _,spine in ax.spines.items():
            spine.set_visible(True) 
        plt.show()
        if save_figs:
            output = plt.Axes.get_figure(hm)
            output.savefig(write_dir + sexlbl + '_DIFF_' + str(clustID) + '_candidate_cluster' + fig_fname_marker + '.png', bbox_inches='tight', format=image_format, dpi=300)
            plt.close(output)


        
        # pv_fig = pv.copy()
        # pv_fig[pv_fig>alpha] = alpha
        # hmmin = str(round(np.min(pv_fig),6))
        # pvf = sns.heatmap(pv_fig, 
        #                  vmax=0.05,
        #                  cmap="rocket_r", 
        #                  cbar_kws={'label': 'p-value\n(min=' + hmmin + ')', 'shrink': 0.8, 'pad': 0.02},
        #                  xticklabels=False,
        #                  yticklabels=False
        #                  )
        # pvf.set_xlabel('Phase Frequency (Hz)', fontsize=16)
        # pvf.set_ylabel('Amplitude Frequency (Hz)\n', fontsize=16)
        # pvf.set_title(ttl + '\n' + reglbl2)
        # pvf.set_xticks(my_xticks_i, my_xticks)
        # pvf.set_yticks(my_yticks_i, my_yticks)
        # pvf.tick_params(direction='out', labelsize=14, length=4, width=1, colors='k', bottom=True, left=True)
        # pvf.collections[0].colorbar.ax.tick_params(labelsize=14)
        # pvf.figure.axes[-1].yaxis.label.set_size(16)
        
        
        pv_fig = pv.copy()
        pv_fig[pv_fig>alpha] = alpha
        hmmin = str(round(np.min(pv_fig),6))
        pvf = sns.heatmap(pv_fig, 
                         vmax=0.05,
                         cmap="rocket_r", 
                         cbar_kws={'label': 'p-value', 'shrink': 0.9, 'pad': 0.1, 'aspect': 10},
                         xticklabels=False,
                         yticklabels=False
                         )
        pvf.set_xlabel('Phase Frequency (Hz)', fontsize=16, labelpad=10)
        pvf.set_ylabel('Amplitude Frequency (Hz)\n', fontsize=16, labelpad=-2)
        # pvf.set_title(ttl)
        pvf.set_title(sexlbl + '\nSignificance Matrix', fontsize=17, pad=8)
        pvf.set_xticks(my_xticks_i, my_xticks)
        pvf.set_yticks(my_yticks_i, my_yticks)
        cbar = pvf.collections[0].colorbar
        this_cbar_lbl = cbar.ax.yaxis.get_label().get_text()
        cbar.set_label(this_cbar_lbl, labelpad=-75, color='black')
        pvf.tick_params(direction='out', labelsize=16, length=4, width=1, colors='k', bottom=True, left=True)
        # THE NEXT TWO LINES CONTROL WHAT SIZE THE TEXT IS FOR COLORBAR TICKS AND LABEL
        pvf.collections[0].colorbar.ax.tick_params(labelsize=12) # COLORBAR LABEL TEXT SIZE
        pvf.figure.axes[-1].yaxis.label.set_size(14) # COLORBAR TICKS TEXT SIZE
        pvf.figure.axes[-1].locator_params(axis='y', nbins=5)
        ax = plt.gca()
        for pasfb in pdsfb:
            fpl = pasfb[0]
            fph = pasfb[1]
            fal = pasfb[2]
            fah = pasfb[3]
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
                    alpha=0.5,
                    edgecolor='lime',
                    fill=False,
                    linestyle=':',
                    lw=1 ))
        # NOW WE ADD RECTANGLES SURROUNDING THE PAC DOMAINS BEING ANALZED 
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
                edgecolor='lime',
                fill=False,
                lw=1 ))
        for _,spine in ax.spines.items():
            spine.set_visible(True)    
        plt.show()
        if save_figs:
            output = plt.Axes.get_figure(pvf)
            output.savefig(write_dir + sexlbl + '_SIG_' + str(clustID) + '_candidate_cluster' + fig_fname_marker + '.png', bbox_inches='tight', format=image_format, dpi=300)
            plt.close(output)

        
        # MEDIAN PAC COMODULOGRAM FOR AUD GROUP WITH THE CLUSTER BEING TESTED
        # hmm = sns.heatmap(pdpaudvals_df,
    				#   cmap="jet", 
    				#   cbar_kws={'shrink': 0.8, 'pad': 0.02}, 
    				#   mask=(pdpvals_df==0), 
    				#   vmin=vmn, vmax=vmx,
    				#   xticklabels=False,
    				#   yticklabels=False
    				#   )
        # audmap = sns.heatmap(audcomod, 
        #                  cmap="jet", 
        #                  cbar=None, 
        #                  mask=(pdpvals_df>0), 
        #                  vmin=vmn, vmax=vmx,
        #                  xticklabels=False,
        #                  yticklabels=False,
        #                  alpha=0.3
        #                  )
        # audmap.set_xlabel('Phase Frequency (Hz)', fontsize=16)
        # audmap.set_ylabel('Amplitude Frequency (Hz)\n', fontsize=16, labelpad=-8)
        # # audmap.set_title('Median PAC in ' + sexlbl + ' (AUD)', fontsize=16)
        # audmap.set_title(sexlbl + ' (AUD)', fontsize=16)
        # audmap.set_xticks(xstdFreqs_i, xstdFreqs)
        # audmap.set_yticks(ystdFreqs_i, ystdFreqs)
        # audmap.tick_params(direction='out', labelsize=14, length=4, width=1, colors='k', bottom=True, left=True)
        # audmap.collections[0].colorbar.ax.tick_params(labelsize=14)
        # audmap.figure.axes[-1].yaxis.label.set_size(14)
        
        
        # NOW WE GENERATE EFFECT SIZE COMODULOGRAM WITH PAC CLUSTER DOMAINS HIGHLIGHTED pval_mx
        hmm = sns.heatmap(pdpaudvals_df,
    				  cmap="jet", 
                      cbar_kws={'label': 'PAC strength', 'shrink': 0.9, 'pad': 0.1, 'aspect': 10},
    				  mask=(pdpvals_df==0), 
    				  vmin=vmn, vmax=vmx,
    				  xticklabels=False,
    				  yticklabels=False
    				  )
        audmap = sns.heatmap(audcomod, 
                         cmap="jet", 
                         cbar=None, 
                         mask=(pdpvals_df>0), 
                         vmin=vmn, vmax=vmx,
                         xticklabels=False,
                         yticklabels=False,
                         alpha=0.3
                         )
        audmap.set_xlabel('Phase Frequency (Hz)', fontsize=16, labelpad=10)
        audmap.set_ylabel('Amplitude Frequency (Hz)\n', fontsize=16, labelpad=-2)
        # audmap.set_title(ttl)
        # audmap.set_title(sexlbl + '\nMedian PAC - AUD', fontsize=17, pad=8)
        audmap.set_title('AUD ' + sexlbl + '\nCluster: ' + reglbl3, fontsize=17, pad=12)        
        # audmap.set_title('Median PAC in ' + sexlbl + ' (AUD)', fontsize=17, pad=8)
        audmap.set_xticks(my_xticks_i, my_xticks)
        audmap.set_yticks(my_yticks_i, my_yticks)
        cbar = audmap.collections[0].colorbar
        this_cbar_lbl = cbar.ax.yaxis.get_label().get_text()
        cbar.set_label(this_cbar_lbl, labelpad=-75, color='black')
        audmap.tick_params(direction='out', labelsize=16, length=4, width=1, colors='k', bottom=True, left=True)
        # THE NEXT TWO LINES CONTROL WHAT SIZE THE TEXT IS FOR COLORBAR TICKS AND LABEL
        audmap.collections[0].colorbar.ax.tick_params(labelsize=12) # COLORBAR LABEL TEXT SIZE
        audmap.figure.axes[-1].yaxis.label.set_size(14) # COLORBAR TICKS TEXT SIZE
        audmap.figure.axes[-1].locator_params(axis='y', nbins=5)
        ax = plt.gca()
        # for pasfb in PACdomainsStdFreqBand:
        for pasfb in pdsfb:
            fpl = pasfb[0]
            fph = pasfb[1]
            fal = pasfb[2]
            fah = pasfb[3]
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
                    alpha=0.5,
                    edgecolor='black',
                    fill=False,
                    linestyle=':',
                    lw=1 ))
        # NOW WE ADD RECTANGLES SURROUNDING THE PAC DOMAINS BEING ANALZED 
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
                linestyle='-',
                fill=False,
                lw=1 ))
        for _,spine in ax.spines.items():
            spine.set_visible(True)
        # plt.title('Median PAC at FZ in ' + sexlbl + ' (AUD)') #' : 4hz at ' + str(hz4) + ', rl=' + str(border_rl) + ', tb=' + str(border_tb))
        plt.show()
        if save_figs:
            output = plt.Axes.get_figure(audmap)   
            output.savefig(write_dir + sexlbl + '_AUD_MEDIAN_' + str(clustID) + '_candidate_cluster' + fig_fname_marker + '.png', bbox_inches='tight', format=image_format, dpi=300)
            plt.close(output)

            # output = plt.Axes.get_figure(audmap)
            # output.savefig(write_dir + ttl + '_AUD_comod.png', bbox_inches='tight', format=image_format)
            # plt.close(output)
        
        
        # hmm = sns.heatmap(pdpunafvals_df,
    				#   cmap="jet", 
    				#   cbar_kws={'shrink': 0.8, 'pad': 0.02}, 
    				#   mask=(pdpvals_df==0), 
    				#   vmin=vmn, vmax=vmx,
    				#   xticklabels=False,
    				#   yticklabels=False
    				#   )    
        # unafmap = sns.heatmap(unafcomod, 
        #                  cmap="jet", 
        #                  cbar=None, 
        #                  mask=(pdpvals_df>0),
        #                  vmin=vmn, vmax=vmx,
        #                  xticklabels=False,
        #                  yticklabels=False,
        #                  alpha=0.3
        #                  )
        # unafmap.set_xlabel('Phase Frequency (Hz)', fontsize=16)
        # unafmap.set_ylabel('Amplitude Frequency (Hz)\n', fontsize=16, labelpad=-8)
        # # unafmap.set_title('Median PAC at FZ in ' + sexlbl + ' (unaffected)', fontsize=16)
        # unafmap.set_title(sexlbl + ' (unaffected)', fontsize=16)
        # unafmap.set_xticks(xstdFreqs_i, xstdFreqs)
        # unafmap.set_yticks(ystdFreqs_i, ystdFreqs)
        # unafmap.tick_params(direction='out', labelsize=14, length=4, width=1, colors='k', bottom=True, left=True)
        # unafmap.collections[0].colorbar.ax.tick_params(labelsize=14)
        # unafmap.figure.axes[-1].yaxis.label.set_size(14)
        
        
        hmm = sns.heatmap(pdpunafvals_df,
    				  cmap="jet", 
                      cbar_kws={'label': 'PAC strength', 'shrink': 0.9, 'pad': 0.1, 'aspect': 10},
    				  mask=(pdpvals_df==0), 
    				  vmin=vmn, vmax=vmx,
    				  xticklabels=False,
    				  yticklabels=False
    				  )    
        unafmap = sns.heatmap(unafcomod, 
                         cmap="jet", 
                         cbar=None, 
                         mask=(pdpvals_df>0),
                         vmin=vmn, vmax=vmx,
                         xticklabels=False,
                         yticklabels=False,
                         alpha=0.3
                         )
        unafmap.set_xlabel('Phase Frequency (Hz)', fontsize=16, labelpad=10)
        unafmap.set_ylabel('Amplitude Frequency (Hz)\n', fontsize=16, labelpad=-2)
        # unafmap.set_title(sexlbl + '\nMedian PAC - unaffected', fontsize=17, pad=8)
        unafmap.set_title('Unaffected ' + sexlbl + '\nCluster: ' + reglbl3, fontsize=17, pad=12)        
        unafmap.set_xticks(my_xticks_i, my_xticks)
        unafmap.set_yticks(my_yticks_i, my_yticks)
        cbar = unafmap.collections[0].colorbar
        cbar.set_label(this_cbar_lbl, labelpad=-75, color='black')
        this_cbar_lbl = cbar.ax.yaxis.get_label().get_text()
        unafmap.tick_params(direction='out', labelsize=16, length=4, width=1, colors='k', bottom=True, left=True)
        # THE NEXT TWO LINES CONTROL WHAT SIZE THE TEXT IS FOR COLORBAR TICKS AND LABEL
        unafmap.collections[0].colorbar.ax.tick_params(labelsize=12) # COLORBAR LABEL TEXT SIZE
        unafmap.figure.axes[-1].yaxis.label.set_size(14) # COLORBAR TICKS TEXT SIZE
        unafmap.figure.axes[-1].locator_params(axis='y', nbins=5)
        ax = plt.gca()
        # for pasfb in PACdomainsStdFreqBand:
        for pasfb in pdsfb:
            fpl = pasfb[0]
            fph = pasfb[1]
            fal = pasfb[2]
            fah = pasfb[3]
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
                    alpha=0.5,
                    edgecolor='black',
                    fill=False,
                    linestyle=':',
                    lw=1 ))
        # NOW WE ADD RECTANGLES SURROUNDING THE PAC DOMAINS BEING ANALZED 
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
                linestyle='-',
                fill=False,
                lw=1 ))
        for _,spine in ax.spines.items():
            spine.set_visible(True)
        # plt.title('Median PAC at FZ in ' + sexlbl + ' (AUD)') #' : 4hz at ' + str(hz4) + ', rl=' + str(border_rl) + ', tb=' + str(border_tb))
        plt.show()
        if save_figs:
            output = plt.Axes.get_figure(unafmap)   
            output.savefig(write_dir + sexlbl + '_CTL_MEDIAN_' + str(clustID) + '_candidate_cluster' + fig_fname_marker + '.png', bbox_inches='tight', format=image_format, dpi=300)
            plt.close(output)
        
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # HERE IS THE CORRECTION FOR MULTIPLE COMPARISONS
        # AND WHETHER OR NOT THIS CLUSTER GOES ON IN TO NEXT STEPS IN CLUSTER VALIDATION 
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                
            plt.imshow(pv_region, cmap='jet')
            plt.colorbar()
            plt.show()


        cp = false_discovery_control(pv_region)
        cp2 = cp.copy()
        cp_min = cp2.min(axis=None)
        clust_p.append([cp_min])
        cp_min_str = str(round(cp_min,4))
        cp2[cp2>alpha_test] = alpha_test
        im = plt.imshow(cp2, cmap='rocket_r', aspect='auto')
        plt.colorbar(im, label='p-value\n' + '(minimum p = ' + cp_min_str + ')')
        plt.xticks(ix, xtv)
        plt.xlabel('Phase frequency (Hz)')
        plt.yticks(iy, ytv)
        plt.ylabel('Amplitude frequency (Hz)')
        plt.title(ttl + '\n' + reglbl2)
        plt.show()
        if save_figs:
            output = plt.Axes.get_figure(im)   
            output.savefig(write_dir + sexlbl + '_FDR_REGION_' + str(clustID) + '_candidate_cluster' + fig_fname_marker + '.png', bbox_inches='tight', format=image_format, dpi=300)
            plt.close(output)
        
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
                        info_ttl = SEX + ' - has=' + str(len(alc_has_i[0])) + ', hasnt=' + str(len(alc_hasnt_i[0])) + ', p=' + str(round(symptom_stats.pvalue,5))
    
    
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
                        fig = plt.figure()
                        plt.hist(regions[alc_has_i],label='has', alpha=0.5, bins=bins, edgecolor='k')
                        plt.hist(regions[alc_hasnt_i],label='has not', alpha=0.5, bins=bins)
                        # plt.yscale('log')
                        plt.title(info_ttl + '\n' + 'PAC distributions for AUD with vs without ' + sympt_lbls[smp].strip('\t') + '\n' + reglbl2)
                        plt.xlabel('PAC')
                        plt.ylabel('Number of participants')
                        plt.legend()
                        plt.show()
                        if save_figs:
                            fig.savefig(write_dir + sexlbl + '_SYMPTOM_HIST_' + smp + '_candidate_cluster' + str(clustID) + reglbl + fig_fname_marker + '.png', bbox_inches='tight', format=image_format, dpi=300)
                            plt.close(output)
                        else:
                            plt.close()
            
            
            U = clust_stats.statistic
            n1, n2 = len(alc_pac), len(unaff_pac)
            U2 = n1*n2 - U
                            
            med_alc = np.median(alc_pac)
            med_unaf = np.median(unaff_pac)

            # # MANN WHITNEY BY HAND, THOUGH NOT VALIDATED WITH CORRECTION FOR TIES
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
            mdnlblaud = str(round(np.median(alc_pac),4))
            mdnlblunaff = str(round(np.median(unaff_pac),4))
            
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

            
            # GENERATE FIGURE OF PAC DISTRIBUTIONS BY GROUP 

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
            if binnum<17:
                binnum = 17
            # binnum = int(hi-lo + 1)
            # binnum = int((hi-lo)/2 + 2)
            # binnum = 17
            bins = np.linspace(lo/100,hi/100,binnum)
                    
            
            if do_info_figs:
                ulbl = 'Unaffected (Mdn=' + mdnlblunaff + ')'
                albl = 'AUD (Mdn=' + mdnlblaud + ')'
                thisttl = str(clustID) + 'clusterID\n' + reglbl2 + '\n' + clust_stats_ttl # + '\n' + hilolbl
            else:
                ulbl = 'Unaffected'
                albl = 'AUD'
                thisttl = reglbl4
                
                
            fig = plt.figure()
            ax = fig.gca()
            plt.hist(unaff_pac,label=ulbl, alpha=0.8, bins=bins, color='black', histtype='step') #, edgecolor='k')
            plt.hist(alc_pac,label=albl, alpha=0.7, bins=bins, color='grey') #, histtype='step')  #, edgecolor='k')
            plt.axvline(x=med_alc,color='red',linestyle='--')
            plt.axvline(x=med_unaf,color='black',linestyle=':')
            plt.title(thisttl, fontsize=18) 
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.xlabel('PAC', fontsize=18)
            plt.ylabel('Number of participants', fontsize=18)
            plt.legend(fontsize=12)
            formatter = mticker.FormatStrFormatter('%.2f') # %d formats as integer
            ax.xaxis.set_major_formatter(formatter)
            ax.yaxis.get_major_locator().set_params(integer=True)
            plt.show()
            
            # df = pd.DataFrame({'alc':alc_pac, 'unaf': unaff_pac})
            # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 4), sharex=True, sharey=True)
            # # ax = fig.gca()
            # df['unaf'].hist(label=ulbl, alpha=0.8, bins=bins, color='black', histtype='step') #, edgecolor='k')
            # df['alc'].hist(label=albl, alpha=0.7, bins=bins, color='grey') #, histtype='step')  #, edgecolor='k')
            # fig.axvline(x=med_alc,color='red',linestyle='--')
            # fig.axvline(x=med_unaf,color='black',linestyle=':')
            # # plt.yscale('log')
            # plt.title(thisttl, fontsize=18) 
            # plt.xticks(fontsize=16)
            # plt.yticks(fontsize=16)
            # plt.xlabel('PAC', fontsize=18)
            # plt.ylabel('Number of participants', fontsize=18)
            # plt.legend(fontsize=12)
            # # ax.yaxis.get_major_locator().set_params(integer=True)
            # plt.show()
            
            # ax1 = plt.subplot(111)
            # plt.hist(unaff_pac,label=ulbl, alpha=0.8, bins=bins, color='black', histtype='step') #, edgecolor='k')
            # plt.xticks(fontsize=16)
            # plt.yticks(fontsize=16)
            # ax1.yaxis.get_major_locator().set_params(integer=True)
            # plt.title(thisttl, fontsize=18) 
            # plt.xlabel('PAC', fontsize=18)
            # plt.ylabel('Number of participants', fontsize=18)
            # plt.legend(fontsize=12)
            # ax2 = plt.subplot(111, sharex=ax1)
            # plt.hist(alc_pac,label=albl, alpha=0.7, bins=bins, color='grey') #, histtype='step')  #, edgecolor='k')
            # plt.tick_params('x', labelbottom=False)
            # plt.tick_params('y', labelbottom=False)
            # # plt.xticks(fontsize=16)
            # # plt.yticks(fontsize=16)
            # plt.axvline(x=med_alc,color='red',linestyle='--')
            # plt.axvline(x=med_unaf,color='black',linestyle=':')
            # # plt.yscale('log')
            # plt.show()
            if save_figs:
                fig.savefig(write_dir + sexlbl + '_HIST_REGION_' + str(clustID) + '_candidate_cluster' + fig_fname_marker + '.png', bbox_inches='tight', format=image_format, dpi=300)
                plt.close(fig)

            
            # plt.figure(figsize=(8, 6))
            im = plt.imshow(pv_region, cmap='jet', aspect='auto')
            plt.colorbar(im, shrink=0.9, label='p-value\n' + '(minimum p = ' + str(round(pv_region.min(axis=None),4)))
            plt.xticks(ix, xtv)
            plt.xlabel('Phase frequency (Hz)')
            plt.yticks(iy, ytv)
            plt.ylabel('Amplitude frequency (Hz)')
            plt.title(reglbl2 + '\n' + clust_stats_ttl)
            plt.show()
            if save_figs:
                output = plt.Axes.get_figure(im)
                output.savefig(write_dir + sexlbl + '_SIG_REGION_' + str(clustID) + '_candidate_cluster' + fig_fname_marker + '.png', bbox_inches='tight', format=image_format, dpi=300)
                plt.close(output)
        
            hmr = sns.heatmap(es_region, 
                             cmap="icefire", 
                             cbar_kws={'label': 'PAC difference \n(AUD - unaff)', 'shrink': 0.8, 'pad': 0.02}, 
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
            if save_figs:
                output = plt.Axes.get_figure(hmr)   
                output.savefig(write_dir + sexlbl + '_DIFF_REGION_' + str(clustID) + '_candidate_cluster' + fig_fname_marker + '.png', bbox_inches='tight', format=image_format, dpi=300)
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
            
cpfdr = false_discovery_control(clust_p)
# cid = list(range(1,len(clust_p)+1))
# cid = np.array(range(1,len(clust_p)+1))
# cid = cid.reshape(1,-1).T
# ccc = np.hstack((cid,clust_p,cpfdr))
ccc = np.hstack((clust_p,cpfdr))
print(ccc)
print('\n')
cppfdr = false_discovery_control(clust_pac_p)
cccc = np.hstack((clust_pac_p,cppfdr))
print(cccc)


