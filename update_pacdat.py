# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 13:02:22 2024

@author: lifep
"""
import pandas as pd
import coga_support_defs as csd
import numpy as np


base_dir = 'D:\\COGA_eec\\' #  BIOWIZARD
which_pacdat = 'pacdat_MASTER.pkl'
core_pheno_list = 'C:\\Users\\lifep\\OneDrive\\Documents\\COGA_sub_info\\core_pheno_20201120.csv'
core_pheno = pd.read_csv(core_pheno_list)


pacdat = pd.read_pickle(base_dir + which_pacdat)

if 0:
    pacdat.sort_values(by=['ID'],inplace=True)
    pacdat.reset_index(drop=True,inplace=True)
    
    cp = core_pheno.iloc[:,0:90].copy()
    
    # 1. GET OVERLAP OF SUBJECTS IN PACDAT FROM CORE_PHENO
    b = pd.merge(pacdat, cp, how='inner',on=['ID'])
    pacdat = b.to_pickle(base_dir + which_pacdat)

# 2. SORT BY ID AND RESET INDEX IN BOTH OVERLAPPING CORE_PHENO AND PACDAT
# 3. MAKE A NEW COLUMN BASED ON PACDAT INDEX TO BE ATTACHED TO PACDAT ONCE IT'S BEEN BUILT



if 0:
    pacdat['years_alc'] = 0
    
    for i,case in pacdat.iterrows():
        if not(np.isnan(case.alc_dep_ons)):
            if (case.age_this_visit>case.alc_dep_ons):
                pacdat.loc[i, 'years_alc'] = case.age_this_visit - case.alc_dep_ons
        else:
            pacdat.loc[i, 'years_alc'] = np.nan 


# ADD FACTORS TO pacdat FOR LINEAR MIXED MODEL
if 1:
    factors = [ 
        # 'cod5dx', 'cod5sx_cnt'
    ]
    # 'max_drinks', 'max_dpw', 'max_dpw_pwk',
    # 'age_first_drink', 'age_last_drink', 'regular_drinking',
    # 'reg_drink_ons', 'ever_got_drunk', 'age_first_got_drunk'    
    # 'tb_dep_dx', 'tb_dep_sx_cnt', 'tb_dep_ons', 'age_last_cig',
    # 'mjd5dx', 'mjd5sx_cnt', 'age_first_use_mj', 'age_last_use_mj',
    # 'cod5dx', 'cod5sx_cnt', 'co_dep_ons',
    # 'std5dx', 'std5sx_cnt','st_dep_ons', 'age_first_use_st', 'age_last_use_st',
    # 'opd5dx', 'opd5sx_cnt', 'opd5sx_max_cnt', 'opd5dx_sev'
    
    

    for f in factors:
        print(f)
        fdat = core_pheno.iloc[:,core_pheno.columns.get_loc(f)].copy()
        ids =  core_pheno.iloc[:,0].copy()
        cp = pd.DataFrame({'ID' :ids, f: fdat})
        # GET OVERLAP OF SUBJECTS IN PACDAT FROM CORE_PHENO
        pacdat = pd.merge(pacdat, cp, how='inner',on=['ID'])
    pacdat.to_pickle(base_dir + which_pacdat)

    
    # [print(x) for x in core_pheno.columns.values in factors]
    
    # indxs = []
    # for f in factors:
    #     print(f)
    #     indxs.append(core_pheno.columns.get_loc(f))


# pacdat.to_pickle(base_dir + which_pacdat)

