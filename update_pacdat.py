# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 13:02:22 2024

@author: lifep
"""
import pandas as pd
import coga_support_defs as csd


base_dir = 'D:\\COGA_eec\\' #  BIOWIZARD
which_pacdat = 'pacdat_MASTER.pkl'
core_pheno_list = 'C:\\Users\\lifep\\OneDrive\\Documents\\COGA_sub_info\\core_pheno_20201120.csv'


pacdat = pd.read_pickle(base_dir + which_pacdat)
pacdat.sort_values(by=['ID'],inplace=True)
pacdat.reset_index(drop=True,inplace=True)

core_pheno = pd.read_csv(core_pheno_list)
cp = core_pheno.iloc[:,0:90].copy()

# 1. GET OVERLAP OF SUBJECTS IN PACDAT FROM CORE_PHENO
b = pd.merge(pacdat, cp, how='inner',on=['ID'])


# 2. SORT BY ID AND RESET INDEX IN BOTH OVERLAPPING CORE_PHENO AND PACDAT
# 3. MAKE A NEW COLUMN BASED ON PACDAT INDEX TO BE ATTACHED TO PACDAT ONCE IT'S BEEN BUILT



pacdat = b.to_pickle(base_dir + which_pacdat)

