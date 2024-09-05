# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 15:53:45 2024

@author: lifep
"""
import matplotlib.pyplot as plt
import matplotlib.lines as lns
import pandas as pd
import numpy as np

which_pacdat = 'pacdat_MASTER.pkl'

read_dir = "D:\\COGA_eec\\"

# bins=30
bins = np.arange(10,90,2)
sex = 'M'
# GET COGA META DATA
pacdat = pd.read_pickle(read_dir + which_pacdat)
chpac = pacdat[pacdat.channel=='FZ']

if 0:
    if sex=='F':
        sexlbl = 'Females'
    else:
        sexlbl = 'Males'

    plt.figure(figsize=(5, 5))
    plt.subplot(1, 1, 1)
    
    subvis = np.array(chpac[chpac.sex==sex].age_this_visit)
    aud = np.array(chpac[(chpac.sex==sex) & (chpac.AUD_this_visit==True)].age_this_visit)
    unaff = np.array(chpac[(chpac.sex==sex) & (chpac.AUD_this_visit==False)].age_this_visit)
    
    # hh,binsh = np.histogram(subvis, bins=30, density=False)
    h1,bins1 = np.histogram(subvis, bins=bins, density=False)
    h2,bins2 = np.histogram(aud, bins=bins, density=False)
    h3,bins3 = np.histogram(unaff, bins=bins, density=False)
    
    # hh  = np.insert(hh,0,0)
    h1  = np.insert(h1,0,0)
    h2  = np.insert(h2,0,0)
    h3  = np.insert(h3,0,0)
    
    # plt.yscale("log")   
    plt.step(bins1, h1)
    plt.step(bins2, h2) 
    plt.step(bins3, h3)
    # plt.plot(binsh, hh, alpha=0.3)
    
    plt.title('Distribution of ' + sexlbl + ' Ages by Visit ')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.legend(['All', 'AUD', 'Unaff'])    
    
    

if 1:
    plt.figure(figsize=(5, 5))
    plt.subplot(1, 1, 1)
    
    subvis = chpac['age_this_visit']
    # subvis = np.array(chpac.age_this_visit)
    aud = np.array(chpac[(chpac.AUD_this_visit==True)].age_this_visit)
    unaff = np.array(chpac[(chpac.AUD_this_visit==False)].age_this_visit)
    
    hh,binsh = np.histogram(subvis, bins=30, density=False)
    h1,bins1 = np.histogram(subvis, bins=bins, density=False)
    h2,bins2 = np.histogram(aud, bins=bins, density=False)
    h3,bins3 = np.histogram(unaff, bins=bins, density=False)
    
    hh  = np.insert(hh,0,0)
    h1  = np.insert(h1,0,0)
    h2  = np.insert(h2,0,0)
    h3  = np.insert(h3,0,0)
    
    plt.yscale("log")   
    plt.step(bins1, h1)
    plt.step(bins2, h2) 
    plt.step(bins3, h3)
    # plt.plot(binsh, hh, alpha=0.3)
    
    plt.title('Distribution of Ages by Visit ')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.legend(['All', 'AUD', 'Unaff'])    
