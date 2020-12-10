        #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script that tests feedback learning with limited amount of trials.
Similar to Fig.4D
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sklearn.linear_model as lm
import sklearn.metrics as m

#######################
# SET DIRECTORY
#######################  
loaddir = 'data/fig4/'
savdir = 'figures/fig4/'
if not os.path.exists(savdir):
    os.mkdir(savdir)

#######################
# LOAD SIMULATION DATA
#######################
# load main data dictionaries
data = np.load(loaddir+'relearning_results.npy',allow_pickle=True).item()

activity = data['manifold']['original']['activity']
activity2 = activity.reshape(50,-1,800)
T = data['perturbations']['T']
T_within = data['perturbations']['T_within']
T_outside = data['perturbations']['T_outside']
fbw = np.linalg.pinv(T_within)
fbo = np.linalg.pinv(T_outside)

#######################
# ANALYSIS
#######################
reps = 20
corr = np.zeros((reps,50,2))
for i in range(50):
    for j in range(reps):
        pick_trials = np.random.choice(range(activity2.shape[0]),i+1,replace=False)
        until = (i+1)*180 # one trial = 180 
        
        temp_act = activity2[pick_trials]
        temp_act = temp_act.reshape(-1,activity2.shape[-1])
        
        xw = temp_act @ T_within.T
        xo = temp_act @ T_outside.T
        y = temp_act

        regw = lm.LinearRegression() 
        regw.fit(xw,y)                               
        rego = lm.LinearRegression() 
        rego.fit(xo,y)  
        
        corr[j,i,0] = np.corrcoef(fbw.ravel(),regw.coef_.ravel())[0,1]
        corr[j,i,1] = np.corrcoef(fbo.ravel(),rego.coef_.ravel())[0,1]

x = activity @ T.T
y = activity
reg = lm.LinearRegression() 
reg.fit(x,y)   
corr_initial = np.corrcoef(np.linalg.pinv(T).ravel(),reg.coef_.ravel())[0,1]

#%%
#######################
# PLOTS
#######################
col_w = '#800000'
col_o = '#000080'

plt.figure(figsize=(4,3),dpi=96)
plt.errorbar(np.linspace(1,50,50),np.mean(corr[:,:,0],axis=0),np.std(corr[:,:,0],axis=0),color=col_w)
plt.errorbar(np.linspace(1,50,50),np.mean(corr[:,:,1],axis=0),np.std(corr[:,:,1],axis=0),color=col_o)
plt.axhline(np.mean(corr[:,-1,0]),color=col_w,linestyle='--',label='WMP')
plt.axhline(np.mean(corr[:,-1,1]),color=col_o,linestyle='--',label='OMP')
plt.axhline(corr_initial,color='k',linestyle='--',label='Initial')
plt.ylabel('Corr. coef. btw. correct\nand inferred feedback')
plt.xlabel('# Trials')
plt.ylim(0,1)
plt.xlim(1,10)
plt.savefig(savdir+'fig4D.svg',bbox_inches='tight')
