#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create plots for Figure 4.
"""

# import packages
import numpy as np
import matplotlib.pyplot as plt
import os
import sklearn.linear_model as lm

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
data0 = np.load(loaddir+'experiment_results.npy',allow_pickle=True).item()
# load parameters
manifold_trials = data0['params']['manifold_trials']
stimulus = data0['stimulus']
target = data0['target']
reduced_dim = data0['decoding']['reduced_dim']
T = data['perturbations']['T']
T_within = data['perturbations']['T_within']
T_outside = data['perturbations']['T_outside']
feedback_within = data['relearning']['feedback_within']
feedback_outside = data['relearning']['feedback_outside']
# load weights
w00 = np.load(loaddir+'W_initial.npy')          
w0 = np.load(loaddir+'W_stabilized.npy')          
w1w = np.load(loaddir+'W_within.npy')  
w1o = np.load(loaddir+'W_outside.npy')    
non0idx = np.where(w0!=0)
# load output 
activity2 = data['manifold']['original']['activity2']
order_ori = data['manifold']['original']['order']
order_w = data['manifold']['within']['order']
order_o = data['manifold']['outside']['order']
o_original = activity2 @ data['perturbations']['T'].T
o_within_wrong = activity2 @ data['perturbations']['T_within'].T
o_outside_wrong = activity2 @ data['perturbations']['T_outside'].T
a_outside = data['manifold']['outside']['activity'].reshape((activity2.shape[0],
                activity2.shape[1],-1))
o_outside = a_outside[:,:,:data['perturbations']['T_outside'].shape[-1]] @ \
                data['perturbations']['T_outside'].T
a_within = data['manifold']['within']['activity'].reshape((activity2.shape[0],
                activity2.shape[1],-1))
o_within = a_within[:,:,:data['perturbations']['T_within'].shape[-1]] @ \
                data['perturbations']['T_within'].T

#######################
# ANALYSIS
#######################
                
# performance
performance = np.zeros(5)
performance[0] = data['relearning']['cost_original']
performance[1] = data['relearning']['cost_within']
performance[2] = data['relearning']['cost_within_retrained']
performance[3] = data['relearning']['cost_outside']
performance[4] = data['relearning']['cost_outside_retrained']

# weight change standard deviation
weights = np.zeros(2)
weights[0] = np.std(w1w-w0)
weights[1] = np.std(w1o-w0)

# explained variance (same measurement as in Golub et al. 2018 supplement)
explained_variance = np.zeros(3)
proj_original = data['manifold']['original']['evec'].T @  \
        data['manifold']['original']['cov'] @ data['manifold']['original']['evec']
proj_within = data['manifold']['original']['evec'].T @ \
        data['manifold']['within']['cov'] @ data['manifold']['original']['evec']
proj_outside = data['manifold']['original']['evec'].T @ \
        data['manifold']['outside']['cov'] @ data['manifold']['original']['evec']
proj_outside2 = data['perturbations']['P_permute'] @ \
                 data['manifold']['outside']['cov']  @ \
                data['perturbations']['P_permute'].T     
overlap_original = np.trace(proj_original[:reduced_dim])/np.trace(data['manifold']['original']['cov'])
overlap_within = np.trace(proj_within[:reduced_dim])/np.trace(data['manifold']['within']['cov'])
overlap_outside = np.trace(proj_outside[:reduced_dim])/np.trace(data['manifold']['outside']['cov'])
overlap_outside2 = np.trace(proj_outside2[:reduced_dim])/np.trace(data['manifold']['outside']['cov'])
normExplainedVar_within = overlap_within/overlap_original
normExplainedVar_outside = overlap_outside/overlap_original
normExplainedVar_outside2 = overlap_outside2/overlap_original
explained_variance[0] = normExplainedVar_within
explained_variance[1] = normExplainedVar_outside
explained_variance[2] = normExplainedVar_outside2
     
#######################
# PLOTS
#######################
# set colors
col_oo = '#3CAEA3'
col_o = '#4a5899'
col_w = '#92374d'
col_ori = 'k'
col_pw = '#800000'
col_po = '#000080'

# FIGURE 1: learned feedback weights
plt.figure(figsize=(8,3),dpi=96)
plt.subplots_adjust(wspace=0.3)
plt.subplot(1,2,1)
plt.scatter(np.linalg.pinv(T_within),feedback_within,c=col_w,alpha=0.2,label='Within')
plt.xlabel('Correct feedback weights')
plt.ylabel('Inferred feedback weights')
plt.title('Within-manifold perturbation (WMP)',color=col_w)
plt.subplot(1,2,2)
plt.scatter(np.linalg.pinv(T_outside),feedback_outside,c=col_o,alpha=0.2,label='Outside')
plt.xlabel('Correct feedback weights')
plt.ylabel('Inferred feedback weights')
plt.title('Outside-manifold perturbation (OMP)',color=col_o)
plt.savefig(savdir+'fig4BC.svg',bbox_inches='tight')

# FIGURE 2: performance
cols = [col_ori,col_pw,col_w,col_po,col_o] 
plt.figure(figsize=(4,3),dpi=96)
for j in range(performance.shape[-1]):
    plt.bar(j,np.mean(performance[j]),yerr=np.std(performance[j]),
            color=cols[j])
#    plt.violinplot(performance[:,j],[j],showmeans=True)
plt.xticks(range(performance.shape[-1]),['Initial','WMP',
           'WMR','OMP','OMR'])
plt.xlim(-0.5,4.5)
plt.ylim(0,1.2)
plt.ylabel('MSE')
plt.savefig(savdir+'fig4E.svg',bbox_inches='tight')

# FIGURE 3: explained variance
cols = [col_w,col_o,col_oo] 
plt.figure(figsize=(4,3),dpi=96)
for j in range(explained_variance.shape[-1]):
    plt.bar(j,explained_variance[j]*100,color=cols[j])
plt.xticks(range(explained_variance.shape[-1]),
           ['I-WMR','I-OMR','P-OMR'],
           rotation=30,ha='right')
plt.xlim(-0.5,2.5)
plt.ylim(0,100)
plt.ylabel('Manifold overlap (%)')
plt.savefig(savdir+'manifold_overlap.svg',bbox_inches='tight')

# FIGURE 4: interpolation
activity = data['manifold']['original']['activity']
P = data['perturbations']['P']
lambs = np.linspace(0,1,50)
scors = np.zeros((len(lambs),2))
overlap = np.zeros(len(lambs))
for j in range(len(lambs)):
    if np.isnan(lambs[j]):
        continue
    Tinc = (1-lambs[j])*T + lambs[j]*T_outside
    x = activity @ Tinc.T
    y = activity
    rego = lm.LinearRegression() 
    rego.fit(x,y)  
    scors[j,0] = np.corrcoef(np.linalg.pinv(Tinc).ravel(),rego.coef_.ravel())[0,1]
    # calculate overlap of P and Tinc
    for i in range(10):
        overlap[j] += (Tinc[0] @ P[i])**2 / np.linalg.norm(Tinc[0])**2
        overlap[j] += (Tinc[1] @ P[i])**2 / np.linalg.norm(Tinc[1])**2
# plot it    
plt.figure(figsize=(4,3),dpi=96)
plt.plot(overlap/2,scors[:,0],'k')
plt.ylabel('Corr. coefr. btw. correct\nand inferred feedback')
plt.xlabel('Overlap btw. BCI readout\nand condensed neural manifold')
plt.xlim(0,1)
plt.ylim(0,1)
plt.savefig(savdir+'fig4G.svg',bbox_inches='tight')
