#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create plots for Figure 2.
"""

# import packages
import numpy as np
import matplotlib.pyplot as plt
import os

#######################
# SET DIRECTORY
#######################  
loaddir = 'data/fig2/'
savdir = 'figures/fig2/'
if not os.path.exists(savdir):
    os.mkdir(savdir)

#######################
# LOAD SIMULATION DATA
#######################
# load main data dictionaries
data = np.load(loaddir+'relearning_results.npy',allow_pickle=True).item()
data0 = np.load(loaddir+'experiment_results.npy',allow_pickle=True).item()
# load parameters
dt = data0['params']['dt']
T = data0['params']['T']
time = data0['params']['time']
tsteps = data0['params']['tsteps']
pulse_length = data0['params']['pulse_length']
manifold_trials = data0['params']['manifold_trials']
stimulus = data0['stimulus']
target = data0['target']
reduced_dim = data0['decoding']['reduced_dim']
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

# FIGURE 1: trajectories
if data0['params']['stimulus_type']=='constant':
    # reconstruct trajectories from velocities
    pos_original = np.zeros(o_original.shape)
    pos_outside = np.zeros(o_outside.shape)
    pos_within = np.zeros(o_within.shape)
    pos_outside_wrong = np.zeros(o_outside.shape)
    pos_within_wrong = np.zeros(o_within.shape)
    for j in range(activity2.shape[1]):
        pos_original[:,j,:] = pos_original[:,j-1,:] + o_original[:,j,:]*dt
        pos_outside[:,j,:] = pos_outside[:,j-1,:] + o_outside[:,j,:]*dt
        pos_within[:,j,:] = pos_within[:,j-1,:] + o_within[:,j,:]*dt
        pos_outside_wrong[:,j,:] = pos_outside_wrong[:,j-1,:] + o_outside_wrong[:,j,:]*dt
        pos_within_wrong[:,j,:] = pos_within_wrong[:,j-1,:] + o_within_wrong[:,j,:]*dt
elif data0['params']['stimulus_type']=='linear':
    # reconstruct trajectories from velocities
    pos_original = o_original
    pos_outside = o_outside
    pos_within = o_within
    pos_outside_wrong = o_outside_wrong
    pos_within_wrong = o_within_wrong
        
plt.figure(figsize=(12,6),dpi=96)
alpha = 1
plt.subplot(2,3,1)
for j in range(manifold_trials):
    plt.plot(pos_within_wrong[j,:,0],pos_within_wrong[j,:,1],col_w,alpha=alpha)
plt.title('Within-manifold')
plt.ylabel('Perturbed')
plt.subplot(2,3,2)
for j in range(manifold_trials):
    plt.plot(pos_outside_wrong[j,:,0],pos_outside_wrong[j,:,1],col_o,alpha=alpha)
plt.title('Outside-manifold')
plt.subplot(2,3,4)
for j in range(manifold_trials):
    plt.plot(pos_within[j,:,0],pos_within[j,:,1],col_w,alpha=alpha)
plt.ylabel('Learned')
plt.subplot(2,3,5)
for j in range(manifold_trials):
    plt.plot(pos_outside[j,:,0],pos_outside[j,:,1],col_o,alpha=alpha)
plt.subplot(2,3,3)
for j in range(manifold_trials):
    plt.plot(pos_original[j,:,0],pos_original[j,:,1],col_ori,alpha=alpha)
plt.title('Original')
plt.savefig(savdir+'fig2A.svg',bbox_inches='tight')

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
plt.savefig(savdir+'fig2B.svg',bbox_inches='tight')


# FIGURE 3: weight change
bins = np.linspace(-0.8,0.8,100)
plt.figure(figsize=(4,3),dpi=96)
plt.hist(w00[non0idx],bins,histtype='step',linewidth=2,color='gray',
         label='Untrained')
plt.hist(w0[non0idx],bins,histtype='step',linewidth=2,color='k',
         label='Initial')
plt.hist(w1w[non0idx],bins,histtype='step',linewidth=2,color=col_w,
         label='WMR')
plt.hist(w1o[non0idx],bins,histtype='step',linewidth=2,color=col_o,
         label='OMR')
plt.legend(loc='upper right')
plt.xlabel('Weight')
plt.ylabel('Histogram')
plt.savefig(savdir+'fig2C_1.svg',bbox_inches='tight')

bins = np.linspace(-0.4,0.4,100)
plt.figure(figsize=(4,3),dpi=96)
plt.hist(w00[non0idx]-w0[non0idx],bins,histtype='step',linewidth=2,color='k',
         label='Untrained -> Initial')
plt.hist(w0[non0idx]-w1w[non0idx],bins,histtype='step',linewidth=2,color=col_w,
         label='Initial -> WMR')
plt.hist(w0[non0idx]-w1o[non0idx],bins,histtype='step',linewidth=2,color=col_o,
         label='Initial -> OMR')
plt.legend(loc='upper right')
plt.xlabel('Weight change')
plt.ylabel('Histogram')
plt.savefig(savdir+'fig2C_2.svg',bbox_inches='tight')

cols = [col_w,col_o] 
plt.figure(figsize=(4,3),dpi=96)
for j in range(2):
    plt.bar(j,np.mean(weights[j]),yerr=np.std(weights[j]),
            color=cols[j])
plt.xticks(range(2),['Within','Outside'])
plt.xlim(-0.5,1.5)
plt.ylim(0,0.05)
plt.ylabel('SD weight change')
plt.savefig(savdir+'fig2C_3.svg',bbox_inches='tight')

# FIGURE 4: explained variance
cols = [col_w,col_o,col_oo] 
plt.figure(figsize=(4,3),dpi=96)
for j in range(explained_variance.shape[-1]):
    plt.bar(j,explained_variance[j]*100,color=cols[j])
plt.xticks(range(explained_variance.shape[-1]),
           ['Initial <-> WMR','Initial <-> OMR','Perturbed <-> OMR'],
           rotation=30,ha='right')
plt.xlim(-0.5,2.5)
plt.ylim(0,100)
plt.ylabel('Manifold overlap (%)')
plt.savefig(savdir+'fig2D.svg',bbox_inches='tight')
