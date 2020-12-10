#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create plots for Figure 3.
"""

# import packages
import numpy as np
import matplotlib.pyplot as plt
import os

#######################
# SET DIRECTORY
#######################  
loaddir = 'data/fig3/'
savdir = 'figures/fig3/'
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
feedback_option = data['relearning']['feedback_option']
sd_fac = data['relearning']['sd_fac']
percent_silent = data['relearning']['percent_silent']
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

if feedback_option=='B':
    # FIGURE 1: performance
    plt.figure(figsize=(4,3),dpi=96)
    plt.scatter(sd_fac,performance[2],c=col_w,label='WMR')
    plt.scatter(sd_fac,performance[4],c=col_o,label='OMR')
    plt.xlim(0,5)
    plt.ylim(0,1.2)
    plt.legend()
    plt.ylabel('MSE')
    plt.xlabel('Noise factor')
    plt.savefig(savdir+'performance.svg',bbox_inches='tight')
    
    # FIGURE 2: explained variance
    plt.figure(figsize=(4,3),dpi=96)
    plt.scatter(sd_fac,explained_variance[0]*100,c=col_w,label='I-WMR')
    plt.scatter(sd_fac,explained_variance[1]*100,c=col_o,label='I-OMR')
    plt.scatter(sd_fac,explained_variance[2]*100,c=col_oo,label='P-OMR')
    plt.xlim(0,5)
    plt.ylim(0,100)
    plt.legend()
    plt.ylabel('Manifold overlap (%)')
    plt.xlabel('Noise factor')
    plt.savefig(savdir+'manifold_overlap.svg',bbox_inches='tight')
elif feedback_option=='C':
    x = (1-percent_silent)*100
    # FIGURE 1: performance
    plt.figure(figsize=(4,3),dpi=96)
    plt.scatter(x,performance[2],c=col_w,label='WMR')
    plt.scatter(x,performance[4],c=col_o,label='OMR')
    plt.xlim(0,100)
    plt.ylim(0,1.2)
    plt.ylabel('MSE')
    plt.legend()
    plt.xlabel('Portion of neurons\nreceiving an error signal (%)')
    plt.savefig(savdir+'performance.svg',bbox_inches='tight')
    
    # FIGURE 2: explained variance
    plt.figure(figsize=(4,3),dpi=96)
    plt.scatter(x,explained_variance[0]*100,c=col_w,label='I-WMR')
    plt.scatter(x,explained_variance[1]*100,c=col_o,label='I-OMR')
    plt.scatter(x,explained_variance[2]*100,c=col_oo,label='P-OMR')
    plt.xlim(0,100)
    plt.legend()
    plt.ylim(0,100)
    plt.ylabel('Manifold overlap (%)')
    plt.xlabel('Portion of neurons\nreceiving an error signal (%)')
    plt.savefig(savdir+'manifold_overlap.svg',bbox_inches='tight')
elif feedback_option=='D':
    x = (1-percent_silent)*100
    # FIGURE 1: performance
    plt.figure(figsize=(4,3),dpi=96)
    plt.scatter(x,performance[2],c=col_w,label='WMR')
    plt.scatter(x,performance[4],c=col_o,label='OMR')
    plt.legend()
    plt.xlim(0,100)
    plt.ylim(0,1.2)
    plt.ylabel('MSE')
    plt.xlabel('Portion of plastic weights (%)')
    plt.savefig(savdir+'performance.svg',bbox_inches='tight')
    
    # FIGURE 2: explained variance
    plt.figure(figsize=(4,3),dpi=96)
    plt.scatter(x,explained_variance[0]*100,c=col_w,label='I-WMR')
    plt.scatter(x,explained_variance[1]*100,c=col_o,label='I-OMR')
    plt.scatter(x,explained_variance[2]*100,c=col_oo,label='P-OMR')
    plt.xlim(0,100)
    plt.ylim(0,100)
    plt.legend()
    plt.ylabel('Manifold overlap (%)')
    plt.xlabel('Portion of plastic weights (%)')
    plt.savefig(savdir+'manifold_overlap.svg',bbox_inches='tight')
