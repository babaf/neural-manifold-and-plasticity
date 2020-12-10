#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation script for Figure 5 [Part 3].
 -> incremental outside learning
"""
# import packages
import numpy as np
import os
import sklearn.linear_model as lm

#######################
# SET DIRECTORY
#######################  
# random seed for this simulation
seed_id = 0
# set directory where results should be saved
savdir = 'data/fig5/'
savdirfigs = 'figures/fig5/'
if not os.path.exists(savdir):
    os.mkdir(savdir)
if not os.path.exists(savdirfigs):
    os.mkdir(savdirfigs)

#######################
# PARAMETERS
#######################    
np.random.seed(seed_id)
dt = 0.01
T = 2
time = np.arange(0,T,dt)
tsteps = len(time)
pulse_length = int(0.2/dt)
targets = 6
stimulus_type = 'constant' # constant, linear, normal
target_max = 0.2 # 0.2 or 0.01
# initial network learning
learning1_trials = 80
delta = 20.
# manifold calculation
manifold_trials = 50
reduced_dim = 10
# relearning
relearning_trials = 80
deltarec = 20.
alphas = np.array([0.25,0.5,0.75,1.])

##################
# TOOLBOX
#################
class RNN(object):
    """
    Class implementing a recurrent network (not following Dale's law).

    Parameters:
    -----------
    * N: number of neurons
    * N_in: how many inputs can the network have
    * N_out: how many neurons are recorded by external device
    * g: recurrent coupling strength
    * p: connection probability
    * tau: neuron time constant
    * dt: set dt for simulation
    * delta: defines initial learning rate for FORCE
    * P_plastic: how many neurons are plastic in the recurrent network
    """
    def __init__(self, N=800, g=1.5, p=0.1, tau=0.1, dt=0.01,
                 N_in=6):
        # set parameters
        self.N = N
        self.g = g
        self.p = p
        self.K = int(p*N)
        self.tau = tau
        self.dt = dt
        
        # create recurrent W
        mask = np.random.rand(self.N,self.N)<self.p
        np.fill_diagonal(mask,np.zeros(self.N))
        self.mask = mask
        self.W = self.g / np.sqrt(self.K) * np.random.randn(self.N,self.N) * mask
        
        # create Win and Wout
        self._N_in = N_in
        self.W_in = (np.random.rand(self.N, self._N_in)-0.5)*2.
        
    @property
    def N_in(self):
        return self._N_in

    @N_in.setter
    def N_in(self, value):
        self._N_in = value
        self.W_in = (np.random.rand(self.N, self._N_in)-0.5)*2.

    def save(self,filename):
        np.savez(
            filename,
            N = self.N,
            K = self.K,
            tau = self.tau,
            g = self.g,
            p = self.p,
            dt = self.dt,
            W_in = self.W_in,
            W = self.W,
            N_in = self._N_in,   
        )
        
    def load(self,filename):
        net = np.load(filename+'.npz')
        self.N = int(net['N'])
        self.dt = float(net['dt'])
        self.K = int(net['K'])
        self.tau = float(net['tau'])
        self.g = float(net['g'])
        self.p = float(net['p'])
        self.W_in = net['W_in']
        self.W = net['W']  
        self._N_in = int(net['N_in'])

    def update_activation(self):
        self.z = np.tanh(self.r)
        
    def update_neurons(self,ext):
        self.r = self.r + self.dt/self.tau * \
             (-self.r + np.dot(self.W, self.z) + np.dot(self.W_in,ext))
            
        self.update_activation()
        
    def simulate(self, T, ext=None, r0=None):

        # define time
        time = np.arange(0,T,self.dt)
        tsteps = int(T/self.dt)

        # create input in case no input is given
        if ext is None:
            ext = np.zeros((tsteps,self.N_in))

        # check if input has the right shape
        if ext.shape[0]!=tsteps or ext.shape[1]!=self.N_in:
            print('ERROR: stimulus shape should be (time x number of input nodes)')
            return
            
        # set initial condition
        if r0 is None:
            self.r = (np.random.rand(self.N)-0.5)*2.
        else:
            self.r = r0
        self.update_activation()
        
        # start simulation
        record_r = np.zeros((tsteps,self.N))
        record_r[0,:] = self.r
        for i in range(1,tsteps):
            self.update_neurons(ext=ext[i])
            # store activity 
            record_r[i,:] = self.r

        return time, record_r, np.tanh(record_r)

    def relearn(self, trials, ext, ntstart, decoder, feedback, target, delta=1.,
                wplastic=None):
        tsteps = ext.shape[1]
        # set up learning
        if wplastic is None:
            self.W_plastic = [np.where(self.W[i,:]!=0)[0] for i in range(self.N)]
        else:
            self.W_plastic = wplastic
        self.P = [1./delta*np.eye(len(self.W_plastic[i])) for i in range(len(self.W_plastic))]
        order = np.random.choice(range(ext.shape[0]),trials,replace=True)
        record_loss = np.zeros(trials)
        # start learning
        for t in range(trials):
            loss = 0.
            self.r = (np.random.rand(self.N)-0.5)*2.
            self.update_activation()
            # loop over time
            for i in range(1,tsteps):
                self.update_neurons(ext=ext[order[t],i])
                # learning part
                if i > ntstart and i%2==0:
                    c = decoder @ self.z
                    errc = c-target[order[t],i]
                    err1 = feedback @ errc
                    loss += np.mean(err1**2) 
                    # ONLY RECURRENT WEIGHT UPDATE
                    for j in range(self.N): 
                        z_plastic = self.z[self.W_plastic[j]]
                        pz = np.dot(self.P[j], z_plastic)
                        norm = (1. + np.dot(z_plastic.T,  pz))
                        self.P[j] -= np.outer(pz, pz)/norm
                        self.W[j, self.W_plastic[j]] -= err1[j] * pz / norm  
            record_loss[t] = loss
            print('Loss in Trial %d is %.5f'%(t+1,loss))        
        return record_loss

    def calculate_manifold(self, trials, ext, ntstart):
        tsteps = ext.shape[1]
        T = self.dt*tsteps
        points = (tsteps-ntstart)
        activity = np.zeros((points*trials,self.N))
        order = np.random.choice(range(ext.shape[0]),trials,replace=True)
        for t in range(trials):
            time, r, z = self.simulate(T,ext[order[t]])
            activity[t*points:(t+1)*points,:] = z[ntstart:,:]
        cov = np.cov(activity.T)
        ev,evec = np.linalg.eig(cov)
        pr = np.round(np.sum(ev.real)**2/np.sum(ev.real**2)).astype(int)
        xi = activity @ evec.real
        return activity,cov,ev.real,evec.real,pr,xi,order
    
####################
# HELPER FUNCTIONS #
####################
def create_stimulus(tsteps,pulse_steps,n_targets=6,amplitude=1.,twod=False):
    # create stimulus
    stimulus = np.zeros((n_targets,tsteps,n_targets))
    if twod:
        phis = np.linspace(0,2*np.pi,targets,endpoint=False)
        for j in range(stimulus.shape[0]):   
            stimulus[j,:pulse_length,0] = amplitude*np.cos(phis[j])
            stimulus[j,:pulse_length,1] = amplitude*np.sin(phis[j])
            stimulus[j,:pulse_length,2:] = 0 
    else:
        for j in range(n_targets):
            stimulus[j,:pulse_steps,j] = amplitude
    return stimulus

def create_target(tsteps,pulse_steps,n_targets=6,stype='constant',target_max=0.2):
    # create target trajectories
    phis = np.linspace(0,2*np.pi,n_targets,endpoint=False)
    rs = np.zeros(tsteps)
    # TARGET DEFINITION
    if stype=='linear':
        # OPTION 1) linear for position
        rs[pulse_steps:] = np.linspace(0,target_max,tsteps-pulse_steps) 
    elif stype=='normal':
        # OPTION 2) Gaussian speed profile
        xx = np.linspace(0,target_max,tsteps-pulse_steps)
        mu = target_max/2.
        sigma = target_max/8.
        rs[pulse_steps:] = target_max*np.exp(-(xx-mu)**2/(2*sigma**2))
    elif stype=='constant':
        # OPTION 3) constant speed
        rs[pulse_steps:] = np.ones(tsteps-pulse_steps)*target_max    
    traj = np.zeros((n_targets,tsteps,2))
    for j in range(n_targets):
        traj[j,:,0] = rs*np.cos(phis[j])
        traj[j,:,1] = rs*np.sin(phis[j])
    
    return traj
  
def decoder_training(inputP,target,order):
    X = np.zeros((inputP.shape[0]*inputP.shape[1], inputP.shape[-1]))
    Y = np.zeros((inputP.shape[0]*inputP.shape[1], 2))
    for j in range(inputP.shape[0]):
        X[j*inputP.shape[1]:(j+1)*inputP.shape[1],:] = inputP[j]
        Y[j*inputP.shape[1]:(j+1)*inputP.shape[1],:] = target[order[j]]
    reg = lm.LinearRegression()
    reg.fit(X,Y)
    y = reg.predict(X)
    mse = np.mean((y-Y)**2)
#    print('MSE = %.4f'%mse)
    return reg.coef_,mse

def feedback_model_training(X,Y):
    reg = lm.LinearRegression() 
    reg.fit(X,Y)                               
    return reg.coef_

def get_cost(result,target,order):
    cost = 0
    for j in range(result.shape[0]):
        error = result[j,:,:]-target[order[j],:,:]
        cost += np.mean(error**2)
    return cost

def select_random_perturbations(activity2,D,P):    
    runs = 200
    cost = np.zeros((runs,2))
    for j in range(runs):
        # set random seed for reproduction
        np.random.seed(j)
        # within-manifold perturbation
        perm_matrix_within = np.eye(reduced_dim)
        np.random.shuffle(perm_matrix_within)
        D_permute = D.copy()
        D_permute[:reduced_dim,:reduced_dim] = D[:reduced_dim,:reduced_dim] @ perm_matrix_within
        T_within = D_permute @ P
        result_within = activity2 @ T_within.T
        cost[j,0] = get_cost(result_within,target[:,pulse_length:,:],order)
        # set random seed for reproduction
        np.random.seed(j)
        # outside-manifold perturbation
        perm_matrix_outside = np.eye(network.N)
        np.random.shuffle(perm_matrix_outside)
        P_permute = P @ perm_matrix_outside
        T_outside = D @ P_permute
        result_outside = activity2 @ T_outside.T
        cost[j,1] = get_cost(result_outside,target[:,pulse_length:,:],order)
    # select closest to 1 perturbations
    dif = abs(cost-np.mean(cost))
    idx = np.argsort(dif,axis=0)
    return idx # first column seeds for within, second column seeds for outside

#%% INITIAL TRAINING
    
# create stimulus
stimulus = create_stimulus(tsteps, pulse_length, n_targets=targets, twod=False)  

# create target
target = create_target(tsteps, pulse_length, n_targets=targets,
                                  stype=stimulus_type, target_max=target_max)  
    
########################
# LOAD from BC run
#######################   
data = np.load(savdir+'results.npy',allow_pickle=True).item()
T_outside = data['T_outside']
alphas = data['alphas']
T = data['T']
D = data['D']
manifold0 = np.load(savdir+'manifold.npy',allow_pickle=True).item()

# create network
network = RNN()          
network.load(savdir+'network')
w1 = np.load(savdir+'W_stabilized.npy')
network.W = w1.copy()


#%% relearning
###############################
# START INCREMENTAL LEARNING
###############################
w_outside = []
fb_outside_rec = []
cost_outside_retrained = []
cost_inc_retrained = []
fb_corr = []
exp_var = []
exp_var_new = []
manifold = {}
for j in range(len(alphas)):
    print('Run %d'%(j+1))
    # set incremental BCI
    Tinc = (1-alphas[j])*T + (alphas[j])*T_outside
    Pinc = np.linalg.pinv(D) @ Tinc
    # RETRAIN
    network.W = np.copy(w1) # start from initial training setup
    # load activity from last run and train feedback on this
    if j==0:
        act = manifold0['original']['activity']
    else:
        act = manifold['outside'+str(j)]['activity']
    fb_outside = feedback_model_training(act @ Tinc.T ,act)
    
    # recurrent relearning
    loss_outside = network.relearn(relearning_trials, stimulus, pulse_length, 
                      Tinc, fb_outside, target, delta=deltarec)
    w_outside.append(np.copy(network.W))
    
    # TESTING 
    # simulate new activiy data 
    activityt,cov,ev,evec,pr,xi,order = network.calculate_manifold(trials=manifold_trials, 
                                                                  ext=stimulus, ntstart=pulse_length)
    manifold.update({'outside'+str(j+1):{'activity':activityt,'order':order,
                           'cov':cov,'ev':ev,'evec':evec,'pr':pr}})
    # reshape
    activity2t = activityt.reshape(manifold_trials,-1,network.N)
    # calculate output
    result_outside = activity2t @ T_outside.T
    cost_outside_retrained.append(get_cost(result_outside,target[:,pulse_length:,:],order))
    # calculate output
    result_outside = activity2t @ Tinc.T
    cost_inc_retrained.append(get_cost(result_outside,target[:,pulse_length:,:],order))

    # calculate explained variance
    proj1 = manifold0['original']['evec'].T @ cov @ manifold0['original']['evec']
    proj2 = manifold0['original']['evec'].T @ manifold0['original']['cov'] @ manifold0['original']['evec']
    proj3 = Pinc @ cov @ Pinc.T     
    match1 = np.trace(proj1[:reduced_dim])/np.trace(cov)
    match2 = np.trace(proj2[:reduced_dim])/np.trace(manifold0['original']['cov'])
    match3 = np.trace(proj3[:reduced_dim])/np.trace(cov)
    exp_var.append(match1/match2)
    exp_var_new.append(match3/match2)

    # only for fb learning
    c = np.corrcoef(np.linalg.pinv(Tinc).ravel(),fb_outside.ravel())
    fb_corr.append(c[0,1])
    print('FB corr=%.2f'%c[0,1])
    
    print('Goal: %.2f --- Outside: %.2f --- Incremental: %.2f'%(0.0,
                    cost_outside_retrained[-1],cost_inc_retrained[-1]))
    print('Manifold match old: %.2f --- Manifold match new: %.2f'%(exp_var[-1],exp_var_new[-1]))

# save it
temp_data = {'alphas':alphas,'fb_corr':fb_corr,'exp_var':exp_var,
             'exp_var_new':exp_var_new,
             'cost_outside_retrained':cost_outside_retrained,
             'cost_inc_retrained':cost_inc_retrained,
             'T':T,'D':D,'T_outside':T_outside}
np.save(savdir+'results_GHI',temp_data)
np.save(savdir+'manifold_GHI',manifold) 
np.save(savdir+'wOM_GHI',w_outside)
np.save(savdir+'wfbOM_GHI',fb_outside)

#%% plot it
col_o = '#4a5899'
col_oi = '#7652aa'
col_oo = '#3CAEA3'

import matplotlib.pyplot as plt

plt.figure(figsize=(4,3),dpi=96)
plt.plot(alphas,fb_corr,'k')
plt.xlabel('$\\alpha$')
plt.ylabel('Corr. coef.\nfeedback learning')
plt.savefig(savdirfigs+'Gfeedbacklearning.svg',bbox_inches='tight')

plt.figure(figsize=(4,3),dpi=96)
plt.plot(alphas,cost_outside_retrained,col_o,label='OMP')
plt.plot(alphas,cost_inc_retrained,col_oi,label='iOMP')
plt.xlabel('$\\alpha$')
plt.ylabel('MSE')
plt.legend()
plt.savefig(savdirfigs+'Hperformance.svg',bbox_inches='tight')

plt.figure(figsize=(4,3),dpi=96)
plt.plot(alphas,exp_var,col_o,label='I-iOMR')
plt.plot(alphas,exp_var_new,col_oo,label='P-iOMR')
plt.legend()
plt.xlabel('$\\alpha$')
plt.ylabel('Manifold overlap (%)')
plt.savefig(savdirfigs+'Imanifoldoverlap.svg',bbox_inches='tight')