#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation script for Figure 2.
 -> perfect feedback signal
"""
# import packages
import numpy as np
import os
import sklearn.linear_model as lm

#######################
# SET DIRECTORY
#######################  
# random seed for this simulation
seed_id = 2
# set directory where results should be saved
savdir = 'data/fig2/'
if not os.path.exists(savdir):
    os.mkdir(savdir)
    
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

# network parameters
N = 800
g = 1.5
p = 0.1
tau = 0.1

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
    
# create network
network = RNN(N=N,g=g,p=p,tau=tau,dt=dt,N_in=targets)            
network.save(savdir+'network')
np.save(savdir+'W_initial',network.W)
w0 = network.W.copy()

#######################
# LEARNING 1
#######################
# create random decoder for initial learning
decoder = np.random.randn(2,network.N)
initial_decoder_fac = 0.04 * (target_max/0.2)
decoder *= (initial_decoder_fac / np.linalg.norm(decoder))
feedback = np.linalg.pinv(decoder)
stabilize_loss = network.relearn(learning1_trials, stimulus, pulse_length, 
                      decoder, feedback, target, delta=delta)
np.save(savdir+'W_stabilized',network.W)
w1 = network.W.copy()

#######################
# MANIFOLD
#######################
activity,cov,ev,evec,pr,xi,order = network.calculate_manifold(trials=manifold_trials, 
                                                              ext=stimulus, ntstart=pulse_length)
# rearrange the activity a bit
activity2 = activity.reshape(manifold_trials,-1,network.N)
xi2 = xi.reshape(manifold_trials,-1,network.N)

#######################
# LEARNING 2
#######################
W_bci4,l4 = decoder_training(xi2[:,:,:reduced_dim],target[:,pulse_length:,:],order)

#######################
# SAVING
#######################
data = {'params':{'dt':dt,
                 'T':T,
                 'time':time,
                 'tsteps':tsteps,
                 'pulse_length':pulse_length,
                 'manifold_trials':manifold_trials,
                 'target_max':target_max,
                 'stimulus_type':stimulus_type,
                 'N':N,'tau':tau,'g':g,'p':p},
    'stimulus':stimulus,
    'target':target,
    'stabilizing':{'learning_trials':learning1_trials,
                   'delta':delta,
                 'decoder':decoder,
                 'feedback':feedback,
                 'stabilize_loss':stabilize_loss},
    'manifold':{'activity':activity,
                'activity2':activity2,
                'xi':xi,
                'xi2':xi2,
                'cov':cov,'ev':ev,'evec':evec,'pr':pr,'order':order},
    'decoding':{ 'reduced_dim':reduced_dim,
                 'weights':W_bci4,'loss':l4}}
np.save(savdir+'experiment_results',data)

#%% PERTURBATION

# original transformation
P = evec.real.T
D = np.zeros((2,network.N))
D[:,:reduced_dim] = W_bci4
T = D @ P
result_original = activity2 @ T.T
cost_original = get_cost(result_original,target[:,pulse_length:,:],order)
  
# select equal perturbation result shuffling seeds
idx_shuffling_seeds = select_random_perturbations(activity2,D,P) # within, outside
within_seed = idx_shuffling_seeds[0,0] # 79
outside_seed = idx_shuffling_seeds[0,1] # 189

#######################
# PERTURBATION
#######################  
# within-manifold perturbation
np.random.seed(within_seed)
perm_matrix_within = np.eye(reduced_dim)
np.random.shuffle(perm_matrix_within)
D_permute = D.copy()
D_permute[:reduced_dim,:reduced_dim] = D[:reduced_dim,:reduced_dim] @ perm_matrix_within
T_within = D_permute @ P
result_within = activity2 @ T_within.T
cost_within = get_cost(result_within,target[:,pulse_length:,:],order)

# outside-manifold perturbation
np.random.seed(outside_seed)
perm_matrix_outside = np.eye(network.N)
np.random.shuffle(perm_matrix_outside)
P_permute = P @ perm_matrix_outside
T_outside = D @ P_permute
result_outside = activity2 @ T_outside.T
cost_outside = get_cost(result_outside,target[:,pulse_length:,:],order)

#%% RELEARNING

# set up readout and feedback
decoder_outside = T_outside
decoder_within = T_within
# FEEBACK OPTION 1: perfect teacher
feedback_within = np.linalg.pinv(decoder_within) 
feedback_outside = np.linalg.pinv(decoder_outside)

# RETRAIN
network.W = np.copy(w1) 
loss_within = network.relearn(relearning_trials, stimulus, pulse_length, 
                      decoder_within, feedback_within, target, delta=deltarec)
w_within = np.copy(network.W)

network.W = np.copy(w1)
loss_outside = network.relearn(relearning_trials, stimulus, pulse_length, 
                      decoder_outside, feedback_outside, target, delta=deltarec)
w_outside = np.copy(network.W)


#######################
# TESTING
#######################  
# set weights
network.W = w_within
# simulate new activiy data 
activity,cov,ev,evec,pr,xi,order = network.calculate_manifold(trials=manifold_trials,
                                                              ext=stimulus, ntstart=pulse_length)
manifold = {'within':{'activity':activity,'xi':xi,'order':order,
                       'cov':cov,'ev':ev,'evec':evec,'pr':pr}}
# reshape
activity2 = activity.reshape(manifold_trials,-1,network.N)
# calculate output
result_within = activity2 @ T_within.T
cost_within_retrained = get_cost(result_within,target[:,pulse_length:,:],order)

# set weights
network.W = w_outside
# simulate new activiy data 
activity,cov,ev,evec,pr,xi,order = network.calculate_manifold(trials=manifold_trials, 
                                                              ext=stimulus, ntstart=pulse_length)
manifold.update({'outside':{'activity':activity,'xi':xi,'order':order,
                       'cov':cov,'ev':ev,'evec':evec,'pr':pr}})
# reshape
activity2 = activity.reshape(manifold_trials,-1,network.N)
# calculate output
result_outside = activity2 @ T_outside.T
cost_outside_retrained = get_cost(result_outside,target[:,pulse_length:,:],order)

#####################
# SAVE
#####################
manifold.update({'original':data['manifold']})
dic = {'relearning':{
           'loss_outside':loss_outside,'loss_within':loss_within,
           'feedback_within':feedback_within,'feedback_outside':feedback_outside,
           'relearning_trials':relearning_trials,'delta':delta,
           'cost_within_retrained':cost_within_retrained,
           'cost_outside_retrained':cost_outside_retrained,
           'cost_within':cost_within,'cost_outside':cost_outside,
           'cost_original':cost_original},
       'manifold':manifold,
       'perturbations':
           {'within_seed':within_seed,'outside_seed':outside_seed,
            'P':P,'D':D,'T':T,'perm_matrix_outside':perm_matrix_outside,
            'perm_matrix_within':perm_matrix_within,'P_permute':P_permute,
            'D_permute':D_permute,'T_outside':T_outside,'T_within':T_within}}

np.save(savdir+'relearning_results',dic)
np.save(savdir+'W_within',w_within)
np.save(savdir+'W_outside',w_outside)

