# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 20:50:07 2024

@author: kaiyi
"""

import time
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
from mnl_env import MNLEnvironment
from laplace_ts import ThompsonSamplingLaplace
from tqdm import tqdm
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
p = comm.Get_size()
if rank==rank:
    start = time.time()

    # global parameters------------------------------------------------------
    max_T = 5000
    tau = 10 #initial exploration phase
    n_products = 3
    batch_size = 200
    exploration_rate = 0.5
    regularization_factor_H = 1
    regularization_factor_mle = 1
    mle_lr = 0.1
    mle_steps = 100
    l = 0
    u = 30
    #----------------------------------------------------------------------------
    alphas_input = np.ones(n_products)
    betas_input = np.array([0.1,0.2,0.3])
    
    alphas_true = nn.Parameter(torch.tensor(alphas_input,dtype=torch.float))
    betas_true = nn.Parameter(torch.tensor(betas_input,dtype=torch.float))

    
    #---------------------------------------------------------------------------
    # define instances
    env_model = MNLEnvironment(alphas_true, 
                               betas_true, 
                               el=0, 
                               u=30, 
                               T=max_T)

    ts_laplace = ThompsonSamplingLaplace(tau,
                            exploration_rate,
                            n_products,
                            batch_size,
                            regularization_factor_H,
                            regularization_factor_mle,
                            mle_lr,
                            mle_steps,
                            )
    #--------------------------------------------------------------------------
    results_laplace = pd.DataFrame()
    with tqdm(total=max_T, desc=f"Rank {rank}") as pbar:
        
        for T in tqdm(range(0,max_T)):
            d_laplace = {}
            parameter_laplace = env_model.next_step(ts_laplace)
            
            d_laplace["simple_regret"] = env_model.simple_regret_history
            d_laplace["cum_regret"] = env_model.regret_history

    
        
    df_laplace_temp = pd.DataFrame(d_laplace)

    results_laplace = pd.concat([results_laplace, df_laplace_temp], ignore_index=True)
    results_laplace.to_csv(r"/home/ks4n19/MNL_langevin_pytorch/output_data_3products/laplace_random_varTau/B200/results_laplace_"+str(rank+20)+"_"+str(max_T)+"_"+str(tau)+".csv", index=False)
    
    end = time.time()
    
    


