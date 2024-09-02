# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 21:00:24 2024

@author: kaiyi
"""

import time
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
from mnl_env import MNLEnvironment
from langevin_ts import ThompsonSamplingLangevin
from tqdm import tqdm
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
p = comm.Get_size()
if rank==rank:
    start = time.time()
    # global parameters------------------------------------------------------
    max_T = 5000
    tau = 1000 #initial exploration phase
    n_products = 3
    batch_size = 200
    N_t = 50
    eta_scale = 0.01
    psi_inverse = 0.5
    regularization_item = 1
    #------------------------------------------------------------------------
    alphas_input = np.ones([n_products])
    betas_input = np.array([0.1,0.2,0.3])
    
    alphas_true = nn.Parameter(torch.tensor(alphas_input,dtype=torch.float))
    betas_true = nn.Parameter(torch.tensor(betas_input,dtype=torch.float))

    # define instances
    env_model = MNLEnvironment(alphas_true, 
                               betas_true, 
                               el=0, 
                               u=30, 
                               T=max_T)

    ts_langevin = ThompsonSamplingLangevin(tau,
                                        n_products,
                                        batch_size,
                                        N_t,
                                        eta_scale,
                                        psi_inverse,
                                        regularization_item)

    #--------------------------------------------------------------------------
    results_langevin = pd.DataFrame()
    with tqdm(total=max_T, desc=f"Rank {rank}") as pbar:
        for T in tqdm(range(0,max_T)):
            d_langevin = {}
            parameters_langevin = env_model.next_step(ts_langevin)
            
            d_langevin["simple_regret"] = env_model.simple_regret_history
            d_langevin["cum_regret"] = env_model.regret_history

    df_langevin_temp = pd.DataFrame(d_langevin)

    results_langevin = pd.concat([results_langevin, df_langevin_temp], ignore_index=True)
    results_langevin.to_csv(r"/home/ks4n19/MNL_langevin_pytorch/output_data_3products/langevin_random_varTau/B200/results_langevin_"+str(rank+30)+"T"+str(max_T)+"B"+str(batch_size)+"Tau"+str(tau)+".csv", index=False)


    end = time.time()
    
