import time
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
from mnl_env import MNLEnvironment
from laplace_uniform import TSLaplace_uniform
from tqdm import tqdm
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
p = comm.Get_size()
if rank==rank:
    start = time.time()

    # global parameters------------------------------------------------------
    max_T = 20000
    n_products = 9
    batch_size = 200
    exploration_rate = 0.5
    regularization_factor_H = 1
    regularization_factor_mle = 1
    mle_lr = 0.1
    mle_steps = 100
    low = 0
    upper = 40
    parameter_bound = 5
    #----------------------------------------------------------------------------
    alphas_input = np.load(r"/home/ks4n19/MNL_langevin_pytorch/nielsen_data/nielsen_alphas_9.npy")[:9]
    betas_input = -np.load(r"/home/ks4n19/MNL_langevin_pytorch/nielsen_data/nielsen_betas_9.npy")[:9]/32

    alphas_true = nn.Parameter(torch.tensor(alphas_input,dtype=torch.float))
    betas_true = nn.Parameter(torch.tensor(betas_input,dtype=torch.float))

    
    #---------------------------------------------------------------------------
    # define instances
    env_model = MNLEnvironment(alphas_true, 
                               betas_true, 
                               low, 
                               upper, 
                               max_T,)

    ts_laplace = TSLaplace_uniform(
                            exploration_rate,
                            n_products,
                            batch_size,
                            regularization_factor_H,
                            regularization_factor_mle,
                            mle_lr,
                            mle_steps,
                            parameter_bound,
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
    results_laplace.to_csv(r"/home/ks4n19/mnl_pytorch_code/coffee/laplace_uniform/B200/results_laplace_"+str(rank+30)+"T"+str(max_T)+"B"+str(batch_size)+".csv", index=False)
    
    end = time.time()