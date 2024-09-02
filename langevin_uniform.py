import numpy as np
import torch
from mnl_model import MultipleMNLModel


class TSLangevin_uniform:
    def __init__(
        self,
        K,
        batch_size,
        N_t,
        eta_scale,
        psi_inverse,
        regularization_factor_mle,
        M,
        
    ):
        """Langevin Sampling for NonContextual Prices and Promotions Experiments

        Args:
            K (int): Number of products
            batch_size (int): Batch size
            N_t (int): Number of MCMC steps
            eta_scale (float): Step size
            regularization_factor_mle (float): Regulatization factor for MLE
            fix_model (boolean): Whether to fix the model during the experimentation phase or resample every round
            psi_inverse (float): Multipicative factor on the noise variance
            M(float): upper bound of MNL model parameters
        """

        self.K = K
        self.batch_size = batch_size
        self.regularization_factor_mle = regularization_factor_mle
        self.N_t = N_t
        self.eta_scale = eta_scale
        self.psi_inverse = psi_inverse
        self.last_grad = None
        
        self.M = M
        
        #sample the initial values from non-informative priors
        self.alpha_post = torch.distributions.Uniform(-M, M).sample((K,))
        self.beta_post = torch.distributions.Uniform(0, M).sample((K,))
        # set the inital mle parameters
        self.alpha_bar = self.alpha_post
        self.beta_bar = self.beta_post

    def next_price(self, env):
        """What price to play at the current state of the environment

        Args:
            env (Environment):
        
        Returns:
           prices and additional data
        """
        if env.t < self.batch_size:
            model_bar = MultipleMNLModel(
                self.alpha_post,
                self.beta_post,
            )
            
            p = model_bar.max_mean_demand_price()
            data = {}
            
        else:
            if (env.t - self.batch_size) % self.batch_size == 0:
                
                last_time = env.t - ((env.t - self.batch_size) % self.batch_size)
                
                N_t = self.N_t
                result = MultipleMNLModel.langevin_step(
                    env.choice_history[:last_time],
                    env.price_history[:last_time],
                    K=self.K,
                    N_t=N_t,
                    eta=self.eta_scale / last_time,
                    psi_inverse=self.psi_inverse,
                    alphas_s_old=self.alpha_bar,
                    betas_s_old=self.beta_bar,
                    regularization=self.regularization_factor_mle,)
                
                self.alpha_bar, self.beta_bar, self.last_grad = result

                self.model_bar = MultipleMNLModel(
                    self.alpha_bar,
                    self.beta_bar,
                )
            p = self.model_bar.max_mean_demand_price()
            
            data = {
                "alpha_bar": self.alpha_bar,
                "beta_bar": self.beta_bar,
            }
            

        return p, data# addtional_information