import numpy as np
import torch
from mnl_model import MultipleMNLModel

class TSLaplace_uniform:
    def __init__(
        self,
        exploration_rate,
        K,
        batch_size,
        regularization_factor_H,
        regularization_factor_mle,
        mle_lr,
        mle_steps,
        M,
    ):
        """Thompson Sampling with Laplace approximation with uniform prior
        Args:
            tau (int): Number of initial exploration rounds
            exploration_rate (float): Multipicative factor on the normal distribution variance
            K (int): Number of products
            batch_size (int): Batch size
            regularization_factor_H (float): Regularization factor on the Hessian matrix
            regularization_factor_mle (float): Regulatization factor for MLE
            mle_lr (float): Learning rate of MLE
            M(float):upper bound of MNL parameters
        """
        self.K = K
        self.exploration_rate = exploration_rate
        self.H = regularization_factor_H * torch.eye(2 * K)
        self.batch_size = batch_size

        self.regularization_factor_H = regularization_factor_H
        self.regularization_factor_mle = regularization_factor_mle
        self.mle_lr = mle_lr
        self.last_grad = None
        self.mle_steps = mle_steps
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
            env (Environment): Environment
        Returns:
            optimal action (price), and additional data
        """
        
        def hessian_for_sample(model, p):
            q = model.mean_demand(p).unsqueeze(1)
            mult = torch.cat([torch.ones(self.K), -p]).unsqueeze(0)
            H_A = torch.diag(-q[:, 0]) + q.matmul(q.T)
            H_A_repeat = H_A.repeat(2, 2)
            return -(mult * H_A_repeat) * mult.T
        
        if env.t < self.batch_size:
            model_post = MultipleMNLModel(
                self.alpha_post,
                self.beta_post,
            )
            
            p = model_post.max_mean_demand_price()
            data = {}
        else:
            
            if (env.t - self.batch_size) % self.batch_size == 0:
                mle_steps = self.mle_steps
                
                (
                    self.alpha_bar,
                    self.beta_bar,
                    self.last_grad,
                ) = MultipleMNLModel.mle(
                    env.choice_history[: env.t],
                    env.price_history[: env.t],
                    env.model.K,
                    alphas_s_old=self.alpha_bar,
                    betas_s_old=self.beta_bar,
                    steps=mle_steps,
                    lr=self.mle_lr,
                    regularization=self.regularization_factor_mle,
                )
                self.theta_bar = torch.cat(
                    [self.alpha_bar, self.beta_bar]
                )
                self.model_bar = MultipleMNLModel(
                    self.alpha_bar,
                    self.beta_bar,
                )
                if env.t == self.batch_size:
                    for i, p in enumerate(env.price_history):
                        self.H += hessian_for_sample(
                            self.model_bar, p
                        )  
                self.H_fixed = self.H.clone()  

                # sampling from approximated posterior distribution
                theta_post_dist = (
                    torch.distributions.multivariate_normal.MultivariateNormal(
                        self.theta_bar,
                        precision_matrix=((self.exploration_rate**-2) * self.H_fixed),
                        validate_args=False,
                    )
                )
                theta_post = theta_post_dist.sample()
                
                self.alpha_post = theta_post[: self.K]
                self.beta_post = theta_post[self.K :]

            model_post = MultipleMNLModel(
                self.alpha_post,
                self.beta_post,
            )
            
            p = model_post.max_mean_demand_price()
            self.H += hessian_for_sample(self.model_bar, p)
            data = {
                "alpha_bar": self.theta_bar[: self.K],
                "beta_bar": self.theta_bar[self.K : ],
            }
            if self.last_grad is not None:
                data["last_grad"] = self.last_grad     
        
        return p, data
