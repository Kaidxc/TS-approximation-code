import numpy as np
import torch


class MultipleMNLModel:
    def __init__(self, 
                 alphas, 
                 betas,  
                 el=0, 
                 u=40,):
        """Multinomial Logistic Demand Model
        Args:

            alphas (torch.tensor): Alpha parameters
            betas (torch.tensor): Beta parameters
            gammas (torch.tensor): Gamma parameters
            el (float, optional): Lower bound of prices. Defaults to 0.
            u (float, optional): Upper bound of prices. Defaults to 40.
        """
        self.K = alphas.shape[0]
        self.alphas = alphas
        self.betas = betas
        self.el = el
        self.u = u

    def U_f(self, p):
        """Utility function for products at specific prices and promotions

        Args:
            p (torch.tensor): Prices

        Returns:
            torch.tensor: Utilities
        """
        U = self.alphas - self.betas * p 
        d = 1 if len(p.shape) == 1 else (p.shape[0], 1)
        return torch.cat([torch.zeros(d), U], dim=len(p.shape) - 1)

    def Q_f(self, p):
        """choice probabilities (outside option is included) for products at specific prices

        Args:
            p (torch.tensor): Prices

        Returns:
            torch.tensor: Quantities (sum to 1)
        """
        if len(p.shape) == 1:
            return torch.softmax(self.U_f(p), 0)
        else:
            return torch.softmax(self.U_f(p), 1)
        
    def mean_demand(self, p):
        """Mean demand for products without outside option at specific prices

        Args:
            p (torch.tensor): Prices
        Returns:
            torch.tensor: Quantities without outside option
        """
        if len(p.shape) == 1:
            return self.Q_f(p)[1:]
        else:
            return self.Q_f(p)[:, 1:]

    def mean_profit(self, p):
        """Mean profit at specific prices

        Args:
            p (torch.tensor): Prices

        Returns:
            torch.tensor: Mean profit
        """
        demand = self.mean_demand(p)
        return demand.dot(p)

    def sample_choice(self, p):
        """Sample a choice at specific prices

        Args:
            p (torch.tensor): Prices

        Returns:
            int: User choice
        """
        q_f = self.Q_f(p)
        return np.random.choice(np.arange(self.K + 1), p=q_f.detach().numpy())

    def max_mean_demand_price(self):
        """Price that Maximum mean demand happens

        Returns:
            torch.tensor: Optimal price vector
        """

        def rhs(b_hat):
            try:
                res = (
                    1
                    / self.betas
                    * torch.exp(
                        -(1 + self.betas * b_hat)+ self.alphas
                        )
                )
                return res.sum()
            except:
                print("exception", (-(1 + self.betas * b_hat) + self.alphas, b_hat))

        def binary_search(start, end):
            if end - start < 1e-9:
                return start
            mid = (start + end) / 2
            if rhs(mid) - mid > 0:
                return binary_search(mid, end)
            else:
                return binary_search(start, mid)

        B = 1
        while True:
            
            if rhs(B) - B < 0:
                break
            B *= 1.5
        B0 = binary_search(0, B)
        
        best_price = torch.maximum(
            torch.tensor([0]), 1 / self.betas + B0)
        return best_price
        
    def max_mean_demand(self):
        """Maximum mean demand of the model

        Returns:
            torch.tensor: Maximum mean demand
        """
        p = self.max_mean_demand_price()
        return self.mean_demand(p)
    
    
    @classmethod
    def mle(
            cls,
            Is,
            Ps,
            K,
            alphas_s_old = None,
            betas_s_old = None,
            steps = None,
            lr = None,
            regularization=None):
        
        if (
            alphas_s_old is not None
            and betas_s_old is not None
        ):
            alphas_s = torch.autograd.Variable(
                alphas_s_old.detach().clone(), requires_grad=True
            )
            betas_s = torch.autograd.Variable(
                betas_s_old.detach().clone(), requires_grad=True
            )
        else:#  if parameters are None, then uniformly sampling K values for each 
            alphas_s = torch.autograd.Variable(torch.rand(K), requires_grad=True)
            betas_s = torch.autograd.Variable(torch.rand(K), requires_grad=True)
        
        if lr is None:
            lr = 1e-1
        if steps is None:
            steps = 100   
        optimizer = torch.optim.Adam([alphas_s, betas_s],lr=lr)
        loss = torch.nn.CrossEntropyLoss(reduction="sum")
        cnt = 0
        grad_norm = 10
        while cnt < steps:
           cnt += 1
           model_s = MultipleMNLModel(
               alphas_s, betas_s, 
               )
           U_f = model_s.U_f(Ps)
           ll = loss(U_f, Is)
           if regularization:
               
               ll += regularization * (
                   torch.norm(alphas_s) ** 2
                   + torch.norm(betas_s) ** 2
               )
           ll = ll / len(Is)
           ll.backward(retain_graph=True)
           grad_norm = torch.sqrt(
               torch.norm(alphas_s.grad) ** 2
               + torch.norm(betas_s.grad) ** 2
           )
           grad_norm = grad_norm.item()
           optimizer.step()
           optimizer.zero_grad()
          
        return (
            alphas_s.detach().clone(),
            betas_s.detach().clone(),
            grad_norm,
            )
    
    @classmethod
    def langevin_step(
            cls,
            Is,
            Ps,
            K,
            N_t,# Langevin step
            eta,
            psi_inverse,
            alphas_s_old=None,
            betas_s_old=None,
            regularization=None
            ):
        if (
            alphas_s_old is not None
            and betas_s_old is not None
            ):
            alphas_s = torch.autograd.Variable(
                alphas_s_old.detach().clone(), requires_grad=True
            )
            betas_s = torch.autograd.Variable(
                betas_s_old.detach().clone(), requires_grad=True
            )
        else:
            alphas_s = torch.autograd.Variable(torch.ones(K), requires_grad=True)
            betas_s = torch.autograd.Variable(torch.ones(K), requires_grad=True)
        loss = torch.nn.CrossEntropyLoss(reduction="sum")
        cnt = 0
        while cnt<N_t:
            cnt += 1
            model_s = MultipleMNLModel(
                alphas_s,betas_s,
                )
            U_f = model_s.U_f(Ps)
            ll = loss(U_f,Is)
            if regularization:
                    ll += regularization*(
                        torch.norm(alphas_s)**2
                        + torch.norm(betas_s) ** 2
                        )
            
            ll.backward(retain_graph=True)
            with torch.no_grad():
                grad_norm = torch.sqrt(
                    torch.norm(alphas_s.grad)**2
                    + torch.norm(betas_s.grad)**2
                    )/len(Is)
                grad_norm = grad_norm.item()
                
                alphas_s -= eta*alphas_s.grad + np.sqrt(
                    2*eta*psi_inverse
                    )*torch.randn(K)
                alphas_s.grad.zero_()
                
                betas_s -= eta*betas_s.grad + np.sqrt(
                    2*eta*psi_inverse
                    )*torch.randn(K)
                betas_s.grad.zero_()

        return (
            alphas_s.detach().clone(),
            betas_s.detach().clone(),
            grad_norm,
            )